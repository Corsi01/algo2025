import matplotlib
matplotlib.use('Agg') 

import os
import numpy as np
from tqdm import tqdm
from utils.submission_utils_ood import (
    get_optimized_model_configs,
    create_ensemble_from_optimized_models,
    make_prediction
)


def load_stimulus_features_ood(root_data_dir):
    """
    Load the stimulus features of all modalities (visual + audio + language) for
    the OOD movies.
    """
    features_ood = {}

    # Load the audio features
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'audio_emo', 'features_ood.npy')
    features_ood['audio'] = np.load(stimuli_dir, allow_pickle=True).item()
    
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'audio', 'features_ood.npy')
    features_ood['audio2'] = np.load(stimuli_dir, allow_pickle=True).item()

    # Load the language features
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'language', 'features_ood.npy')
    features_ood['language_pooled'] = np.load(stimuli_dir, allow_pickle=True).item()
    
    # Load the language_multi features
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'language_multi', 'features_ood.npy')
    features_ood['language_multi'] = np.load(stimuli_dir, allow_pickle=True).item()
    
    # Load the visual features
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'visual', 'features_ood.npy')
    features_ood['saliency'] = np.load(stimuli_dir, allow_pickle=True).item()
    
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'visual_videomae2', 'features_ood.npy')
    features_ood['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    return features_ood


def load_dummy_language_features(root_data_dir):
    """
    Load dummy language features from Friends training data for chaplin movies
    """
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'friends_movie10', 'language', 'features_train.npy')
    features_train = np.load(stimuli_dir, allow_pickle=True).item()
    
    # Get the first language feature vector as dummy (shape: 250)
    dummy_language = features_train['s01e01a'][0]
    
    print(f"Loaded dummy language features with shape: {dummy_language.shape}")
    return dummy_language


def align_features_and_fmri_samples_ood_triple(features_ood, root_data_dir, dummy_language,
                                             visual_extra_params=None, audio_extra_params=None):
    """
    Align the stimulus feature with the fMRI response samples for OOD movies,
    using dummy language features for chaplin movies and language_multi for passepartout.
    Uses visual instead of saliency for subjects 2,3,5 in chaplin1/2 and planetearth1/2.
    """
    aligned_features_ood = {}
    hrf_delay = 2
    stimulus_window = 7

    # Loop over subjects
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_ood[f'sub-0{sub}'] = {}

        # Load the OOD movie fMRI samples
        samples_dir = os.path.join(root_data_dir, 
            'fmri', f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_ood_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        # Loop over the OOD movies
        for epi, samples in fmri_samples.items():
            features_epi = []

            # Loop over fMRI samples
            for s in range(samples):
                f_all = np.empty(0)

                # Loop across modalities
                for mod in features_ood.keys():
                    
                    # Skip language_multi se non è il caso specifico
                    if mod == 'language_multi' and epi not in ['passepartout1', 'passepartout2']:
                        continue

                    # Determine which visual modality to use
                    visual_mod_to_use = None
                    if mod == 'saliency':
                        # Use saliency for subject 1 always, or for subjects 2,3,5 in all movies except chaplin/planetearth
                        if sub == 1 or epi not in ['chaplin1', 'chaplin2', 'planetearth1', 'planetearth2']:
                            visual_mod_to_use = 'saliency'
                        else:
                            # Skip saliency for subjects 2,3,5 in chaplin/planetearth (visual will be used instead)
                            continue
                    elif mod == 'visual':
                        # Use visual for subjects 2,3,5 in chaplin/planetearth movies
                        if sub in [2, 3, 5] and epi in ['chaplin1', 'chaplin2', 'planetearth1', 'planetearth2']:
                            visual_mod_to_use = 'visual'
                        else:
                            # Skip visual for other cases
                            continue

                    # All features now use stimulus window logic
                    if mod in ['visual', 'audio', 'saliency', 'language_pooled', 'audio2', 'language_multi']:
                        # Skip language_pooled for passepartout movies (use language_multi instead)
                        if mod == 'language_pooled' and epi in ['passepartout1', 'passepartout2']:
                            continue
                        
                        # Standard window logic (same for all modalities)
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        
                        # Handle language_pooled for chaplin movies with dummy features
                        if mod == 'language_pooled' and epi in ['chaplin1', 'chaplin2']:
                            # Use dummy language features replicated for the window
                            dummy_window = np.tile(dummy_language, (stimulus_window, 1))  # (7, 250)
                            f_combined = dummy_window.flatten()  # (1750,)
                            print(f"    Using dummy language features for {epi}, shape: {dummy_window.shape} -> {f_combined.shape}")
                        else:
                            # Normal processing for other modalities or non-chaplin movies
                            # Use the determined visual modality if applicable
                            current_mod = visual_mod_to_use if visual_mod_to_use else mod
                            
                            if idx_end > len(features_ood[current_mod][epi]):
                                idx_end = len(features_ood[current_mod][epi])
                                idx_start = idx_end - stimulus_window
                            
                            # Base window
                            f_base = features_ood[current_mod][epi][idx_start:idx_end]
                            f_combined = f_base.flatten()
                            
                            # Extra window for visual modalities (both saliency and visual)
                            if (mod == 'saliency' or mod == 'visual') and visual_extra_params is not None:
                                extra_hrf_delay = visual_extra_params['hrf_delay']
                                extra_stimulus_window = visual_extra_params['stimulus_window']
                                
                                # Calculate indices for extra window
                                if s < (extra_stimulus_window + extra_hrf_delay):
                                    extra_idx_start = 0
                                    extra_idx_end = extra_idx_start + extra_stimulus_window
                                else:
                                    extra_idx_start = s - extra_hrf_delay - extra_stimulus_window + 1
                                    extra_idx_end = extra_idx_start + extra_stimulus_window
                                
                                # Handle edge cases
                                if extra_idx_start < 0:
                                    # Padding with zeros at the beginning
                                    n_zeros_needed = abs(extra_idx_start)
                                    actual_start = 0
                                    actual_end = min(extra_idx_end, len(features_ood[current_mod][epi]))
                                    
                                    if actual_end > actual_start:
                                        f_available = features_ood[current_mod][epi][actual_start:actual_end]
                                    else:
                                        f_available = np.empty((0, features_ood[current_mod][epi].shape[1]))
                                    
                                    feature_dim = features_ood[current_mod][epi].shape[1]
                                    f_zeros = np.zeros((n_zeros_needed, feature_dim))
                                    f_extra = np.concatenate([f_zeros, f_available], axis=0)
                                    
                                    if len(f_extra) > extra_stimulus_window:
                                        f_extra = f_extra[:extra_stimulus_window]
                                    elif len(f_extra) < extra_stimulus_window:
                                        missing = extra_stimulus_window - len(f_extra)
                                        f_extra = np.concatenate([f_extra, np.zeros((missing, feature_dim))], axis=0)
                                else:
                                    if extra_idx_end > len(features_ood[current_mod][epi]):
                                        extra_idx_end = len(features_ood[current_mod][epi])
                                        extra_idx_start = max(0, extra_idx_end - extra_stimulus_window)
                                    f_extra = features_ood[current_mod][epi][extra_idx_start:extra_idx_end]
                                
                                f_combined = np.concatenate([f_combined, f_extra.flatten()])
                            
                            # Extra window for audio (audio, audio2)
                            elif (mod == 'audio' or mod == 'audio2') and audio_extra_params is not None:
                                extra_hrf_delay = audio_extra_params['hrf_delay']
                                extra_stimulus_window = audio_extra_params['stimulus_window']
                                
                                # Similar logic as visual extra window
                                if s < (extra_stimulus_window + extra_hrf_delay):
                                    extra_idx_start = 0
                                    extra_idx_end = extra_idx_start + extra_stimulus_window
                                else:
                                    extra_idx_start = s - extra_hrf_delay - extra_stimulus_window + 1
                                    extra_idx_end = extra_idx_start + extra_stimulus_window
                                
                                if extra_idx_start < 0:
                                    n_zeros_needed = abs(extra_idx_start)
                                    actual_start = 0
                                    actual_end = min(extra_idx_end, len(features_ood[mod][epi]))
                                    
                                    if actual_end > actual_start:
                                        f_available = features_ood[mod][epi][actual_start:actual_end]
                                    else:
                                        f_available = np.empty((0, features_ood[mod][epi].shape[1]))
                                    
                                    feature_dim = features_ood[mod][epi].shape[1]
                                    f_zeros = np.zeros((n_zeros_needed, feature_dim))
                                    f_extra = np.concatenate([f_zeros, f_available], axis=0)
                                    
                                    if len(f_extra) > extra_stimulus_window:
                                        f_extra = f_extra[:extra_stimulus_window]
                                    elif len(f_extra) < extra_stimulus_window:
                                        missing = extra_stimulus_window - len(f_extra)
                                        f_extra = np.concatenate([f_extra, np.zeros((missing, feature_dim))], axis=0)
                                else:
                                    if extra_idx_end > len(features_ood[mod][epi]):
                                        extra_idx_end = len(features_ood[mod][epi])
                                        extra_idx_start = max(0, extra_idx_end - extra_stimulus_window)
                                    f_extra = features_ood[mod][epi][extra_idx_start:extra_idx_end]
                                
                                f_combined = np.concatenate([f_combined, f_extra.flatten()])
                        
                        f_all = np.append(f_all, f_combined)

                # Append the stimulus features of all modalities for this sample
                features_epi.append(f_all)

            # Add the episode stimulus features to the features dictionary
            aligned_features_ood[f'sub-0{sub}'][epi] = np.asarray(features_epi, dtype=np.float32)

    return aligned_features_ood


def prepare_aligned_features_triple(features_ood, root_data_dir, dummy_language):
    """
    Prepare aligned features for both standard and different configurations (different input memory for different networks)
    """
    print("Aligning features with optimized memory windows ...")
    
    # Standard features (for visual, dorsattn, multi networks)
    print("  1. Standard features (visual memory: hrf_delay=11, stimulus_window=5)")
    aligned_features_standard = align_features_and_fmri_samples_ood_triple(
        features_ood, root_data_dir, dummy_language,
        visual_extra_params={'hrf_delay': 11, 'stimulus_window': 5}
    )
    
    # Different features (for sommot network) 
    print("  2. Different features (visual: hrf_delay=14, audio: hrf_delay=10, stimulus_window=5)")
    aligned_features_different = align_features_and_fmri_samples_ood_triple(
        features_ood, root_data_dir, dummy_language,
        visual_extra_params={'hrf_delay': 14, 'stimulus_window': 5},
        audio_extra_params={'hrf_delay': 10, 'stimulus_window': 5}
    )
    
    return aligned_features_standard, aligned_features_different


def generate_predictions_triple(ensemble_with_text, ensemble_language_multi, ensemble_with_text_videomae, aligned_features_standard, aligned_features_different):
    """
    Generate predictions for all subjects and episodes using triple ensemble approach
    
    Parameters:
    - ensemble_with_text: Standard ensemble with text features
    - ensemble_language_multi: Ensemble with multilingual features (for passepartout)
    - ensemble_with_text_videomae: VideoMAE ensemble (for subjects 2,3,5 on chaplin & planetearth)
    - aligned_features_standard: Standard features for all subjects/episodes
    - aligned_features_different: Different features for all subjects/episodes
    """
    print("\nGenerating predictions with triple ensemble...")
    submission_predictions = {}
   
    # Define which subjects and episodes use VideoMAE ensemble
    videomae_targets = {
        2: ['chaplin1', 'chaplin2', 'planetearth1', 'planetearth2'],
        3: ['chaplin1', 'chaplin2', 'planetearth1', 'planetearth2'], 
        5: ['chaplin1', 'chaplin2', 'planetearth1', 'planetearth2']
    }
   
    desc = "Predicting fMRI responses with triple optimized ensemble models"
    for sub, features_standard in tqdm(aligned_features_standard.items(), desc=desc):
       
        submission_predictions[sub] = {}
       
        # Extract subject number from key (e.g., 'sub-01' -> 1)
        subject_number = int(sub.split('-')[1])
        print(f"\nProcessing {sub} (subject {subject_number})")
        # Get corresponding different features for this subject
        features_different = aligned_features_different[sub]
       
        # Process each episode
        for epi, feat_standard_epi in features_standard.items():
            # Get corresponding different features for this episode
            feat_different_epi = features_different[epi]
           
            print(f"  Episode {epi}: standard {feat_standard_epi.shape}, "
                  f"different {feat_different_epi.shape}")
           
            # Select appropriate ensemble based on subject and episode
            if subject_number in videomae_targets and epi in videomae_targets[subject_number]:
                # Use VideoMAE ensemble for subjects 2,3,5 on chaplin and planetearth
                ensemble = ensemble_with_text_videomae
                ensemble_type = "VideoMAE ensemble"
                print(f"    Using {ensemble_type} for subject {subject_number} on {epi}")
                
            elif epi in ['passepartout1', 'passepartout2']:
                # Use ensemble trained WITH MULTILINGUAL text features
                ensemble = ensemble_language_multi
                ensemble_type = "MULTILINGUAL text ensemble"
                print(f"    Using {ensemble_type}")
                
            else:
                # Use ensemble trained WITH text features (default case)
                ensemble = ensemble_with_text
                ensemble_type = "standard text ensemble"
                if epi in ['chaplin1', 'chaplin2']:
                    print(f"    Using {ensemble_type} (dummy language for {epi})")
                else:
                    print(f"    Using {ensemble_type}")
           
            # Make prediction using the selected ensemble
            fmri_pred = make_prediction(
                ensemble=ensemble,
                features_standard=feat_standard_epi,
                features_different=feat_different_epi,
                subject_id=subject_number
            )
           
            # Save the prediction
            submission_predictions[sub][epi] = fmri_pred
            
            # Log prediction shape for verification
            print(f"Prediction shape: {fmri_pred.shape}")
   
    return submission_predictions


def main():
    
    root_data_dir = 'data'
    
    print("="*60)
    print("OPTIMIZED MULTI-SUBJECT OOD SUBMISSION")
    print("="*60)
    
    # 1. Load optimized model configurations for BOTH ensembles
    print("\n1. Loading optimized model configurations...")
    print("  a) Loading models WITH text features:")
    model_configs_with_text, parcel_lists = get_optimized_model_configs('models/best_model/')
    
    print("  b) Loading models WITH MULTILINGUAL text features:")
    model_configs_language_multi, _ = get_optimized_model_configs('models/model_multilingual/')
    
    print("  c) Loading models WITH text features + VIDEOMAE:")
    model_configs_with_text_videomae, parcel_lists = get_optimized_model_configs('models/model_videomae2/')
    
    # 2. Load stimulus features
    print("\n2. Loading stimulus features for OOD movies...")
    features_ood = load_stimulus_features_ood(root_data_dir)
    
    print(f"Loaded modalities: {list(features_ood.keys())}")
    for modality, features_dict in features_ood.items():
        n_episodes = len(features_dict)
        example_episode = list(features_dict.keys())[0]
        example_shape = features_dict[example_episode].shape
        print(f"  {modality}: {n_episodes} episodes, example shape: {example_shape}")
    
    # 3. Load dummy language features for chaplin movies
    print("\n3. Loading dummy language features...")
    dummy_language = load_dummy_language_features(root_data_dir)
    
    # 4. Prepare aligned features with optimized memory windows
    print("\n4. Preparing aligned features...")
    aligned_features_standard, aligned_features_different = prepare_aligned_features_triple(
        features_ood, root_data_dir, dummy_language
    )
    
    # 5. Create BOTH ensemble models
    print("\n5. Creating ensemble models...")
    print("  a) Creating ensemble WITH text features:")
    ensemble_with_text = create_ensemble_from_optimized_models(
        models_dir='optimize_models/',
        model_configs=model_configs_with_text,
        parcel_lists=parcel_lists
    )
    
    print("  b) Creating ensemble WITH MULTILINGUAL text features:")
    ensemble_language_multi = create_ensemble_from_optimized_models(
        models_dir='optimize_models_language_multi/',
        model_configs=model_configs_language_multi,
        parcel_lists=parcel_lists
    )
    
    print("\n5. Creating ensemble models...")
    print("  c) Creating ensemble WITH text features + VIDEOMAE:")
    ensemble_with_text_videomae = create_ensemble_from_optimized_models(
        models_dir='optimize_models_videomae2/',
        model_configs=model_configs_with_text_videomae,
        parcel_lists=parcel_lists
    )
    
    # 6. Generate predictions using appropriate ensemble for each movie
    print("\n6. Generating predictions with triple ensemble approach...")
    submission_predictions = generate_predictions_triple(
        ensemble_with_text, ensemble_language_multi, ensemble_with_text_videomae, aligned_features_standard, aligned_features_different
    )
    
    # 7. Save submission files
    print("\n7. Saving submission files...")
    import zipfile
    
    save_dir = 'data/submission/oodmulti_subject/mixed/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the predicted fMRI dictionary as a .npy file
    output_file = os.path.join(save_dir, "fmri_predictions_ood.npy")
    np.save(output_file, submission_predictions)
    print(f"Predictions saved to: {output_file}")
    
    # Zip the saved file for submission
    zip_file = os.path.join(save_dir, "fmri_predictions_ood.zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Submission file zipped as: {zip_file}")
    
    
    return submission_predictions


if __name__ == "__main__":
    submission_predictions = main()
