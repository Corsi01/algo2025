"""
Single ensemble submission script for optimized multi-subject fMRI encoding models - OOD movies
Uses one ensemble (with text) and dummy language features for chaplin movies
"""
import matplotlib
matplotlib.use('Agg') 

import os
import numpy as np
from tqdm import tqdm
from submission_utils_ood import (
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
    
    # Load the visual features
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 'pca',
        'ood', 'visual', 'features_ood.npy')
    features_ood['saliency'] = np.load(stimuli_dir, allow_pickle=True).item()

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


def align_features_and_fmri_samples_ood_single(features_ood, root_data_dir, dummy_language,
                                               visual_extra_params=None, audio_extra_params=None):
    """
    Align the stimulus feature with the fMRI response samples for OOD movies,
    using dummy language features for chaplin movies.
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

                # Define the order of modalities to ensure consistent feature vector
                # Order: saliency, audio, audio2, language_pooled
                modality_order = ['audio', 'audio2', 'language_pooled', 'saliency']
                
                for mod in modality_order:
                    if mod not in features_ood.keys() and mod != 'language_pooled':
                        continue
                    
                    # All features use stimulus window logic
                    if mod in ['visual', 'audio', 'saliency', 'language_pooled', 'audio2']:
                        
                        # Standard window logic (same for all modalities)
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        
                        # Handle language_pooled for chaplin movies
                        if mod == 'language_pooled' and epi in ['chaplin1', 'chaplin2']:
                            # Use dummy language features replicated for the window
                            dummy_window = np.tile(dummy_language, (stimulus_window, 1))  # (7, 250)
                            f_combined = dummy_window.flatten()  # (1750,)
                            print(f"    Using dummy language features for {epi}, shape: {dummy_window.shape} -> {f_combined.shape}")
                        else:
                            # Normal processing for other modalities or non-chaplin movies
                            if idx_end > len(features_ood[mod][epi]):
                                idx_end = len(features_ood[mod][epi])
                                idx_start = idx_end - stimulus_window
                            
                            # Base window
                            f_base = features_ood[mod][epi][idx_start:idx_end]
                            f_combined = f_base.flatten()
                            
                            # Extra window for visual (saliency)
                            if mod == 'saliency' and visual_extra_params is not None:
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

def prepare_aligned_features_single(features_ood, root_data_dir, dummy_language):
    """
    Prepare aligned features for both standard and different configurations using single ensemble approach
    """
    print("Aligning features with optimized memory windows (single ensemble approach)...")
    
    # Standard features (for visual, dorsattn, multi networks)
    print("  1. Standard features (visual memory: hrf_delay=11, stimulus_window=5)")
    aligned_features_standard = align_features_and_fmri_samples_ood_single(
        features_ood, root_data_dir, dummy_language,
        visual_extra_params={'hrf_delay': 11, 'stimulus_window': 5}
    )
    
    # Different features (for sommot network) 
    print("  2. Different features (visual: hrf_delay=14, audio: hrf_delay=10, stimulus_window=5)")
    aligned_features_different = align_features_and_fmri_samples_ood_single(
        features_ood, root_data_dir, dummy_language,
        visual_extra_params={'hrf_delay': 14, 'stimulus_window': 5},
        audio_extra_params={'hrf_delay': 10, 'stimulus_window': 5}
    )
    
    # Show alignment results
    print("\nAlignment completed! Feature shapes:")
    for sub in ['sub-01', 'sub-02', 'sub-03', 'sub-05']:
        example_episode = list(aligned_features_standard[sub].keys())[0]
        standard_shape = aligned_features_standard[sub][example_episode].shape
        different_shape = aligned_features_different[sub][example_episode].shape
        print(f"  {sub}: standard {standard_shape}, different {different_shape}")
    
    return aligned_features_standard, aligned_features_different


def generate_predictions_single(ensemble, aligned_features_standard, aligned_features_different):
    """
    Generate predictions for all subjects and episodes using single ensemble model
    """
    print("\nGenerating predictions with single ensemble...")
    submission_predictions = {}
    
    desc = "Predicting fMRI responses with single optimized ensemble model"
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
            
            if epi in ['chaplin1', 'chaplin2']:
                print(f"    Using dummy language features for {epi}")
            
            # Make prediction using the single ensemble
            fmri_pred = make_prediction(
                ensemble=ensemble,
                features_standard=feat_standard_epi,
                features_different=feat_different_epi,
                subject_id=subject_number
            )
            
            # Save the prediction
            submission_predictions[sub][epi] = fmri_pred
    
    return submission_predictions


def main():
    """
    Main function to generate optimized submission with single ensemble approach
    """
    root_data_dir = 'data'
    
    print("="*60)
    print("OPTIMIZED MULTI-SUBJECT OOD SUBMISSION WITH SINGLE ENSEMBLE")
    print("="*60)
    
    # 1. Load optimized model configurations for single ensemble
    print("\n1. Loading optimized model configurations...")
    model_configs, parcel_lists = get_optimized_model_configs('optimize_models/')
    
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
    aligned_features_standard, aligned_features_different = prepare_aligned_features_single(
        features_ood, root_data_dir, dummy_language
    )
    
    # 5. Create single ensemble model (with text features)
    print("\n5. Creating single ensemble model...")
    ensemble = create_ensemble_from_optimized_models(
        models_dir='optimize_models/',
        model_configs=model_configs,
        parcel_lists=parcel_lists
    )
    
    # 6. Generate predictions using single ensemble for all movies
    print("\n6. Generating predictions with single ensemble approach...")
    submission_predictions = generate_predictions_single(
        ensemble, aligned_features_standard, aligned_features_different
    )
    
    # 7. Save submission files
    print("\n7. Saving submission files...")
    import zipfile
    
    save_dir = 'data/submission/ood/multi_subject/optimized_single/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the predicted fMRI dictionary as a .npy file
    output_file = os.path.join(save_dir, "fmri_predictions_ood_single.npy")
    np.save(output_file, submission_predictions)
    print(f"Predictions saved to: {output_file}")
    
    # Zip the saved file for submission
    zip_file = os.path.join(save_dir, "fmri_predictions_ood_single.zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Submission file zipped as: {zip_file}")
    
    # 8. Display summary
    print("\n" + "="*60)
    print("SINGLE ENSEMBLE SUBMISSION GENERATION COMPLETED!")
    print("="*60)
    
    # Count movies by type
    total_episodes = 0
    normal_episodes = 0
    chaplin_episodes = 0
    
    for subject, episodes_dict in submission_predictions.items():
        n_episodes = len(episodes_dict)
        total_episodes += n_episodes
        
        for episode in episodes_dict.keys():
            if episode in ['chaplin1', 'chaplin2']:
                chaplin_episodes += 1
            else:
                normal_episodes += 1
        
        print(f"{subject}: {n_episodes} episodes")
        
        # Show first episode shape as example
        first_epi = list(episodes_dict.keys())[0]
        shape = episodes_dict[first_epi].shape
        print(f"  Example shape: {shape}")
    
    print(f"\nTotal episodes predicted: {total_episodes}")
    print(f"Normal episodes (real language): {normal_episodes//4}")     # Divide by 4 subjects
    print(f"Chaplin episodes (dummy language): {chaplin_episodes//4}")  # Divide by 4 subjects
    
    print(f"\nSingle ensemble configurations:")
    for network in ['visual', 'dorsattn', 'sommot', 'multi']:
        config = model_configs[network]
        print(f"  {network}: input_dim={config['input_dim']}, embedding_dim={config['embedding_dim']}, "
              f"hidden_dim={config['hidden_dim']}")
    
    return submission_predictions


if __name__ == "__main__":
    submission_predictions = main()
