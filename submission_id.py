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
from utils.data_utils import atlas_schaefer


def load_stimulus_features_friends_s7(root_data_dir):
    """
    Load the stimulus features of all modalities (visual + audio + language) for
    Friends season 7.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.

    Returns
    -------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.

    """

    features_friends_s7 = {}

    ### Load the audio features ###
    stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features',
        'pca', 'friends_movie10', 'audio2', 'features_test.npy')
    features_friends_s7['audio'] = np.load(stimuli_dir,
        allow_pickle=True).item()
    
    stimuli_dir = os.path.join('results', 'stimulus_features',
        'pca_new2', 'friends_movie10', 'audio', 'features_test.npy')
    features_friends_s7['audio2'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the language features ###
    stimuli_dir = os.path.join('results',  'stimulus_features', 'pca_new3', 
        'friends_movie10', 'language', 'features_test.npy')
    features_friends_s7['language_pooled'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the saliency (visual) features ###
    stimuli_dir = os.path.join(root_data_dir, 'results',  'stimulus_features',  
        'pca', 'friends_movie10', 'saliency_statistical', 'features_test.npy')
    features_friends_s7['saliency'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    return features_friends_s7


def align_features_and_fmri_samples_friends_s7(features_friends_s7,
    root_data_dir, visual_extra_params=None, audio_extra_params=None):
    """
    Align the stimulus feature with the fMRI response samples for Friends season
    7 episodes, later used to predict the fMRI responses for challenge
    submission.

    Parameters
    ----------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.
    root_data_dir : str
        Root data directory.
    visual_extra_params : dict, optional
        Extra parameters for visual modalities (saliency).
        Should contain 'hrf_delay' and 'stimulus_window' keys.
    audio_extra_params : dict, optional
        Extra parameters for audio modalities (audio, audio2).
        Should contain 'hrf_delay' and 'stimulus_window' keys.

    Returns
    -------
    aligned_features_friends_s7 : dict
        Aligned stimulus features for each subject and Friends season 7 episode.

    """

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    hrf_delay = 2
    stimulus_window = 7

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir,
            'fmri', f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            features_epi = []

            ### Loop over fMRI samples ###
            for s in range(samples):
                # Empty variable containing the stimulus features of all
                # modalities for each sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features_friends_s7.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio' or mod == 'saliency' or mod == 'language_pooled' or mod == 'audio2':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = idx_end - stimulus_window
                        
                        f_base = features_friends_s7[mod][epi][idx_start:idx_end]
                        f_combined = f_base.flatten()
                        
                        if mod == 'saliency' and visual_extra_params is not None:
                            extra_hrf_delay = visual_extra_params['hrf_delay']
                            extra_stimulus_window = visual_extra_params['stimulus_window']
                            
                            if s < (extra_stimulus_window + extra_hrf_delay):
                                extra_idx_start = 0
                                extra_idx_end = extra_idx_start + extra_stimulus_window
                            else:
                                extra_idx_start = s - extra_hrf_delay - extra_stimulus_window + 1
                                extra_idx_end = extra_idx_start + extra_stimulus_window
                            
                            if extra_idx_start < 0:
                                n_zeros_needed = abs(extra_idx_start)
                                actual_start = 0
                                actual_end = min(extra_idx_end, len(features_friends_s7[mod][epi]))
                                
                                if actual_end > actual_start:
                                    f_available = features_friends_s7[mod][epi][actual_start:actual_end]
                                else:
                                    f_available = np.empty((0, features_friends_s7[mod][epi].shape[1]))
                                
                                feature_dim = features_friends_s7[mod][epi].shape[1]
                                f_zeros = np.zeros((n_zeros_needed, feature_dim))
                                f_extra = np.concatenate([f_zeros, f_available], axis=0)
                                
                                if len(f_extra) > extra_stimulus_window:
                                    f_extra = f_extra[:extra_stimulus_window]
                                elif len(f_extra) < extra_stimulus_window:
                                    missing = extra_stimulus_window - len(f_extra)
                                    f_extra = np.concatenate([f_extra, np.zeros((missing, feature_dim))], axis=0)
                            else:
                                if extra_idx_end > len(features_friends_s7[mod][epi]):
                                    extra_idx_end = len(features_friends_s7[mod][epi])
                                    extra_idx_start = max(0, extra_idx_end - extra_stimulus_window)
                                f_extra = features_friends_s7[mod][epi][extra_idx_start:extra_idx_end]
                            
                            f_combined = np.concatenate([f_combined, f_extra.flatten()])
                        
                        elif (mod == 'audio' or mod == 'audio2') and audio_extra_params is not None:
                            extra_hrf_delay = audio_extra_params['hrf_delay']
                            extra_stimulus_window = audio_extra_params['stimulus_window']
                            
                            if s < (extra_stimulus_window + extra_hrf_delay):
                                extra_idx_start = 0
                                extra_idx_end = extra_idx_start + extra_stimulus_window
                            else:
                                extra_idx_start = s - extra_hrf_delay - extra_stimulus_window + 1
                                extra_idx_end = extra_idx_start + extra_stimulus_window
                            
                            if extra_idx_start < 0:
                                
                                n_zeros_needed = abs(extra_idx_start)
                                actual_start = 0
                                actual_end = min(extra_idx_end, len(features_friends_s7[mod][epi]))
                                
                                if actual_end > actual_start:
                                    f_available = features_friends_s7[mod][epi][actual_start:actual_end]
                                else:
                                    f_available = np.empty((0, features_friends_s7[mod][epi].shape[1]))
                                
                                feature_dim = features_friends_s7[mod][epi].shape[1]
                                f_zeros = np.zeros((n_zeros_needed, feature_dim))
                                f_extra = np.concatenate([f_zeros, f_available], axis=0)
                                
                                if len(f_extra) > extra_stimulus_window:
                                    f_extra = f_extra[:extra_stimulus_window]
                                elif len(f_extra) < extra_stimulus_window:
                                    missing = extra_stimulus_window - len(f_extra)
                                    f_extra = np.concatenate([f_extra, np.zeros((missing, feature_dim))], axis=0)
                            else:
                                if extra_idx_end > len(features_friends_s7[mod][epi]):
                                    extra_idx_end = len(features_friends_s7[mod][epi])
                                    extra_idx_start = max(0, extra_idx_end - extra_stimulus_window)
                                f_extra = features_friends_s7[mod][epi][extra_idx_start:extra_idx_end]
                            
                            f_combined = np.concatenate([f_combined, f_extra.flatten()])
                        
                        f_all = np.append(f_all, f_combined)

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                            f = features_friends_s7[mod][epi][-1,:]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        f_all = np.append(f_all, f.flatten())

                ### Append the stimulus features of all modalities for this sample ###
                features_epi.append(f_all)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = np.asarray(
                features_epi, dtype=np.float32)

    return aligned_features_friends_s7


def prepare_aligned_features_friends_s7(features_friends_s7, root_data_dir):
    """
    Prepare aligned features for both standard and different configurations (different input memory for different networks)
    """
    print("Aligning features with optimized memory windows ...")
    
    # Standard features (for visual, dorsattn, multi networks)
    print("  1. Standard features (visual memory: hrf_delay=11, stimulus_window=5)")
    aligned_features_standard = align_features_and_fmri_samples_friends_s7(
        features_friends_s7, root_data_dir,
        visual_extra_params={'hrf_delay': 11, 'stimulus_window': 5}
    )
    
    # Different features (for sommot network) 
    print("  2. Different features (visual: hrf_delay=14, audio: hrf_delay=10, stimulus_window=5)")
    aligned_features_different = align_features_and_fmri_samples_friends_s7(
        features_friends_s7, root_data_dir,
        visual_extra_params={'hrf_delay': 14, 'stimulus_window': 5},
        audio_extra_params={'hrf_delay': 10, 'stimulus_window': 5}
    )
    
    return aligned_features_standard, aligned_features_different


def generate_predictions_friends_s7(ensemble, aligned_features_standard, aligned_features_different):
    """
    Generate predictions for all subjects and episodes using optimized ensemble
    
    Parameters:
    - ensemble: optimized ensemble model
    - aligned_features_standard: Standard features for all subjects/episodes
    - aligned_features_different: Different features for all subjects/episodes
    """
    print("\nGenerating predictions with optimized ensemble...")
    submission_predictions = {}
   
    desc = "Predicting fMRI responses with optimized ensemble model"
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
            
            # Make prediction using the optimized ensemble
            fmri_pred = make_prediction(
                ensemble=ensemble,
                features_standard=feat_standard_epi,
                features_different=feat_different_epi,
                subject_id=subject_number
            )
           
            # Save the prediction
            submission_predictions[sub][epi] = fmri_pred
            
            # Log prediction shape for verification
            print(f"    Prediction shape: {fmri_pred.shape}")
   
    return submission_predictions


def main():
    
    root_data_dir = 'data'
    
    print("="*60)
    print("OPTIMIZED FRIENDS S7 SUBMISSION")
    print("="*60)
    
    # 1. Load optimized model configurations
    print("\n1. Loading optimized model configurations...")
    model_configs_with_text, parcel_lists = get_optimized_model_configs('optimize_models/model_best/')
    
    # 2. Load stimulus features for Friends S7
    print("\n2. Loading stimulus features for Friends season 7...")
    features_friends_s7 = load_stimulus_features_friends_s7(root_data_dir)
    
    print(f"Loaded modalities: {list(features_friends_s7.keys())}")
    for modality, features_dict in features_friends_s7.items():
        n_episodes = len(features_dict)
        example_episode = list(features_dict.keys())[0]
        example_shape = features_dict[example_episode].shape
        print(f"  {modality}: {n_episodes} episodes, example shape: {example_shape}")
    
    # 3. Prepare aligned features with optimized memory windows
    print("\n3. Preparing aligned features...")
    aligned_features_standard, aligned_features_different = prepare_aligned_features_friends_s7(
        features_friends_s7, root_data_dir
    )
    
    # 4. Create optimized ensemble model
    print("\n4. Creating optimized ensemble model...")
    ensemble_with_text = create_ensemble_from_optimized_models(
        models_dir='optimize_models/model_best/',
        model_configs=model_configs_with_text,
        parcel_lists=parcel_lists
    )
    
    # 5. Generate predictions using optimized ensemble
    print("\n5. Generating predictions with optimized ensemble...")
    submission_predictions = generate_predictions_friends_s7(
        ensemble_with_text, aligned_features_standard, aligned_features_different
    )
    
    # 6. Save submission files
    print("\n6. Saving submission files...")
    import zipfile
    
    save_dir = 'data/submission/friends_s7/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the predicted fMRI dictionary as a .npy file
    output_file = os.path.join(save_dir, "fmri_predictions_friends_s7.npy")
    np.save(output_file, submission_predictions)
    print(f"Predictions saved to: {output_file}")
    
    # Zip the saved file for submission
    zip_file = os.path.join(save_dir, "fmri_predictions_friends_s7.zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Submission file zipped as: {zip_file}")
    return submission_predictions


if __name__ == "__main__":
    submission_predictions = main()
