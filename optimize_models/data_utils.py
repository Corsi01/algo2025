import os
import h5py
import numpy as np
import pandas as pd
import random
import torch
from scipy.stats import pearsonr
import nibabel as nib
from nilearn import plotting, datasets
from nilearn.maskers import NiftiLabelsMasker


root_data_dir = '../data'
modality = 'all'



def load_stimulus_features(root_data_dir, modality):
    
    features = {}

    
    
    ### Load the audio features ###
    if modality == 'audio' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features',
            'pca', 'friends_movie10', 'audio_emo', 'features_train.npy')
        features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()
        
    ### Load the language features ###
    if modality == 'audio2' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 
            'pca', 'friends_movie10', 'audio', 'features_train.npy')
        features['audio2'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the language features ###
    if modality == 'language' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir ,'results', 'stimulus_features', 
            'pca', 'friends_movie10', 'language', 'features_train.npy')
        features['language_pooled'] = np.load(stimuli_dir, allow_pickle=True).item()
        
    if modality == 'saliency' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'results', 'stimulus_features', 
            'pca', 'friends_movie10', 'visual', 'features_train.npy')
        features['saliency'] = np.load(stimuli_dir, allow_pickle=True).item()
    
    return features


def load_fmri(root_data_dir, subject):
 
    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir,    #### removed 'algonauts_2025.competitrs'
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir,   #### removed 'algonauts_2025.competitrs'
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    return fmri

def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies):

    ### Empty data variables ###
    aligned_features = []
    aligned_fmri = np.empty((0,1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[:len(id)]]

        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split = fmri[split]
            # Exclude the first and last fMRI samples, AND the first hrf_delay samples
            # to ensure proper temporal alignment
            fmri_split = fmri_split[excluded_samples_start + hrf_delay:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):
                # Empty variable containing the stimulus features of all
                # modalities for each fMRI sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features.keys():

                    ### Visual and audio features ###
                    # Model each fMRI sample using the N stimulus feature samples 
                    # up to the corresponding stimulus time (accounting for proper indexing)
                    if mod == 'saliency' or mod == 'audio' or mod == 'language_pooled' or mod == 'audio2' or mod == 'emotion':
                        # Calculate the stimulus indices for this fMRI sample
                        # s represents the fMRI sample index in the trimmed array
                        # The corresponding stimulus time is s + excluded_samples_start
                        stimulus_time = s + excluded_samples_start
                        idx_start = stimulus_time - stimulus_window + 1
                        idx_end = stimulus_time + 1
                        
                        # Handle case where we need samples before the beginning
                        if idx_start < 0:
                            idx_start = 0
                            idx_end = stimulus_window
                        
                        # Handle case where we need samples beyond the end
                        if idx_end > len(features[mod][split]):
                            idx_end = len(features[mod][split])
                            idx_start = max(0, idx_end - stimulus_window)
                        
                        f = features[mod][split][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample
                    elif mod == 'language':
                        # The stimulus time corresponding to this fMRI sample
                        stimulus_time = s + excluded_samples_start
                        idx = stimulus_time
                        
                        # Handle case where we need a sample beyond the end
                        if idx >= len(features[mod][split]):
                            idx = len(features[mod][split]) - 1
                        
                        f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())

                 ### Append the stimulus features of all modalities for this sample ###
                aligned_features.append(f_all)

    ### Convert the aligned features to a numpy array ###
    aligned_features = np.asarray(aligned_features, dtype=np.float32)

    ### Output ###
    return aligned_features, aligned_fmri

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    subject : int
        Subject number used to train and validate the encoding model.
    modality : str
        Feature modality used to train and validate the encoding model.

    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    ### Map the prediction accuracy onto a 3D brain atlas for plotting ###
    atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join('../data',   ###### removed 'algonauts_2025.competitors'
        'fmri', f'sub-0{subject}', 'atlas', atlas_file)
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: " + str(mean_encoding_accuracy)
    display = plotting.plot_glass_brain(
        encoding_accuracy_nii,
        display_mode="lyrz",
        cmap='hot_r',
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title
    )
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)
    
    # Salva il plot
    #import os
    os.makedirs('results', exist_ok=True)
    save_path = f'results/encoding_accuracy_sub{subject}_{modality}.png'
    display.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    
    plotting.show()
    
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    normalized_vector = vector / norm
    return normalized_vector

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atlas_schaefer():
    
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    labels = atlas['labels']

    labels_str = [label.decode('utf-8') for label in labels]
    networks = [label.split('_')[2] for label in labels_str]
    hemispheres = [label.split('_')[1] for label in labels_str]


    parcel_info = {
        'parcel_id': list(range(0, 1000)),
        'label': labels_str,
        'network_yeo7': networks,
        'hemisphere': hemispheres
    }

    df = pd.DataFrame(parcel_info)
    
    return df


def analyze_encoding_accuracy(fmri_val, fmri_val_pred):

    # Compute Correlation
    encoding_accuracy = np.zeros(fmri_val.shape[1], dtype=np.float32)
    for i in range(len(encoding_accuracy)):
        encoding_accuracy[i] = pearsonr(fmri_val[:, i], fmri_val_pred[:, i])[0]

    #  Get both atlas labels (same ROI ordering)
    atlas_7 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    atlas_17 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17)
    labels_7 = atlas_7['labels']
    labels_17 = atlas_17['labels']

    # Extract infos
    hemis, networks_7, networks_17, region_names_7, region_names_17 = [], [], [], [], []

    for l7, l17 in zip(labels_7, labels_17):

        l7 = l7.decode('utf-8') if isinstance(l7, bytes) else l7
        l17 = l17.decode('utf-8') if isinstance(l17, bytes) else l17
        
        p7 = l7.split('_')
        p17 = l17.split('_')

        hemis.append(p7[1])  # same
        networks_7.append(p7[2])
        networks_17.append(p17[2])
        region_names_7.append('_'.join(p7[3:]))
        region_names_17.append('_'.join(p17[3:]))

    # Create  df
    df = pd.DataFrame({
        'ROI': range(1000),
        'Hemisphere': hemis,
        'Network_7net': networks_7,
        'RegionName_7net': region_names_7,
        'Network_17net': networks_17,
        'RegionName_17net': region_names_17,
        'Correlation': encoding_accuracy
    })
    
    print("\nMean accuracy for network (7-net):")
    print(df.groupby('Network_7net')['Correlation'].mean().round(3).sort_values(ascending=False))

