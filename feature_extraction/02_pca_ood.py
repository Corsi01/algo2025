import argparse
import os
import numpy as np
import h5py
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='language',
                    choices=['visual', 'visual_videomae2','language', 'audio', 'audio_low', 'audio_emo', 'language_multi', 'multimodal'],
                    help='Type of features to extract')
parser.add_argument('--project_dir', default='data/', type=str)
args = parser.parse_args()

print('>>> Stimulus features PCA OOD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Output directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
    'pca', 'ood', args.modality)

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# =============================================================================
# OOD stimulus features
# =============================================================================
if args.modality != 'language' and args.modality != 'language_multi':
    movies = ['chaplin', 'mononoke', 'passepartout', 'planetearth',
        'pulpfiction', 'wot']
else:
    movies = ['mononoke', 'passepartout', 'planetearth',
        'pulpfiction', 'wot']

# Load the stimulus features for the OOD movies
episode_names = []
chunks_per_episode = []
for movie in tqdm(movies):
    
    if args.modality in ['language', 'audio', 'visual']:
    
        data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
            'raw', 'ood', args.modality, 'ood_'+movie+'_features_'+args.modality+
            '.h5')
        data = h5py.File(data_dir, 'r')
        for e, episode in enumerate(data.keys()):
            if movie == movies[0] and e == 0: # if first episode of first movie
                features = np.asarray(data[episode][args.modality])
            else:
                features = np.append(
                    features, np.asarray(data[episode][args.modality]), 0)
            
            chunks_per_episode.append(len(data[episode][args.modality]))
            episode_names.append(episode)
        ###########################################################################
    
    if args.modality == 'language_multi':
        
        data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
            'raw', 'ood', args.modality, 'ood_'+movie+'_features_language.h5')
        data = h5py.File(data_dir, 'r')
        for e, episode in enumerate(data.keys()):
            if movie == movies[0] and e == 0:
                features = np.asarray(data[episode]['language'])
            else:
                features = np.append(
                    features, np.asarray(data[episode]['language']), 0)
            
            chunks_per_episode.append(len(data[episode]['language']))
            episode_names.append(episode)
        ###########################################################################
        
    
    if args.modality == 'visual_videomae2':
        
        data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
            'raw', 'ood', args.modality, 'ood_'+movie+'_features_visual.h5')
        data = h5py.File(data_dir, 'r')
        for e, episode in enumerate(data.keys()):
            
            raw = np.asarray(data[episode]['visual'])                     # (n_samples, 2883584)
            reshaped = raw.reshape(-1, 2048, 1408)                      # (n_samples, 2048, 1408)
            # Pooling
            avg_pool = reshaped.mean(axis=1)                           
            max_pool = reshaped.max(axis=1)                            
            # Concatena pooling
            pooled = np.concatenate([avg_pool, max_pool], axis=1) 
            
            if movie == movies[0] and e == 0: # if first episode of first movie
                features = pooled
            else:
                features = np.append(features, pooled, axis=0)
            
            chunks_per_episode.append(len(pooled))
            episode_names.append(episode)
        ###########################################################################
    
    if args.modality == 'audio_emo':
    
        data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
            'raw', 'ood', 'audio_low', 'ood_'+movie+'_features_audio_low_level.h5')
        data = h5py.File(data_dir, 'r')
        for e, episode in enumerate(data.keys()):
            if movie == movies[0] and e == 0: 
                features1 = np.asarray(data[episode]['audio_opensmile'])
            else:
                features1 = np.append(
                    features1, np.asarray(data[episode]['audio_opensmile']), 0)
            
            chunks_per_episode.append(len(data[episode]['audio_opensmile']))
            episode_names.append(episode)
            
            print(episode, features1.shape)
      
        data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
            'raw', 'ood', 'audio_emo', 'ood_'+movie+'_emo_features_audio.h5')
        data = h5py.File(data_dir, 'r')
        for e, episode in enumerate(data.keys()):
            if movie == movies[0] and e == 0: 
                features2 = np.asarray(data[episode]['audio'])
            else:
                features2 = np.append(
                    features2, np.asarray(data[episode]['audio']), 0)
            
            
            print(episode, features2.shape)
        ###########################################################################
    
    del data

if args.modality == 'audio_emo':
    features2 = features2.squeeze(1)
    features = np.concatenate([features1, features2], axis=1)
    print(features.shape)
    
elif args.modality == 'visual':
    features = features.reshape(features.shape[0], -1)
print(features.shape)
# Convert NaN values to zeros (PCA doesn't accept NaN values)
features = np.nan_to_num(features)

# z-score the features
scaler_param = np.load(os.path.join(args.project_dir, 'results',
    'stimulus_features', 'pca', 'friends_movie10', args.modality,
    'scaler_param.npy'), allow_pickle=True).item()
scaler = StandardScaler()
scaler.mean_ = scaler_param['mean_']
scaler.scale_ = scaler_param['scale_']
scaler.var_ = scaler_param['var_']
features = scaler.transform(features)
del scaler, scaler_param

# Downsample the features using PCA
pca_param = np.load(os.path.join(args.project_dir, 'results',
    'stimulus_features', 'pca', 'friends_movie10', args.modality,
    'pca_param.npy'), allow_pickle=True).item()
n_components = 250
pca = PCA(n_components=n_components, random_state=seed)
pca.components_ = pca_param['components_']
pca.explained_variance_ = pca_param['explained_variance_']
pca.explained_variance_ratio_ = pca_param['explained_variance_ratio_']
pca.singular_values_ = pca_param['singular_values_']
pca.mean_ = pca_param['mean_']
features = pca.transform(features)
features = features.astype(np.float32)
del pca, pca_param

# Convert the features to float32
features = features.astype(np.float32)

# Reshape the features into individual episodes
features_test = {}
count = 0
for e, epi in enumerate(episode_names):
    chunks = chunks_per_episode[e]
    features_test[epi] = features[count:count+chunks]
    count += chunks
del features

# Save the features
data = np.save(os.path.join(save_dir, 'features_ood.npy'), features_test)
