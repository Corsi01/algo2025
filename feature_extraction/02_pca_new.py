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
                    choices=['visual', 'visual_videomae', 'language', 'audio',  'audio_emo', 'language_multilingual'],
                    help='Type of features to extract')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--test', type=int, default=1)
parser.add_argument('--project_dir', default='../data/', type=str)
args = parser.parse_args()

print('>>> Stimulus features PCA <<<')
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
    'pca', 'friends_movie10', args.modality)

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
# =============================================================================
# Downsample the train stimulus features (Friends s1-s6 + Movie10)
# =============================================================================
if args.train == 1:

    # Get stimulus features directories
    stimuli_list = []
    base_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
        'raw')

    friends_seasons = [1, 2, 3, 4, 5, 6]
    movie10_movies = ['bourne', 'figures', 'life', 'wolf']

    if args.modality in ['audio', 'visual', 'language']:
        for i in friends_seasons:
            stimuli_list.append(os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(i)+'_features_'+args.modality+'.h5'))
        for i in movie10_movies:
            stimuli_list.append(os.path.join(base_dir, 'movie10', args.modality,
                'movie10_'+i+'_features_'+args.modality+'.h5'))

    elif args.modality == 'visual_videomae':
        for i in friends_seasons:
            stimuli_list.append(os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(i)+'_features_visual.h5'))
        for i in movie10_movies:
            stimuli_list.append(os.path.join(base_dir, 'movie10', args.modality,
                'movie10_'+i+'_features_visual.h5'))
            
    elif args.modality == 'language_multilingual':
        for i in friends_seasons:
            stimuli_list.append(os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(i)+'_features_language.h5'))
        for i in movie10_movies:
            stimuli_list.append(os.path.join(base_dir, 'movie10', args.modality,
                'movie10_'+i+'_features_language.h5'))
        
    elif args.modality == 'audio_emo':
        stimuli_list1 = []
        stimuli_list2 = []
        for i in friends_seasons:
            stimuli_list.append(os.path.join(base_dir, 'friends', 'audio_low',
                'friends_s'+str(i)+'_features_audio_low_level.h5'))
            stimuli_list2.append(os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(i)+'_emo_features_audio.h5'))
        for i in movie10_movies:
            stimuli_list.append(os.path.join(base_dir, 'movie10', 'audio_low',
                'movie10_'+i+'_features_audio_low_level.h5'))
            stimuli_list2.append(os.path.join(base_dir, 'movie10', args.modality,
                'movie10_'+i+'_emo_features_audio.h5'))

    if args.modality != 'audio_emo':
    # Load the stimulus features for the encoding train stimuli
        movie_splits = []
        chunks_per_movie = []
        for i, stim_dir in tqdm(enumerate(stimuli_list)):
            data = h5py.File(stim_dir, 'r')
            for m, movie in enumerate(data.keys()):
            
                if i == 0 and m == 0: # if first episode of first season
                    if args.modality in ['audio', 'language', 'visual']:
                        features = np.asarray(data[movie][args.modality])
                    elif args.modality == 'language_multilingual':
                        features = np.asarray(data[movie]['language'])
                    elif args.modality == 'visual_videomae':
                        raw = np.asarray(data[movie]['visual']
                        reshaped = raw.reshape(-1, 2048, 1408)  # (n_samples, 2048, 1408)
                        # Pooling
                        avg_pool = reshaped.mean(axis=1)                           
                        max_pool = reshaped.max(axis=1)                            
                        pooled = np.concatenate([avg_pool, max_pool], axis=1) 
                        features = pooled 
                else:
                    if args.modality in ['audio', 'language', 'visual']:
                        features = np.append(
                            features, np.asarray(data[movie][args.modality]), 0)      
                    elif args.modality == 'language_multilingual':
                        features = np.append(
                            features, np.asarray(data[movie]['language']), 0)
                    elif args.modality == 'visual_videomae':
                        raw = np.asarray(data[movie]['visual']
                        reshaped = raw.reshape(-1, 2048, 1408)  # (n_samples, 2048, 1408)
                        # Pooling
                        avg_pool = reshaped.mean(axis=1)                           
                        max_pool = reshaped.max(axis=1)                            
                        pooled = np.concatenate([avg_pool, max_pool], axis=1) 
                        features = np.append(features, pooled, axis=0)
        
                if args.modality in ['audio', 'language', 'visual']:
                    chunks_per_movie.append(len(data[movie][args.modality]))
                elif args.modality == 'language_multilingual':
                    chunks_per_movie.append(len(data[movie]['language']))
                elif args.modality == 'visual_videomae':
                    chunks_per_movie.append(len(data[movie]['visual']))
                
                movie_splits.append(movie)
            del data

    else: ### audio_emo
        
################## insert here #########################
    
    
    if args.modality == 'visual':
        features = features.reshape(features.shape[0], -1)
    elif args.modality == 'audio_emo':
        features2 = features2.squeeze(1)
        features = np.concatenate([features1, features2], axis=1)
    print(features.shape)
    features = np.nan_to_num(features)

    #z-score the features
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    # Save the z-score parameters
    scaler_param = {}
    scaler_param['mean_'] = scaler.mean_
    scaler_param['scale_'] = scaler.scale_
    scaler_param['var_'] = scaler.var_
    np.save(os.path.join(save_dir, 'scaler_param.npy'), scaler_param)
    del scaler, scaler_param

    # Downsample the features using PCA
    
    n_components = 250
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(features)
    features = pca.transform(features)
    features = features.astype(np.float32)
    # Save the PCA parameters
    pca_param = {}
    pca_param['components_'] = pca.components_
    pca_param['explained_variance_'] = pca.explained_variance_
    pca_param['explained_variance_ratio_'] = pca.explained_variance_ratio_
    pca_param['singular_values_'] = pca.singular_values_
    pca_param['mean_'] = pca.mean_
    np.save(os.path.join(save_dir, 'pca_param.npy'), pca_param)
    del pca, pca_param

    # Convert the features to float32
    features = features.astype(np.float32)

    # Reshape the features into individual movie splits
    features_train = {}
    count = 0
    for m, movie in enumerate(movie_splits):
        chunks = chunks_per_movie[m]
        features_train[movie] = features[count:count+chunks]
        count += chunks
    del features

    # Save the train features
    data = np.save(os.path.join(save_dir, 'features_train.npy'),
        features_train)
    del features_train


# =============================================================================
# Downsample the test stimulus features (Friends s7)
# =============================================================================
if args.test == 1:

    # Load the stimulus features for the test seasons
    test_seasons = [7]
    movie_splits = []
    chunks_per_movie = []

    for season in tqdm(test_seasons):
        if args.modality in ['audio', 'language', 'visual']:
            data_dir = os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(season)+'_features_'+args.modality+'.h5')
        elif args.modality == 'language_multilingual':
            data_dir = os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(season)+'_features_language.h5')
        elif args.modality == 'visual_videomae':
            data_dir = os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(season)+'_features_visual.h5')
        elif args.modality == 'audio_emo':
            data_dir1 = os.path.join(base_dir, 'friends', 'audio_low',
                'friends_s'+str(season)+'_features_audio_low_level.h5')
            data_dir2 = os.path.join(base_dir, 'friends', args.modality,
                'friends_s'+str(season)+'emo_features_audio.h5')
    
        if args.modality != 'audio_emo':
            data = h5py.File(data_dir, 'r')
            for m, movie in enumerate(data.keys()):
                if season == test_seasons[0] and m == 0: # if first episode of first season
                
                    if args.modality in ['audio', 'visual', 'language']:
                        features = np.asarray(data[movie][args.modality])
                    elif args.modality == 'language_multilingual':
                        features = np.asarray(data[movie]['language'])
                    elif args.modality == 'visual_videomae':
                        raw = np.asarray(data[movie]['visual']
                        reshaped = raw.reshape(-1, 2048, 1408)  # (n_samples, 2048, 1408)
                    # Pooling
                        avg_pool = reshaped.mean(axis=1)                           
                        max_pool = reshaped.max(axis=1)                            
                        pooled = np.concatenate([avg_pool, max_pool], axis=1)
                        features = pooled
                else:
                    if args.modality in ['audio', 'visual', 'language']:
                        features = np.append(features, np.asarray(data[movie][args.modality]), 0)
                    elif args.modality == 'language_multilingual':
                        features = np.append(features, np.asarray(data[movie]['language']), 0)
                    elif args.modality == 'visual_videomae':
                        raw = np.asarray(data[movie]['visual']
                        reshaped = raw.reshape(-1, 2048, 1408)  # (n_samples, 2048, 1408)
                        # Pooling
                        avg_pool = reshaped.mean(axis=1)                           
                        max_pool = reshaped.max(axis=1)                            
                        pooled = np.concatenate([avg_pool, max_pool], axis=1)
                        features = np.append(features, pooled, axis=0)
                    
                if args.modality in ['audio', 'visual', 'language']:
                    chunks_per_movie.append(len(data[movie][args.modality]))
                elif args.modality == 'language_multilingual':
                    chunks_per_movie.append(len(data[movie]['language']))
                elif args.modality == 'visual_videomae':
                    chunks_per_movie.append(len(data[movie]['visual']))

                movie_splits.append(movie)
            del data

        else: #### audio emo
    ###############i nsert code here #####################

    if args.modality == 'visual':
        features = features.reshape(features.shape[0], -1)
    if args.modality == 'audio_emo':
        features2 = features2.squeeze(1)
        features = np.concatenate([features1, features2], axis=1)
    
    print(features.shape)
    # Convert NaN values to zeros (PCA doesn't accept NaN values)
    features = np.nan_to_num(features)

    # z-score the features
    scaler_param = np.load(os.path.join(save_dir, 'scaler_param.npy'),
        allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_param['mean_']
    scaler.scale_ = scaler_param['scale_']
    scaler.var_ = scaler_param['var_']
    features = scaler.transform(features)
    del scaler, scaler_param

    # Downsample the features using PCA
    pca_param = np.load(os.path.join(save_dir, 'pca_param.npy'),
        allow_pickle=True).item()
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
    for m, movie in enumerate(movie_splits):
        chunks = chunks_per_movie[m]
        features_test[movie] = features[count:count+chunks]
        count += chunks
    del features

    # Save the features
    data = np.save(os.path.join(save_dir, 'features_test.npy'), features_test)
    del features_test
