import argparse
import os
import numpy as np
import h5py
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='visual')
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

    # Friends
    friends_seasons = [1, 2, 3, 4, 5, 6]
    for i in friends_seasons:
        #stimuli_list.append(os.path.join(base_dir, 'friends', args.modality,
        #    'friends_s'+str(i)+'_features_'+args.modality+'.h5'))
        stimuli_list.append(os.path.join(base_dir, 'friends', args.modality,
            'friends_s'+str(i)+'_features_language.h5'))
    # Movie10
    movie10_movies = ['bourne', 'figures', 'life', 'wolf']
    for i in movie10_movies:
        #stimuli_list.append(os.path.join(base_dir, 'movie10', args.modality,
         #   'movie10_'+i+'_features_'+args.modality+'.h5'))
        stimuli_list.append(os.path.join(base_dir, 'movie10', args.modality,
            'movie10_'+i+'_features_language.h5'))

    # Load the stimulus features for the encoding train stimuli
    movie_splits = []
    chunks_per_movie = []
    for i, stim_dir in tqdm(enumerate(stimuli_list)):
        data = h5py.File(stim_dir, 'r')
        for m, movie in enumerate(data.keys()):
            if i == 0 and m == 0: # if first episode of first season
                if args.modality == 'language_multi' or args.modality == 'visual' or args.modality == 'audio' or args.modality == 'emotion':
                    #features = np.asarray(data[movie][args.modality])
                    features = np.asarray(data[movie]['language'])
                else:
                    print('errror')
            else:
                if args.modality == 'language_multi' or args.modality == 'visual' or args.modality == 'audio' or args.modality == 'emotion':
                    features = np.append(
                        #features, np.asarray(data[movie][args.modality]), 0)
                        features, np.asarray(data[movie]['language']), 0)
                else:
                    print('error')
        
            if args.modality == 'language_multi' or args.modality == 'visual' or args.modality == 'audio' or args.modality == 'emotion':
                #chunks_per_movie.append(len(data[movie][args.modality]))
                chunks_per_movie.append(len(data[movie]['language']))
            else:
                print('error')
            movie_splits.append(movie)
        del data

    # Convert NaN values to zeros (PCA doesn't accept NaN values)
    print(features.shape)
    if args.modality == 'visual':
        features = features.reshape(features.shape[0], -1)
    #features = np.concatenate([features[:, :1024], features[:, -1024:]], axis=1)
    print(features.shape)
    features = np.nan_to_num(features)

    # z-score the features
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
        data_dir = os.path.join(base_dir, 'friends', args.modality,
            #'friends_s'+str(season)+'_features_'+args.modality+'.h5')
            'friends_s'+str(season)+'_features_language.h5')
        data = h5py.File(data_dir, 'r')
        for m, movie in enumerate(data.keys()):
            if season == test_seasons[0] and m == 0: # if first episode of first season
                if args.modality == 'language_multi' or args.modality == 'visual' or args.modality == 'audio' or args.modality == 'emotion':
                    #features = np.asarray(data[movie][args.modality])
                    features = np.asarray(data[movie]['language'])
                else:
                    print('error')
            else:
                if args.modality == 'language_multi' or args.modality == 'visual' or args.modality == 'audio' or args.modality == 'emotion':
                    features = np.append(
                        #features, np.asarray(data[movie][args.modality]), 0)
                        features, np.asarray(data[movie]['language']), 0)
                else:
                    print('error')
            if args.modality == 'language_multi' or args.modality == 'visual' or args.modality == 'audio' or args.modality == 'emotion':
                #chunks_per_movie.append(len(data[movie][args.modality]))
                chunks_per_movie.append(len(data[movie]['language']))
            else:
                print('error')
            movie_splits.append(movie)
        del data

    print(features.shape)
    if args.modality == 'visual':
        features = features.reshape(features.shape[0], -1)
    #features = np.concatenate([features[:, :1024], features[:, -1024:]], axis=1)
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
