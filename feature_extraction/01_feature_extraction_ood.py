import argparse
import os
import torch
from tqdm import tqdm

from feature_extraction_utils_new import frames_transform, define_frames_transform
from feature_extraction_utils_new import load_vinet_model, load_language_model, load_language_model_multilingual, get_emotion_audio_model, get_vision_model
from feature_extraction_utils_ood import  extract_language_features, extract_visual_features
from feature_extraction_utils_ood import extract_audio_features, extract_audio_emo_features, extract_lowlevel_audio_features


parser = argparse.ArgumentParser()
#parser.add_argument('--movies', type=list, default=['chaplin', 'mononoke',
 #   'passepartout', 'planetearth', 'pulpfiction', 'wot'])
parser.add_argument('--movies', type=list, default=['pulpfiction'])
parser.add_argument('--modality', type=str, default='language',
                    choices=['visual', 'language', 'language_multilingual','audio', 'audio_low', 'audio_emo'],
                    help='Type of features to extract')
parser.add_argument('--tr', type=float, default=1.49,
                    help='fMRI repetition time')
parser.add_argument('--num_used_tokens', type=int, default=510,
                    help='Maximum number of tokens for language model context')
parser.add_argument('--project_dir', default='../data', type=str,
                    help='Project directory path')
parser.add_argument('--sr', type=int, default=22050)
args = parser.parse_args()

print('>>> Extract Advanced Stimulus Features <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Check for GPU
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nUsing device: {device}')

# =============================================================================
# Output directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
    'raw', 'ood', args.modality)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

print(f'Save directory: {save_dir}')

# =============================================================================
# Load the models used for feature extraction
# =============================================================================
print('\nLoading models...')

if args.modality == 'visual':
    # Load the video transform and model
    transform, transform2 = frames_transform(args)
    model = load_vinet_model(device)
    print('ViNet model loaded successfully')

elif args.modality == 'language':
    # Load the BERT model and tokenizer
    model, tokenizer = load_language_model(device)

elif args.modality == 'language_multilingual':
    # Load the BERT model and tokenizer
    model, tokenizer = load_language_model_multilingual(device)
 
elif args.modality == 'audio_emo':
    # Load Wav2Vec2 model
    processor, model = get_audio_model(device)
    
# =============================================================================
# Extract features for each movie split
# =============================================================================
print('\nExtracting OOD features...')
 
for movie in tqdm(args.movies):
    movie_splits_list = list_movie_splits(args, movie)
    print(f'\nFound {len(movie_splits_list)} movie splits: {movie_splits_list}')
    for movie_split in tqdm(movie_splits_list):
    
        try:
            if args.modality == 'visual':
                extract_visual_features(
                    args,
                    movie,
                    movie_split,
                    model,
                    transform,
                    transform2,
                    device,
                    save_dir
                )

            elif args.modality == 'language' or args.modality == 'language_multilingual':
                if movie != 'chaplin':
                    extract_language_features(
                        args,
                        movie,
                        movie_split,
                        model,
                        tokenizer,
                        device,
                        save_dir)
            
            elif args.modality == 'audio':
                extract_audio_features(
                    args,
                    movie,
                    movie_split,
                    device,
                    save_dir
                )
   
            elif args.modality == 'audio_low':
                extract_lowlevel_audio_features(
                    args,
                    movie,
                    movie_split,
                    device,
                    save_dir
                )
       
            elif args.modality == 'audio_emo':
                extract_audio_emo_features(
                    args,
                    movie,
                    movie_split,
                    processor,
                    model,
                    device,
                    save_dir
                )
  
        except Exception as e:
            print(f' Error processing {movie_split}: {str(e)}')
            continue

print(f'\n Feature extraction completed!')
print(f'Features saved in: {save_dir}')
