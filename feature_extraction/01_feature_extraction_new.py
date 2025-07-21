import argparse
import os
import torch
from tqdm import tqdm

from feature_extraction_utils_new import frames_transform, define_frames_transform, list_movie_splits
from feature_extraction_utils_new import load_vinet_model, load_language_model, get_emotion_audio_model. get_vision_model
from feature_extraction_utils_new import extract_visual_features, extract_visual_features_videomae
from feature_extraction_utils_new import extract_language_features
from feature_extraction_utils_new import extract_lowlevel_audio_features, extract_audio_features, extract_emoton_audio_features


parser = argparse.ArgumentParser()
parser.add_argument('--movie_type', type=str, default='movie10',
					choices=['friends', 'movie10'],
					help='Type of movie dataset')
parser.add_argument('--stimulus_type', type=str, default='wolf',
					help='Specific stimulus (season for friends, movie for movie10)')
parser.add_argument('--modality', type=str, default='language',
					choices=['visual', 'visual_videomae','language', 'audio', 'audio_low', 'audio_emo'],
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
	'raw_', args.movie_type, args.modality)

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

elif args.modality == 'visual_videomae':
	# Load the video transform and model
	transform = define_frames_transform(args)
	feature_extractor, model_layer = get_vision_model(args, device)
	print('VideoMAE2 model loaded successfully')

elif args.modality == 'language':
	# Load the BERT model and tokenizer
	model, tokenizer = load_language_model(device)

elif args.modality == 'audio_emo':
	processor, model = get_emotion_audio_model(device)

# =============================================================================
# Get movie splits
# =============================================================================
movie_splits_list = list_movie_splits(args)
print(f'\nFound {len(movie_splits_list)} movie splits: {movie_splits_list}')

# =============================================================================
# Extract features for each movie split
# =============================================================================
print('\nExtracting features...')

for movie_split in tqdm(movie_splits_list, desc="Processing movie splits"):
	print(f'\n▶ Processing: {movie_split}')
	
	try:
		if args.modality == 'visual':
			extract_visual_features(
				args,
				movie_split,
				model,
				transform,
				transform2,
				device,
				save_dir
			)
		elif args.modality == 'visual_videomae':
			extract_visual_features_videomae(
				args,
				movie_split,
				feature_extractor,
				model_layer,
				transform,
				device,
				save_dir
			)

		elif args.modality == 'language':
			extract_language_features(
				args,
				movie_split,
				model,
				tokenizer,
				device,
				save_dir
			)
			
		elif args.modality == 'audio':
			extract_audio_features(
				args,
				movie_split,
				device,
				save_dir
			)
   
		elif args.modality == 'audio_low':
			extract_lowlevel_audio_features(
				args,
				movie_split,
				device,
				save_dir
			)

		elif args.modality == 'audio_emo':
			extract_emotion_audio_features(
				args,
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
