import os
import glob
from pathlib import Path
import sys

import h5py
import numpy as np
import string
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image
import panns_inference
from panns_inference import AudioTagging
import librosa
import moten
import cv2

import torch
import torchvision.transforms as T
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from functools import partial


def frames_transform(args):
	"""Define the transforms for ViNet model input frames.
	
	Parameters
	----------
	args : Namespace
		Input arguments.
	
	Returns
	-------
	transform : torchvision.transforms.Compose
		Transform pipeline for video frames.
	transform2 : torchvision.transforms.Compose
		Transform pipeline for normalization only.
	"""
	transform = T.Compose([
		T.Resize((224, 384)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225])
	])

	transform2 = T.Compose([  
		T.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225])
	])

	return transform, transform2


def load_vinet_model(device):
	"""Load the pre-trained ViNet model.
	
	Parameters
	----------
	device : str
		Device to load the model on ('cuda' or 'cpu').
	
	Returns
	-------
	model : VideoSaliencyModel
		Loaded ViNet model.
	"""
	sys.path.append(os.path.abspath("ViNET"))
	from model import VideoSaliencyModel
 
	model = VideoSaliencyModel()
	model.load_state_dict(torch.load("ViNET/ViNet_Hollywood.pt", map_location=device))
	model.to(device)
	model.eval()

	return model


#def load_language_model(device):
	"""Load the BERT model with proper configuration.
	
	Parameters
	----------
	device : str
		Device to load the model on ('cuda' or 'cpu').
	
	Returns
	-------
	model : BertModel
		Configured BERT model.
	tokenizer : BertTokenizer
		BERT tokenizer.
	"""
	#from transformers import BertTokenizer, BertModel
	
	#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	#model = BertModel.from_pretrained('bert-base-uncased')
	#model.eval()
	#model = model.to(device)
	
	# Enable output of hidden states and attentions
	#model.config.output_hidden_states = True  
	#model.config.output_attentions = True
	
	#return model, tokenizer


def load_language_model(device):
    """Load the RoBERTa-base model with proper configuration.

    Parameters
    ----------
    device : str
        Device to load the model on ('cuda' or 'cpu').

    Returns
    -------
    model : RobertaModel
        Configured RoBERTa-base model.
    tokenizer : RobertaTokenizer  
        RoBERTa-base tokenizer.
    """
    from transformers import RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()
    model = model.to(device)

    # Enable output of hidden states and attentions
    model.config.output_hidden_states = True
    model.config.output_attentions = True

    print("RoBERTa-base loaded successfully!")
    return model, tokenizer

def list_movie_splits(args):
	"""List the available movies splits for the selected movie type, for which
	the stimulus features will be extracted.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	movie_splits_list : list
		List of movie splits for which the stimulus features are extracted.
	"""

	### List movie splits ###
	# Movie directories
	if args.modality == 'language':
		movie_dir = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'transcripts',
			args.movie_type, args.stimulus_type)
		file_type = 'tsv'
	else:
		movie_dir = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type)
		file_type = 'mkv'
	
	# List the movie splits
	if args.movie_type == 'friends':
		movie_splits_list = [
			x.split("/")[-1].split(".")[0][8:]
			for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
		]
	elif args.movie_type == 'movie10':
		if args.modality != 'language':
			movie_splits_list = [
				x.split("/")[-1].split(".")[0]
				for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
			]
		else:
			movie_splits_list = [
				x.split("/")[-1].split(".")[0][8:]
				for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
			]

	return movie_splits_list


def extract_visual_features(args, movie_split, model, 
	transform, transform2, device, save_dir):
	"""Extract and save saliency-masked visual features from video chunks.
	
	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split : str
		Movie split for which features are extracted.
	model : VideoSaliencyModel
		Loaded ViNet model.
	transform : torchvision.transforms.Compose
		Transform pipeline for video frames.
	transform2 : torchvision.transforms.Compose
		Transform pipeline for normalization only.
	device : str
		Device for computation ('cuda' or 'cpu').
	save_dir : str
		Directory to save features.
	"""

	# Set up hook to capture backbone features
	last_output = None
 
	def hook_backbone(module, input, output):
		nonlocal last_output
		last_output = output
        
	model.backbone.base4[-1].register_forward_hook(hook_backbone)

	### Temporary directory ###
	temp_dir = os.path.join(save_dir, 'temp')
	if not os.path.isdir(temp_dir):
		os.makedirs(temp_dir)

	### Stimulus path ###
	if args.movie_type == 'friends':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, 'friends_'+movie_split+'.mkv')
	elif args.movie_type == 'movie10':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, movie_split+'.mkv')

	### Divide the movie in chunks of length TR ###
	clip = VideoFileClip(stim_path)
	start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]

	### Loop over movie chunks ###
	visual_features = []
	resize = T.Resize((224, 384))
	
	try:
		for start in tqdm(start_times, desc=f"Processing {movie_split}"):

			### Save the chunk clips ###
			clip_chunk = clip.subclip(start, start + args.tr)
			chunk_path = os.path.join(temp_dir, 'visual_'+str(args.stimulus_type)+'.mp4')
			clip_chunk.write_videofile(chunk_path, verbose=False, logger=None)

			### Load the video chunk frames ###
			video_chunk = VideoFileClip(chunk_path)
			frames = [chunk_frames for chunk_frames in video_chunk.iter_frames()]
		
			if len(frames) == 0:
				print(f"Warning: No frames extracted for chunk at {start:.2f}s")
				continue
			
			# Resize frames
			resized_frames = [resize(Image.fromarray(f)) for f in frames]
	  
			# Create original tensor for masking
			orig_tensor = torch.stack([
				torch.from_numpy(np.array(f).transpose(2, 0, 1)).float() / 255.0 
				for f in resized_frames
			]).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, 3, T, H, W)
	 
			# Create frames tensor for saliency prediction
			frames_tensor = []
			for f in frames:
				frame_pil = Image.fromarray(f)
				frame_tensor = transform(frame_pil)
				frames_tensor.append(frame_tensor)
			clip_tensor = torch.stack(frames_tensor).permute(1, 0, 2, 3).unsqueeze(0).to(device)
	 
			# Get saliency maps
			with torch.no_grad():
				output = model(clip_tensor)  # (1, 1, 1, H, W)
			   
			# Prepare mask
			mask = (output - output.min()) / (output.max() - output.min() + 1e-9)
			mask = mask.squeeze(2)  # (1, 1, H, W)              
			mask = mask.repeat(1, 3, 1, 1)  # (1, 3, H, W)
			mask = mask.unsqueeze(2).repeat(1, 1, orig_tensor.shape[2], 1, 1) 
			
			# Apply mask
			masked_tensor = orig_tensor * mask
			video = masked_tensor.squeeze(0)
			
			# Normalize masked frames
			normalized_frames = []
			for i in range(video.shape[1]):  # loop over T
				frame = video[:, i, :, :]  # (3, H, W)
				norm_frame = transform2(frame)
				normalized_frames.append(norm_frame)

			masked_clip_tensor = torch.stack(normalized_frames, dim=1).unsqueeze(0).to(device)
			
			# Extract features from masked video
			with torch.no_grad():
				_ = model(masked_clip_tensor)

			if last_output is not None:
				feat = last_output.squeeze(0).cpu().numpy()  # (1024, 4, 7, 17)

				# Compute statistical features
				mean_feat = np.mean(feat, axis=(1, 2, 3))     # (1024,)
				std_feat = np.std(feat, axis=(1, 2, 3))      
				max_feat = np.max(feat, axis=(1, 2, 3)) 

				combined = np.stack([mean_feat, std_feat, max_feat], axis=1)  
				visual_features.append(combined)
			
			# Clean up
			video_chunk.close()
			if os.path.exists(chunk_path):
				os.remove(chunk_path)
	
	finally:
		clip.close()
 
	### Format the visual features ###
	if not visual_features:
		print(f"Warning: No visual features extracted for {movie_split}")
		visual_features = np.zeros((1, 1024, 3), dtype='float32')
	else:
		visual_features = np.array(visual_features, dtype='float32')

	### Save the visual features ###
	out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
		'_features_visual.h5')
	flag = 'a' if Path(out_file).exists() else 'w'
	with h5py.File(out_file, flag) as f:
		if movie_split in f:
			del f[movie_split]  # Remove existing group if present
		group = f.create_group(movie_split)
		group.create_dataset('visual', data=visual_features, dtype=np.float32)
	
	print(f"Saved {movie_split} visual features with shape {visual_features.shape}")


def extract_language_features(args, movie_split, model, tokenizer, device, save_dir):
	"""Extract and save advanced language features with multiple pooling strategies.

	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split : str
		Movie split for which the features are extracted and saved.
	model : BertModel
		BERT model with hidden states and attention outputs enabled.
	tokenizer : BertTokenizer
		Tokenizer corresponding to the language model.
	device : str
		Whether to compute on 'cpu' or 'gpu'.
	save_dir : str
		Save directory.
	"""

	### Stimulus path ###
	if args.movie_type == 'friends':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'transcripts',
			args.movie_type, args.stimulus_type, 'friends_'+movie_split+'.tsv')
	elif args.movie_type == 'movie10':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'transcripts',
			args.movie_type, args.stimulus_type, 'movie10_'+movie_split+'.tsv')

	### Read the transcripts ###
	df = pd.read_csv(stim_path, sep='\t')
	df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

	### Empty feature lists ###
	tokens = []  # Tokens of the complete transcripts
	pooled_all = []
 
	### Loop over text chunks ###
	for i in tqdm(range(df.shape[0]), desc=f"Processing {movie_split} language"):
		# Each row/sample of the df corresponds to one fMRI TR

		### Tokenize raw text ###
		if not df.iloc[i]["is_na"]:  # Only tokenize if words were spoken during a chunk
			
			tr_text = df.iloc[i]["text_per_tr"]
			tr_clean = tr_text.translate(str.maketrans('', '', string.punctuation))
			chunk_tokens = tokenizer.tokenize(tr_clean)
   
			if len(chunk_tokens) == 0:
				pooled_all.append(np.full(768 * 4, np.nan, dtype=np.float32))
				continue
    
			tokens.extend(chunk_tokens)
			input_ids = tokenizer.convert_tokens_to_ids(tokens[-(args.num_used_tokens):])
			input_ids = [101] + input_ids + [102]  # Add special tokens
			input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

			with torch.no_grad():
				outputs = model(input_tensor)
				layer8 = outputs.hidden_states[8][0][1:-1]  # (T, 768)
				
				attn = outputs.attentions[8][0, :, 0, 1:-1]
				#print(f"attn shape BEFORE mean: {attn.shape}")
				attn_weights = attn.mean(dim=0).cpu().numpy()
				#print(f"attn_weights shape AFTER mean: {attn_weights.shape}")		
				
			chunk_len = len(chunk_tokens)
			chunk_embeds = layer8[-chunk_len:].cpu().numpy()
			attn_weights = attn_weights[-chunk_len:]  
			attn_weights = attn_weights / attn_weights.sum() if attn_weights.sum() > 0 else None
		
			# Multiple pooling strategies
			# Pooling 1 - Mean
			mean_pool = np.mean(chunk_embeds, axis=0)
			# Pooling 2 - Max
			max_pool = np.max(chunk_embeds, axis=0)
			# Pooling 3 - Attention-weighted
			attn_pool = np.average(chunk_embeds, axis=0, weights=attn_weights) if attn_weights is not None else mean_pool
			# Pooling 4 - Standard Deviation
			std_pool = np.std(chunk_embeds, axis=0)
			
			# Concatenate all pooling results
			pooled = np.concatenate([
				mean_pool,
				max_pool,
				attn_pool,
				std_pool
			], axis=0)
			
			pooled_all.append(pooled.astype(np.float32))
        
		else:
			# No text in this TR
			pooled_all.append(np.full(768 * 4, np.nan, dtype=np.float32))

	### Format features ###
	pooled_all = np.array(pooled_all, dtype='float32')

	### Save the language features ###
	out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
		'_features_language.h5')
	flag = 'a' if Path(out_file).exists() else 'w'
	with h5py.File(out_file, flag) as f:
		if movie_split in f:
			del f[movie_split]  # Remove existing group if present
		group = f.create_group(movie_split)
		group.create_dataset('language', data=pooled_all, dtype=np.float32)
	
	print(f"Saved {movie_split} language features with shape {pooled_all.shape}")
 
 

def extract_audio_features(args, movie_split, device, save_dir):
	"""Extract and save the audio features from the .mkv file of the selected
	movie split.
	Parameters
	----------
	args : Namespace
		Input arguments.
	movie_split : str
		Movie split for which the features are extracted and saved.
	device : str
		Whether to compute on 'cpu' or 'gpu'.
	save_dir : str
		Save directory.
	"""
	### Temporary directory ###
	temp_dir = os.path.join(save_dir, 'temp')
	if os.path.isdir(temp_dir) == False:
		os.makedirs(temp_dir)
	
	### Initialize PANNs model ###
	at = AudioTagging(checkpoint_path=None, device=device)
	
	### Stimulus path ###
	if args.movie_type == 'friends':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, 'friends_'+movie_split+'.mkv')
	elif args.movie_type == 'movie10':
		stim_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
			args.stimulus_type, movie_split+'.mkv')
	
	### Divide the movie in chunks of length TR ###
	clip = VideoFileClip(stim_path)
	start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]
	
	### Loop over movie chunks ###
	audio_features_panns = []
	
	for start in start_times:
		### Save the chunk clips ###
		clip_chunk = clip.subclip(start, start + args.tr)
		chunk_path = os.path.join(temp_dir, 'audio_'+str(args.stimulus_type)+
			'.wav')
		clip_chunk.audio.write_audiofile(chunk_path, verbose=False)
		
		### Load the video chunk audio ###
		y, sr = librosa.load(chunk_path, sr=args.sr, mono=True)
		
		
		### Extract PANNs features ###
		# PANNs expects 32kHz, so resample if needed
		if sr != 32000:
			y_panns = librosa.resample(y, orig_sr=sr, target_sr=32000)
		else:
			y_panns = y
			
		# PANNs inference returns (clipwise_output, embedding)
		# We want the embedding [1] which is (2048,) dimensional
		try:
			_, embedding = at.inference(y_panns[None, :])  # Add batch dimension
			audio_features_panns.append(embedding.squeeze())
		except Exception as e:
			print(f"PANNs error for chunk {start}: {e}")
			# Fallback: zero vector
			audio_features_panns.append(np.zeros(2048, dtype=np.float32))
	
	### Format the audio features ###
	audio_features_panns = np.array(audio_features_panns, dtype='float32')
	
	### Save the audio features ###
	out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
		'_features_audio.h5')
	flag = 'a' if Path(out_file).exists() else 'w'
	with h5py.File(out_file, flag) as f:
		group = f.create_group(movie_split)
		group.create_dataset('audio', data=audio_features_panns, dtype=np.float32)
  
  
def process_chunk_motion(stim_path, start, tr, fps, target_H, target_W, pyramid_params):
    import warnings
    warnings.filterwarnings("ignore")
    from moviepy.editor import VideoFileClip
    import moten

    temp_path = f"temp_chunk_{start:.2f}.mp4"
    try:
        VideoFileClip(stim_path).subclip(start, start + tr).write_videofile(
            temp_path, audio=False, verbose=False, logger=None)
    except:
        return np.zeros(pyramid_params['n_features'], dtype=np.float32)

    try:
        luminance = moten.io.video2luminance(temp_path, nimages=int(tr * fps))
        os.remove(temp_path)
    except:
        os.remove(temp_path)
        return np.zeros(pyramid_params['n_features'], dtype=np.float32)

    T, _, _ = luminance.shape
    resized = np.zeros((T, target_H, target_W), dtype=np.float32)
    for t in range(T):
        resized[t] = cv2.resize(luminance[t], (target_W, target_H), interpolation=cv2.INTER_AREA)

    pyramid = moten.pyramids.MotionEnergyPyramid(
    	stimulus_vhsize=(target_H, target_W),
    	stimulus_fps=fps,
    	temporal_frequencies=[2, 4],         
    	spatial_frequencies=[4, 8, 16],          
    	spatial_directions=[0, 45, 90, 135, 180]
)

    try:
        features = pyramid.project_stimulus(resized).mean(axis=0)
    except:
        features = np.zeros(pyramid_params['n_features'], dtype=np.float32)

    return features



def extract_motion_features(args, movie_split, save_dir='.'):
    out_file = os.path.join(save_dir, f"{args.movie_type}_{args.stimulus_type}_features_motion.h5")
    flag = 'a' if Path(out_file).exists() else 'w'

    # Skip if already processed
    if flag == 'a':
        with h5py.File(out_file, 'r') as f:
            if movie_split in f:
                print(f"✔ Skipping '{movie_split}' — already extracted.")
                return

    if args.movie_type == 'friends':
        stim_path = os.path.join(args.project_dir, 'data',
            'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
            args.stimulus_type, f'friends_{movie_split}.mkv')
    elif args.movie_type == 'movie10':
        stim_path = os.path.join(args.project_dir, 'data',
            'algonauts_2025.competitors', 'stimuli', 'movies', args.movie_type,
            args.stimulus_type, f'{movie_split}.mkv')

    clip = VideoFileClip(stim_path)
    fps = int(clip.fps)
    tr = args.tr
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]

    H, W = clip.h, clip.w
    target_H, target_W = H // 4, W // 4

    pyramid_params = {
        'temporal_frequencies': [2, 4],
        'spatial_frequencies': [4, 8, 16],
        'orientations': 5,
        'n_features': 930
    }

    func = partial(
        process_chunk_motion,
        stim_path,
        tr=tr,
        fps=fps,
        target_H=target_H,
        target_W=target_W,
        pyramid_params=pyramid_params
    )

    print(f"Using {cpu_count()} CPU cores for motion feature extraction...")
    with Pool(processes=cpu_count()) as pool:
        visual_features = list(tqdm(pool.imap(func, start_times), total=len(start_times)))

    visual_features = np.array(visual_features, dtype=np.float32)

    with h5py.File(out_file, flag) as f:
        group = f.create_group(movie_split)
        group.create_dataset('motion', data=visual_features, dtype='float32')
