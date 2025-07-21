import os
import pandas as pd
import numpy as np
import h5py
import string
from pathlib import Path
from moviepy.editor import VideoFileClip
import opensmile
from tqdm import tqdm
import torch
import torch.nn.functional as F
import glob
import sys
from PIL import Image
import panns_inference
from panns_inference import AudioTagging
import librosa
import moten
import cv2
import torchvision.transforms as T
from multiprocessing import Pool, cpu_count
from functools import partial


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
                pooled_all.append(np.full((12*12*11 + 768*3), np.nan, dtype=np.float32))
                continue
    
            tokens.extend(chunk_tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens[-(args.num_used_tokens):])
            input_ids = [101] + input_ids + [102]  # Add special tokens
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                layer8 = outputs.hidden_states[8][0][1:-1]  # (T, 768)
                    
                all_layers = []
                for layer in outputs.attentions:
                    attn = layer[:, :, 1:-1, 1:-1]
                    attn = attn.mean(dim=-1)
                    pad_len = 11 - attn.size(-1)

                    if pad_len > 0:
                        attn = F.pad(attn, (pad_len, 0), value=0)  # pad left
                    elif pad_len < 0:
                        attn = attn[:, :, -11:]
                    #### if pad_len == 0, do nothing, already correct

                    attn_np = attn.squeeze(0).cpu().numpy().flatten()
                    all_layers.append(attn_np)
                feat = np.concatenate(all_layers)

                chunk_len = len(chunk_tokens)
                chunk_embeds = layer8[-chunk_len:].cpu().numpy()
                
                # Multiple pooling strategies
                # Pooling 1 - Mean
                mean_pool = np.mean(chunk_embeds, axis=0)
                # Pooling 2 - Max
                max_pool = np.max(chunk_embeds, axis=0)
                # Pooling 3 - Standard Deviation
                std_pool = np.std(chunk_embeds, axis=0)
                
                # Concatenate all pooling results
                pooled = np.concatenate([
                    mean_pool,
                    max_pool,
                    std_pool,
                    feat
                ], axis=0)
                
                pooled_all.append(pooled.astype(np.float32))
        
        else:
            # No text in this TR
            pooled_all.append(np.full((12*12*11 + 768*3), np.nan, dtype=np.float32))

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


def extract_lowlevel_audio_features(args, movie_split, device, save_dir):
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
    
    ### Initialize OpenSMILE model ###
    # OpenSMILE with eGeMAPS (88 features ottimizzate per emozioni)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
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
    audio_features_opensmile = []
    
    for start in start_times:
        ### Save the chunk clips ###
        clip_chunk = clip.subclip(start, start + args.tr)
        chunk_path = os.path.join(temp_dir, 'audio_'+str(args.stimulus_type)+
            '.wav')
        clip_chunk.audio.write_audiofile(chunk_path, verbose=False)
        
        ### Extract OpenSMILE features ###
        try:
            # OpenSMILE può lavorare direttamente sul file
            smile_features = smile.process_file(chunk_path)
            # smile_features è un DataFrame, prendiamo i valori
            audio_features_opensmile.append(smile_features.values.flatten())
        except Exception as e:
            print(f"OpenSMILE error for chunk {start}: {e}")
            # Fallback: zero vector (eGeMAPS ha 88 features)
            audio_features_opensmile.append(np.zeros(88, dtype=np.float32))
    
    ### Format the audio features ###
    audio_features_opensmile = np.array(audio_features_opensmile, dtype='float32')
    
    ### Save the audio features ###
    out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
        '_features_audio_low_level.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.create_group(movie_split)
        group.create_dataset('audio_opensmile', data=audio_features_opensmile, dtype=np.float32)


def extract_visual_features(args, movie_split, model, 
    transform, transform2, device, save_dir):
    """Extract and save saliency-masked visual features from video chunks.
    Includes automatic resume functionality - removes last incomplete split and skips completed ones.
    
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
    
    # RESUME LOGIC: Setup output file and handle resume
    out_file = os.path.join(save_dir, args.movie_type+'_'+args.stimulus_type+
        '_features_visual.h5')
    
    # Always check and remove last incomplete split if file exists
    if os.path.exists(out_file):
        try:
            # Read existing splits
            with h5py.File(out_file, 'r') as f:
                splits = sorted(list(f.keys()))
            
            if splits:
                last_split = splits[-1]
                
                # If current movie_split is the last one, remove it (presumed incomplete)
                if movie_split == last_split:
                    with h5py.File(out_file, 'a') as f:
                        if last_split in f:
                            del f[last_split]
                            print(f"Removed potentially incomplete split: {last_split}")
                
                # If current movie_split is already processed (and not the last), skip it
                elif movie_split in splits:
                    print(f"{movie_split} already processed, skipping...")
                    return
                    
        except Exception as e:
            print(f"Error handling resume: {e}")

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
            
            # FIX: Limita a 44 frame se ce ne sono troppi (per gestire video a 60fps)
            original_frame_count = len(frames)
            if len(frames) > 50:
                indices = np.linspace(0, len(frames)-1, 44, dtype=int)
                frames = [frames[i] for i in indices]
                print(f"DEBUG: Reduced frames from {original_frame_count} to {len(frames)} for {movie_split}")
        
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
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        # No automatic deletion - resume friendly
        group = f.create_group(movie_split)
        group.create_dataset('visual', data=visual_features, dtype=np.float32)
    
    print(f"Saved {movie_split} visual features with shape {visual_features.shape}")
