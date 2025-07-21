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
import panns_inference
from panns_inference import AudioTagging
import librosa
import glob
import sys
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
#import torchaudio
import audeer
import audonnx


############ function for movie split list ###################

def list_movie_splits(args, movie):
    """List the available movie splits for a given movie, for which the
    features will be extracted.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    movie : str
        Movie for which the features are extracted.

    Returns
    -------
    movie_splits_list : list
        List of movie splits for which the stimulus features are extracted.

    """

    ### List episodes ###
    if args.modality == 'language':
        movie_dir = os.path.join(args.project_dir,
            'algonauts_2025.competitors', 'stimuli', 'transcripts', 'ood',
            movie)
        file_type = 'tsv'
        movie_splits_list = [
            x.split("/")[-1].split(".")[0][4:]
            for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
            if x.split("/")[-1].split(".")[0][-1] in ["1", "2"]
        ]
    else:
        movie_dir = os.path.join(args.project_dir,
            'algonauts_2025.competitors', 'stimuli', 'movies', 'ood', movie)
        file_type = 'mkv'
        movie_splits_list = [
            x.split("/")[-1].split(".")[0][5:-6]
            for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
            if x.split("/")[-1].split(".")[0][-7] in ["1", "2"]
        ]

    ### Output ###
    return movie_splits_list

#####################################################

######### functions to load models ##############

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
    sys.path.append(os.path.abspath("../ViNET"))
    from model import VideoSaliencyModel
 
    model = VideoSaliencyModel()
    model.load_state_dict(torch.load("ViNET/ViNet_Hollywood.pt", map_location=device))
    model.to(device)
    model.eval()

    return model

def get_audio_model(device):
    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    cache_root = audeer.mkdir('cache')
    model_root = audeer.mkdir('model')

    archive_path = audeer.download_url(url, cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)
    model = audonnx.load(model_root)
    return None, model

####################################################

################ LANGUAGE ##########################

def extract_language_features(args, movie, movie_split, model, tokenizer, device, save_dir):
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
    stim_path = os.path.join(args.project_dir, 
        'algonauts_2025.competitors', 'stimuli', 'transcripts', 'ood',
        movie, 'ood_'+movie_split+'.tsv')

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
    out_file = os.path.join(save_dir, 'ood_'+movie+'_features_language.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        if movie_split in f:
            del f[movie_split]  # Remove existing group if present
        group = f.create_group(movie_split)
        group.create_dataset('language', data=pooled_all, dtype=np.float32)
    
    print(f"Saved {movie_split} language features with shape {pooled_all.shape}")
    

############################ AUDIO 1 & 2 #########################

def extract_audio_features(args, movie, movie_split, device, save_dir):
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
    stim_path = os.path.join(args.project_dir, 
        'algonauts_2025.competitors', 'stimuli', 'movies', 'ood', movie,
        'task-'+movie_split+'_video.mkv')
    
    ### Divide the movie in chunks of length TR ###
    clip = VideoFileClip(stim_path)
    start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]
    
    ### Loop over movie chunks ###
    audio_features_panns = []
    
    for start in start_times:
        ### Save the chunk clips ###
        clip_chunk = clip.subclip(start, start + args.tr)
        chunk_path = os.path.join(temp_dir, 'audio_'+movie+'.wav')
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
    out_file = os.path.join(save_dir, 'ood_'+movie+'_features_audio.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.create_group(movie_split)
        group.create_dataset('audio', data=audio_features_panns, dtype=np.float32)


def extract_lowlevel_audio_features(args, movie, movie_split, device, save_dir):
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
    stim_path = os.path.join(args.project_dir, 
        'algonauts_2025.competitors', 'stimuli', 'movies', 'ood', movie,
        'task-'+movie_split+'_video.mkv')
    
    ### Divide the movie in chunks of length TR ###
    clip = VideoFileClip(stim_path)
    start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]
    
    ### Loop over movie chunks ###
    audio_features_opensmile = []
    
    for start in start_times:
        ### Save the chunk clips ###
        clip_chunk = clip.subclip(start, start + args.tr)
        chunk_path = os.path.join(temp_dir, 'audio_'+movie+'.wav')
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
    out_file = os.path.join(save_dir, 'ood_'+movie+'_features_audio_low_level.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.create_group(movie_split)
        group.create_dataset('audio_opensmile', data=audio_features_opensmile, dtype=np.float32)


@torch.no_grad()
def extract_audio_emo_features(args, movie, movie_split, processor, model, device, save_dir):
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

    ### Stimulus path ###
    stim_path = os.path.join(args.project_dir, 
        'algonauts_2025.competitors', 'stimuli', 'movies', 'ood', movie,
        'task-'+movie_split+'_video.mkv')

    ### Divide the movie in chunks of length TR ###
    clip = VideoFileClip(stim_path)
    start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]

    ### Loop over movie chunks ###
    audio_features = []
    audio_logits = []
    for start in start_times:

        ### Save the chunk clips ###
        clip_chunk = clip.subclip(start, start + args.tr)
        chunk_path = os.path.join(temp_dir, 'audio_'+movie+'.wav')
        clip_chunk.audio.write_audiofile(chunk_path, verbose=False)

        ### Load the video chunk audio ###
        y, sr = torchaudio.load(chunk_path)
        y = y.mean(0) # stereo 2 mono
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        y = resampler(y)

        ### Extract the audio features ###
        res = model(y.numpy(), 16000)
        audio_features.append(res['hidden_states'])
        audio_logits.append(res['logits'])

    ### Format the audio features ###
    audio_features = np.array(audio_features, dtype='float32')
    audio_logits = np.array(audio_logits, dtype='float32')

    ### Save the audio features ###
    out_file = os.path.join(save_dir, 'ood_'+movie+'_emo_features_audio.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.create_group(movie_split)
        group.create_dataset('audio', data=audio_features, dtype=np.float32)

    out_file = os.path.join(save_dir, 'ood_'+movie+'_emo_logits_audio.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.create_group(movie_split)
        group.create_dataset('audio', data=audio_logits, dtype=np.float32)


#################### VISUAL FEATURES ##################################


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


def extract_visual_features(args, movie, movie_split, model, 
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
    stim_path = os.path.join(args.project_dir,
        'algonauts_2025.competitors', 'stimuli', 'movies', 'ood', movie,
        'task-'+movie_split+'_video.mkv')

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
            chunk_path = os.path.join(temp_dir, 'visual_'+movie+'.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, logger=None)

            ### Load the video chunk frames ###
            video_chunk = VideoFileClip(chunk_path)
            frames = [chunk_frames for chunk_frames in video_chunk.iter_frames()]
            
            if len(frames) > 50:
                indices = np.linspace(0, len(frames)-1, 44, dtype=int)
                frames = [frames[i] for i in indices]

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
    out_file = os.path.join(save_dir, 'ood_'+movie+'_features_visual.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        if movie_split in f:
            del f[movie_split]  # Remove existing group if present
        group = f.create_group(movie_split)
        group.create_dataset('visual', data=visual_features, dtype=np.float32)
    
    print(f"Saved {movie_split} visual features with shape {visual_features.shape}")
