import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from data_utils import set_seed
from multisubject_utils import MultiSubjectMLP, MultiSubjectDataset


def align_features_and_fmri_samples_extended(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies,
    visual_extra_params=None, audio_extra_params=None):
    """
    Extended version of the original function with optional extra windows for visual and audio.
    """

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
            fmri_split = fmri_split[excluded_samples_start + hrf_delay:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features.keys():

                    ### Visual and audio features (ORIGINAL) ###
                    if mod == 'visual' or mod == 'audio' or mod == 'language_pooled' or mod == 'audio2' or mod == 'saliency':
                        
                        stimulus_time = s + excluded_samples_start
                        idx_start = stimulus_time - stimulus_window + 1
                        idx_end = stimulus_time + 1
                        
                        if idx_start < 0:
                            idx_start = 0
                            idx_end = stimulus_window
                        if idx_end > len(features[mod][split]):
                            idx_end = len(features[mod][split])
                            idx_start = max(0, idx_end - stimulus_window)
                        
                        f_base = features[mod][split][idx_start:idx_end]
                        f_combined = f_base.flatten()
                        
                        # Extra window for visual memory
                        if mod == 'saliency' and visual_extra_params is not None:
                            extra_hrf_delay = visual_extra_params['hrf_delay']
                            extra_stimulus_window = visual_extra_params['stimulus_window']
                            
                            extra_stimulus_time = s + excluded_samples_start - (extra_hrf_delay - hrf_delay)
                            extra_idx_start = extra_stimulus_time - extra_stimulus_window + 1
                            extra_idx_end = extra_stimulus_time + 1
                            
                            if extra_idx_start < 0:
                                # Padding 
                                n_zeros_needed = abs(extra_idx_start)
                                actual_start = 0
                                actual_end = min(extra_idx_end, len(features[mod][split]))
                                
                                if actual_end > actual_start:
                                    f_available = features[mod][split][actual_start:actual_end]
                                else:
                                    f_available = np.empty((0, features[mod][split].shape[1]))
                                
                                feature_dim = features[mod][split].shape[1]
                                f_zeros = np.zeros((n_zeros_needed, feature_dim))
                                f_extra = np.concatenate([f_zeros, f_available], axis=0)
                                
                                if len(f_extra) > extra_stimulus_window:
                                    f_extra = f_extra[:extra_stimulus_window]
                                elif len(f_extra) < extra_stimulus_window:
                                    missing = extra_stimulus_window - len(f_extra)
                                    f_extra = np.concatenate([f_extra, np.zeros((missing, feature_dim))], axis=0)
                            else:
                                if extra_idx_end > len(features[mod][split]):
                                    extra_idx_end = len(features[mod][split])
                                    extra_idx_start = max(0, extra_idx_end - extra_stimulus_window)
                                f_extra = features[mod][split][extra_idx_start:extra_idx_end]
                            
                            f_combined = np.concatenate([f_combined, f_extra.flatten()])
                        
                        # Extra window for audio/audio2 memory
                        elif (mod == 'audio' or mod == 'audio2') and audio_extra_params is not None:
                            extra_hrf_delay = audio_extra_params['hrf_delay']
                            extra_stimulus_window = audio_extra_params['stimulus_window']
                            
                            extra_stimulus_time = s + excluded_samples_start - (extra_hrf_delay - hrf_delay)
                            extra_idx_start = extra_stimulus_time - extra_stimulus_window + 1
                            extra_idx_end = extra_stimulus_time + 1
                            
                            if extra_idx_start < 0:
                                # Padding
                                n_zeros_needed = abs(extra_idx_start)
                                actual_start = 0
                                actual_end = min(extra_idx_end, len(features[mod][split]))
                                
                                if actual_end > actual_start:
                                    f_available = features[mod][split][actual_start:actual_end]
                                else:
                                    f_available = np.empty((0, features[mod][split].shape[1]))
                                
                                feature_dim = features[mod][split].shape[1]
                                f_zeros = np.zeros((n_zeros_needed, feature_dim))
                                f_extra = np.concatenate([f_zeros, f_available], axis=0)
                                
                                if len(f_extra) > extra_stimulus_window:
                                    f_extra = f_extra[:extra_stimulus_window]
                                elif len(f_extra) < extra_stimulus_window:
                                    missing = extra_stimulus_window - len(f_extra)
                                    f_extra = np.concatenate([f_extra, np.zeros((missing, feature_dim))], axis=0)
                            else:
                                if extra_idx_end > len(features[mod][split]):
                                    extra_idx_end = len(features[mod][split])
                                    extra_idx_start = max(0, extra_idx_end - extra_stimulus_window)
                                f_extra = features[mod][split][extra_idx_start:extra_idx_end]
                            
                            f_combined = np.concatenate([f_combined, f_extra.flatten()])
                        
                        f_all = np.append(f_all, f_combined)

                    ### Language features ###
                    elif mod == 'language':
                        stimulus_time = s + excluded_samples_start
                        idx = stimulus_time
                        
                        if idx >= len(features[mod][split]):
                            idx = len(features[mod][split]) - 1
                        
                        f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())

                aligned_features.append(f_all)

    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    return aligned_features, aligned_fmri


def train_multitask_model(features_list, fmri_list, val_features_list, val_fmri_list, 
                         input_dim=7000, output_dim=1000, decay=3e-5, embedding_dim=64, hidden_dim=800,
                         return_correlation=False, dropout_rate=0.6, seed=777,
                         batch_size=1024, lr=1e-4):
    """
    Training multi-task model for multiple subjects
    """
    
    # Set seed
    if seed is not None:
        set_seed(seed)
    
    num_epochs = 20  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
    train_dataset = MultiSubjectDataset(features_list, fmri_list, subject_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = MultiSubjectDataset(val_features_list, val_fmri_list, subject_ids)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = MultiSubjectMLP(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        num_subjects=4
    ).to(device)
    
    # Optimizer/loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    if not return_correlation:
        print(f"\nStarting multi-subject training...")
    
    for epoch in trange(num_epochs, desc="Training Epochs", disable=return_correlation):
        model.train()
        running_loss = 0.0
        
        for features_batch, fmri_batch, subject_onehot_batch in train_loader:
            features_batch = features_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            subject_onehot_batch = subject_onehot_batch.to(device)
            
            # Forward pass
            backbone_features = model(features_batch, subject_onehot_batch)
            
            # Compute loss
            total_loss = 0.0
            batch_size_total = 0
            
            for subj_id in range(4):
                # Find samples of this subject in the batch
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    # Predict for specific subject
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    # Loss for specific subject
                    subj_loss = criterion(subj_pred, subj_fmri)
                    total_loss += subj_loss * subj_features.size(0)
                    batch_size_total += subj_features.size(0)
            
            # Weighted mean of losses
            if batch_size_total > 0:
                total_loss = total_loss / batch_size_total
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * features_batch.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features_batch, fmri_batch, subject_onehot_batch in val_loader:
                features_batch = features_batch.to(device)
                fmri_batch = fmri_batch.to(device)
                subject_onehot_batch = subject_onehot_batch.to(device)
                
                backbone_features = model(features_batch, subject_onehot_batch)
                
                total_loss = 0.0
                batch_size_total = 0
                
                for subj_id in range(4):
                    subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                    
                    if subject_mask.any():
                        subj_features = backbone_features[subject_mask]
                        subj_fmri = fmri_batch[subject_mask]
                        subj_pred = model.heads[subj_id](subj_features)
                        
                        subj_loss = criterion(subj_pred, subj_fmri)
                        total_loss += subj_loss * subj_features.size(0)
                        batch_size_total += subj_features.size(0)
                
                if batch_size_total > 0:
                    total_loss = total_loss / batch_size_total
                    val_loss += total_loss.item() * features_batch.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        if not return_correlation:
            tqdm.write(f"Epoch {epoch+1:02d} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
    
    # Se return_correlation=True, calcola correlazione PER SOGGETTO e ritorna quella
    if return_correlation:
        model.eval()
        
        subject_data = {subj_id: {'true': [], 'pred': []} for subj_id in range(4)}
        
        with torch.no_grad():
            for features_batch, fmri_batch, subject_onehot_batch in val_loader:
                features_batch = features_batch.to(device)
                fmri_batch = fmri_batch.to(device)
                subject_onehot_batch = subject_onehot_batch.to(device)
                
                backbone_features = model(features_batch, subject_onehot_batch)
                
                # Collect predictions 
                for subj_id in range(4):
                    subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                    
                    if subject_mask.any():
                        subj_features = backbone_features[subject_mask]
                        subj_fmri = fmri_batch[subject_mask]
                        subj_pred = model.heads[subj_id](subj_features)
                        
                        subject_data[subj_id]['true'].append(subj_fmri.cpu().numpy())
                        subject_data[subj_id]['pred'].append(subj_pred.cpu().numpy())
        
        # Compurte corr for each subject
        subject_correlations = []
        
        for subj_id in range(4):
            if len(subject_data[subj_id]['true']) > 0:
                
                subj_true = np.vstack(subject_data[subj_id]['true'])
                subj_pred = np.vstack(subject_data[subj_id]['pred'])
                
                subj_correlations = []
                for p in range(subj_true.shape[1]):
                    corr = pearsonr(subj_true[:, p], subj_pred[:, p])[0]
                    if not np.isnan(corr):
                        subj_correlations.append(corr)
                
                subj_mean_corr = np.mean(subj_correlations) if subj_correlations else 0.0
                subject_correlations.append(subj_mean_corr)
        
        mean_correlation = np.mean(subject_correlations) if subject_correlations else 0.0
        return mean_correlation
    
    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Multi-Subject Training Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model


def train_multitask_model_detailed(features_list, fmri_list, val_features_list, val_fmri_list, 
                                  input_dim=7000, output_dim=1000, decay=3e-5, embedding_dim=64, hidden_dim=800,
                                  dropout_rate=0.6, seed=777):
    
    # Set seed
    if seed is not None:
        set_seed(seed)
    
    lr = 1e-4
    batch_size = 1024
    num_epochs = 20  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
    train_dataset = MultiSubjectDataset(features_list, fmri_list, subject_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = MultiSubjectDataset(val_features_list, val_fmri_list, subject_ids)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = MultiSubjectMLP(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        num_subjects=4
    ).to(device)
    
    # Optimizer/loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for features_batch, fmri_batch, subject_onehot_batch in train_loader:
            features_batch = features_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            subject_onehot_batch = subject_onehot_batch.to(device)
            
            # Forward pass
            backbone_features = model(features_batch, subject_onehot_batch)
            
            # Compute loss
            total_loss = 0.0
            batch_size_total = 0
            
            for subj_id in range(4):
                # Find samples of this subject in the batch
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    # Predict for specific subject
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    # Loss for specific subject
                    subj_loss = criterion(subj_pred, subj_fmri)
                    total_loss += subj_loss * subj_features.size(0)
                    batch_size_total += subj_features.size(0)
            
            # Weighted mean of losses
            if batch_size_total > 0:
                total_loss = total_loss / batch_size_total
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * features_batch.size(0)
    
    # Evaluation
    model.eval()
    subject_data = {subj_id: {'true': [], 'pred': []} for subj_id in range(4)}
    
    with torch.no_grad():
        for features_batch, fmri_batch, subject_onehot_batch in val_loader:
            features_batch = features_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            subject_onehot_batch = subject_onehot_batch.to(device)
            
            backbone_features = model(features_batch, subject_onehot_batch)
            
            # Collect predictions 
            for subj_id in range(4):
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    subject_data[subj_id]['true'].append(subj_fmri.cpu().numpy())
                    subject_data[subj_id]['pred'].append(subj_pred.cpu().numpy())
    
    subject_correlations = []
    
    for subj_id in range(4):
        if len(subject_data[subj_id]['true']) > 0:
           
            subj_true = np.vstack(subject_data[subj_id]['true'])
            subj_pred = np.vstack(subject_data[subj_id]['pred'])
            
            subj_correlations = []
            for p in range(subj_true.shape[1]):
                corr = pearsonr(subj_true[:, p], subj_pred[:, p])[0]
                if not np.isnan(corr):
                    subj_correlations.append(corr)
            
            subj_mean_corr = np.mean(subj_correlations) if subj_correlations else 0.0
            subject_correlations.append(subj_mean_corr)
    
    mean_correlation = np.mean(subject_correlations) if subject_correlations else 0.0
    
    return mean_correlation, subject_correlations


def evaluate_multitask_model(model, features_val_dict, fmri_val_dict, subject_mapping={1: 0, 2: 1, 3: 2, 5: 3}):
    """Evaluates the multi-task model on each subject individually"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for original_subj, mapped_subj in subject_mapping.items():
        print(f"\n=== Evaluating Subject {original_subj} (mapped to {mapped_subj}) ===")
        
        # Get validation data for this subject
        features_val = features_val_dict[original_subj]
        fmri_val = fmri_val_dict[original_subj]
        
        print('fmri val shape', fmri_val.shape)
        
        # Convert to tensor
        X_test = torch.tensor(features_val, dtype=torch.float32).to(device)
        
        # Predict using subject-specific head
        with torch.no_grad():
            y_pred = model.predict_subject(X_test, mapped_subj)
            y_pred = y_pred.detach().cpu().numpy()
            
        print('fmri pred shape', y_pred.shape)
            
        # Compute correlation
        correlations = np.array([
            pearsonr(fmri_val[:, i], y_pred[:, i])[0]
            for i in range(fmri_val.shape[1])
        ])

        mean_corr = np.nanmean(correlations)
        print(f"\nMean Correlation: {mean_corr:.3f}")


def evaluate_multitask_model_by_network(model, features_val_dict, fmri_val_dict, 
                                         parcel_list_limbic, parcel_list_control, 
                                         parcel_list_default, parcel_list_salvent,
                                         subject_mapping={1: 0, 2: 1, 3: 2, 5: 3}):
    """Evaluates the multi-task model on each subject individually, with separate results by network"""
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
   
    networks = {
        'Limbic': (0, len(parcel_list_limbic)),
        'Control': (len(parcel_list_limbic), len(parcel_list_limbic) + len(parcel_list_control)),
        'Default': (len(parcel_list_limbic) + len(parcel_list_control), 
                   len(parcel_list_limbic) + len(parcel_list_control) + len(parcel_list_default)),
        'SalVentAttn': (len(parcel_list_limbic) + len(parcel_list_control) + len(parcel_list_default),
                       len(parcel_list_limbic) + len(parcel_list_control) + len(parcel_list_default) + len(parcel_list_salvent))
    }
    
    results = {}
   
    for original_subj, mapped_subj in subject_mapping.items():
        print(f"\n=== Evaluating Subject {original_subj} (mapped to {mapped_subj}) ===")
       
        # Get validation data for this subject
        features_val = features_val_dict[original_subj]
        fmri_val = fmri_val_dict[original_subj]
       
        print('fmri val shape', fmri_val.shape)
       
        # Convert to tensor
        X_test = torch.tensor(features_val, dtype=torch.float32).to(device)
       
        # Predict using subject-specific head
        with torch.no_grad():
            y_pred = model.predict_subject(X_test, mapped_subj)
            y_pred = y_pred.detach().cpu().numpy()
           
        print('fmri pred shape', y_pred.shape)
        
        # Initialize subject results
        subject_results = {}
        
        # Compute correlations for each network separately
        for network_name, (start_idx, end_idx) in networks.items():
            # Extract predictions and true values for this network
            network_pred = y_pred[:, start_idx:end_idx]
            network_true = fmri_val[:, start_idx:end_idx]
            
            # Compute correlations for this network
            network_correlations = np.array([
                pearsonr(network_true[:, i], network_pred[:, i])[0]
                for i in range(network_true.shape[1])
            ])
            
            network_mean_corr = np.nanmean(network_correlations)
            subject_results[network_name] = {
                'correlations': network_correlations,
                'mean_correlation': network_mean_corr,
                'n_parcels': network_true.shape[1]
            }
            
            print(f"{network_name:12} - Mean Corr: {network_mean_corr:.3f} (n_parcels: {network_true.shape[1]})")
        
        # Overall mean correlation
        all_correlations = np.concatenate([subject_results[net]['correlations'] for net in networks.keys()])
        overall_mean = np.nanmean(all_correlations)
        subject_results['Overall'] = overall_mean
        
        print(f"{'Overall':12} - Mean Corr: {overall_mean:.3f}")
        
        results[original_subj] = subject_results
    
    # Print summary across all subjects
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL SUBJECTS")
    print(f"{'='*60}")
    
    for network_name in list(networks.keys()) + ['Overall']:
        if network_name == 'Overall':
            all_subj_corrs = [results[subj][network_name] for subj in results.keys()]
        else:
            all_subj_corrs = [results[subj][network_name]['mean_correlation'] for subj in results.keys()]
        
        mean_across_subj = np.mean(all_subj_corrs)
        std_across_subj = np.std(all_subj_corrs)
        
        print(f"{network_name:12} - Mean: {mean_across_subj:.3f} ± {std_across_subj:.3f}")
    
    return results


def train_multitask_model_all_data(features_train_dict, fmri_train_dict, 
                                   features_val_dict, fmri_val_dict, input_dim=7000, output_dim=1000,
                                   decay=4e-4, embedding_dim=256, hidden_dim=1000, dropout_rate=0.6, seed=777,
                                   batch_size=1024, lr=1e-4):
    """
    Train the multi-subject model on ALL available data (train + validation)
    Use only for final model before submission!
    
    """
    
    # Set seed
    if seed is not None:
        set_seed(seed)
    
    num_epochs = 20  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on ALL available data (train + validation) with seed={seed}...")
    
    # Combine all available data
    all_features_list = []
    all_fmri_list = []
    subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
    subject_mapping = {1: 0, 2: 1, 3: 2, 5: 3}
    
    for original_subj in [1, 2, 3, 5]:
        # Combine features train + val
        features_all = np.vstack([
            features_train_dict[original_subj], 
            features_val_dict[original_subj]
        ])
        
        # Combine fmri train + val
        fmri_all = np.vstack([
            fmri_train_dict[original_subj], 
            fmri_val_dict[original_subj]
        ])
        
        all_features_list.append(features_all)
        all_fmri_list.append(fmri_all)
    
    print("\n Combined dataset:")
    for i, original_subj in enumerate([1, 2, 3, 5]):
        mapped_subj = subject_mapping[original_subj]
        train_samples = len(features_train_dict[original_subj])
        val_samples = len(features_val_dict[original_subj])
        total_samples = all_features_list[i].shape[0]
        
        print(f"Subject {original_subj} (mapped to {mapped_subj}): "
              f"{train_samples:,} train + {val_samples:,} val = {total_samples:,} total")
    
    total_samples = sum(len(feat) for feat in all_features_list)
    print(f" Total samples across all subjects: {total_samples:,}")
    
    # Create combined dataset
    all_dataset = MultiSubjectDataset(all_features_list, all_fmri_list, subject_ids)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = MultiSubjectMLP(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        num_subjects=4
    ).to(device)
    
    # Optimizer/loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    
    print(f"\n Starting training...")
    print(f"Batches per epoch: {len(all_loader):,}")
    
    for epoch in trange(num_epochs, desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        
        for features_batch, fmri_batch, subject_onehot_batch in all_loader:
            features_batch = features_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            subject_onehot_batch = subject_onehot_batch.to(device)
            
            # Forward pass
            backbone_features = model(features_batch, subject_onehot_batch)
            
            # Compute loss for each subject
            total_loss = 0.0
            batch_size_total = 0
            
            for subj_id in range(4):
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    subj_loss = criterion(subj_pred, subj_fmri)
                    total_loss += subj_loss * subj_features.size(0)
                    batch_size_total += subj_features.size(0)
            
            # Weighted mean of subjects losses
            if batch_size_total > 0:
                total_loss = total_loss / batch_size_total
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * features_batch.size(0)
        
        epoch_train_loss = running_loss / len(all_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch+1:02d} - Train Loss: {epoch_train_loss:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss (All Data)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Multi-Subject Training on All Data')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    print(f"\nTraining completed!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    
    return model


class EnsembleNetworkModel(nn.Module):
    def __init__(self, visual_model, dorsattn_model, sommot_model, multi_model,
                 parcel_list_visual, parcel_list_dorsattn, parcel_list_sommot, parcel_list_multi,
                 feature_configs, subject_mappings, total_parcels=1000):
        """
        Ensemble model combining 4 MultiSubjectMLPs for different neural networks
        """
        super(EnsembleNetworkModel, self).__init__()
        
        # Store the pre-trained models
        self.visual_model = visual_model
        self.dorsattn_model = dorsattn_model
        self.sommot_model = sommot_model
        self.multi_model = multi_model
        
        # Store the parcel lists
        self.parcel_list_visual = parcel_list_visual
        self.parcel_list_dorsattn = parcel_list_dorsattn
        self.parcel_list_sommot = parcel_list_sommot
        self.parcel_list_multi = parcel_list_multi
        
        # Store feature configurations
        self.feature_configs = feature_configs
        self.subject_mappings = subject_mappings
        
        # Create mapping from original indices to prediction positions
        self.total_parcels = total_parcels
        self._create_reconstruction_mapping()
        
        # Freeze the backbone models 
        self._freeze_backbones()
    
    def _create_reconstruction_mapping(self):
        """Create mapping to reconstruct the original order"""
        # Mapping: from original index to (network, location_in_network).
        self.index_to_network = {}
        
        for i, idx in enumerate(self.parcel_list_visual):
            self.index_to_network[idx] = ('visual', i)
            
        for i, idx in enumerate(self.parcel_list_dorsattn):
            self.index_to_network[idx] = ('dorsattn', i)
            
        for i, idx in enumerate(self.parcel_list_sommot):
            self.index_to_network[idx] = ('sommot', i)
            
        for i, idx in enumerate(self.parcel_list_multi):
            self.index_to_network[idx] = ('multi', i)
    
    def _freeze_backbones(self):
        
        for model in [self.visual_model, self.dorsattn_model, self.sommot_model, self.multi_model]:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_backbones(self):
        
        for model in [self.visual_model, self.dorsattn_model, self.sommot_model, self.multi_model]:
            for param in model.parameters():
                param.requires_grad = True
    
    def _prepare_features(self, features_standard, features_different, network_name):
        
        config = self.feature_configs[network_name]
        
        if config['type'] == 'same':
            return features_standard
        elif config['type'] == 'different':
            return features_different
        elif config['type'] == 'subset':
            indices = config['indices']
            return features_standard[:, indices] if indices else features_standard
        else:
            raise ValueError(f"Unknown feature type: {config['type']}")
    
    def forward(self, features_standard, features_different, subject_id):
        """
        Forward pass through all backbones and full reconstruction
        Returns:
             full_prediction:(batch_size, total_parcels) with all predictions reassembled
        """
        batch_size = features_standard.shape[0]
        device = features_standard.device
        
        # Prepare features for each network
        visual_features = self._prepare_features(features_standard, features_different, 'visual')
        dorsattn_features = self._prepare_features(features_standard, features_different, 'dorsattn')
        sommot_features = self._prepare_features(features_standard, features_different, 'sommot')
        multi_features = self._prepare_features(features_standard, features_different, 'multi')
        
        # Map subject_id for each model
        visual_subj_id = self.subject_mappings['visual'][subject_id]
        dorsattn_subj_id = self.subject_mappings['dorsattn'][subject_id]
        sommot_subj_id = self.subject_mappings['sommot'][subject_id]
        multi_subj_id = self.subject_mappings['multi'][subject_id]
        
        # Predict by subject
        visual_pred = self.visual_model.predict_subject(visual_features, visual_subj_id)
        dorsattn_pred = self.dorsattn_model.predict_subject(dorsattn_features, dorsattn_subj_id)
        sommot_pred = self.sommot_model.predict_subject(sommot_features, sommot_subj_id)
        multi_pred = self.multi_model.predict_subject(multi_features, multi_subj_id)
        
        # Reconstruct 1000 parcels
        full_prediction = torch.zeros(batch_size, self.total_parcels, device=device)
        
        for idx in self.parcel_list_visual:
            network, pos = self.index_to_network[idx]
            full_prediction[:, idx] = visual_pred[:, pos]
            
        for idx in self.parcel_list_dorsattn:
            network, pos = self.index_to_network[idx]
            full_prediction[:, idx] = dorsattn_pred[:, pos]
            
        for idx in self.parcel_list_sommot:
            network, pos = self.index_to_network[idx]
            full_prediction[:, idx] = sommot_pred[:, pos]
            
        for idx in self.parcel_list_multi:
            network, pos = self.index_to_network[idx]
            full_prediction[:, idx] = multi_pred[:, pos]
        
        return full_prediction
    
    def get_predicted_indices(self):
        
        all_predicted = (self.parcel_list_visual + self.parcel_list_dorsattn + 
                        self.parcel_list_sommot + self.parcel_list_multi)
        return sorted(all_predicted)


def make_prediction(ensemble, features_standard, features_different, subject_id):
    """
    Makes predictions using ensemble model
    
    Args:
         ensemble: the initialized ensemble model
         features_standard: "normal" features - shape (batch_size, 7000+)
         features_different: "different" features per sommot - shape (batch_size, 3750+)
         subject_id: subject ID (1, 2, 3, or 5)
    
    Returns:
         predictions: array (batch_size, 1000) with predictions for all plots
    """
    
    print(f"Prediction for subject {subject_id}...")
    ensemble.eval()
    
    with torch.no_grad():
        
        if not isinstance(features_standard, torch.Tensor):
            features_standard = torch.tensor(features_standard, dtype=torch.float32)
        if not isinstance(features_different, torch.Tensor):
            features_different = torch.tensor(features_different, dtype=torch.float32)
        
        device = next(ensemble.parameters()).device
        features_standard = features_standard.to(device)
        features_different = features_different.to(device)
        
        print(f" Features standard shape: {features_standard.shape}")
        print(f"Features different shape: {features_different.shape}")
        
        # Predict
        predictions = ensemble(features_standard, features_different, subject_id)
        predictions = predictions.cpu().numpy()
        
        print(f"Output shape: {predictions.shape}")
        print(f"Output type: {type(predictions)}")
    
        
    return predictions


def get_network_parcels(df_atlas):
    
    parcel_lists = {}
    
    parcel_lists['vis'] = df_atlas[df_atlas.network_yeo7 == 'Vis'].parcel_id.tolist()
    parcel_lists['dorsattn'] = df_atlas[df_atlas.network_yeo7 == 'DorsAttn'].parcel_id.tolist()
    parcel_lists['sommot'] = df_atlas[df_atlas.network_yeo7 == 'SomMot'].parcel_id.tolist()
    parcel_lists['limbic'] = df_atlas[df_atlas.network_yeo7 == 'Limbic'].parcel_id.tolist()
    parcel_lists['control'] = df_atlas[df_atlas.network_yeo7 == 'Cont'].parcel_id.tolist()
    parcel_lists['default'] = df_atlas[df_atlas.network_yeo7 == 'Default'].parcel_id.tolist()
    parcel_lists['salvent'] = df_atlas[df_atlas.network_yeo7 == 'SalVentAttn'].parcel_id.tolist()
    
    # Multi network (limbic + control + default + salvent)
    parcel_lists['multi'] = (parcel_lists['limbic'] + parcel_lists['control'] + 
                            parcel_lists['default'] + parcel_lists['salvent'])
    
    return parcel_lists


def train_final_model_on_all_data(features_all_dict, fmri_all_dict, 
                                   input_dim=7000, output_dim=1000,
                                   decay=4e-4, embedding_dim=256, hidden_dim=1000, 
                                   dropout_rate=0.6, seed=777, batch_size=1024, lr=1e-4):
    """
    Train the final multi-subject model on ALL available data (already combined train + validation)
    Use only for final model training after hyperparameter optimization!
    
    Args:
        features_all_dict: Dict with combined train+val features for each subject
        fmri_all_dict: Dict with combined train+val fMRI data for each subject
        ... other parameters for model configuration
    """
    
    # Set seed per riproducibilità
    if seed is not None:
        set_seed(seed)
    
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training FINAL model on ALL available data with seed={seed}...")
    
    # Prepare data for training
    all_features_list = []
    all_fmri_list = []
    subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
    subject_mapping = {1: 0, 2: 1, 3: 2, 5: 3}
    
    for original_subj in [1, 2, 3, 5]:
        features_all = features_all_dict[original_subj]
        fmri_all = fmri_all_dict[original_subj]
        
        all_features_list.append(features_all)
        all_fmri_list.append(fmri_all)
    
    print("\nFinal training dataset:")
    for i, original_subj in enumerate([1, 2, 3, 5]):
        mapped_subj = subject_mapping[original_subj]
        total_samples = all_features_list[i].shape[0]
        
        print(f"Subject {original_subj} (mapped to {mapped_subj}): "
              f"{total_samples:,} total samples")
    
    total_samples = sum(len(feat) for feat in all_features_list)
    print(f"Total samples across all subjects: {total_samples:,}")
    
    # Create dataset
    all_dataset = MultiSubjectDataset(all_features_list, all_fmri_list, subject_ids)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = MultiSubjectMLP(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        num_subjects=4
    ).to(device)
    
    # Optimizer/loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    
    print(f"\nStarting final training...")
    print(f"Batches per epoch: {len(all_loader):,}")
    
    for epoch in trange(num_epochs, desc="Final Training Epochs"):
        model.train()
        running_loss = 0.0
        
        for features_batch, fmri_batch, subject_onehot_batch in all_loader:
            features_batch = features_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            subject_onehot_batch = subject_onehot_batch.to(device)
            
            # Forward pass
            backbone_features = model(features_batch, subject_onehot_batch)
            
            # Compute loss for each subject
            total_loss = 0.0
            batch_size_total = 0
            
            for subj_id in range(4):
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    subj_loss = criterion(subj_pred, subj_fmri)
                    total_loss += subj_loss * subj_features.size(0)
                    batch_size_total += subj_features.size(0)
            
            # Weighted mean of subjects losses
            if batch_size_total > 0:
                total_loss = total_loss / batch_size_total
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * features_batch.size(0)
        
        epoch_train_loss = running_loss / len(all_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch+1:02d} - Final Train Loss: {epoch_train_loss:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Final Training Loss (All Data)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Final Multi-Subject Training on All Available Data')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal training completed!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    
    return model
