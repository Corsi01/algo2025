"""
Optimization utilities for multi-subject fMRI training with persistent storage
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import json
import os
from datetime import datetime
import pandas as pd
from utilities import train_multitask_model
from ..MultiSubjectModel_utils import MultiSubjectMLP, MultiSubjectDataset
import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage


def get_search_spaces():

    search_spaces = {
        'visual': {
            'lr': (5e-5, 5e-4),
            'batch_size': [512, 1024, 2048],
            'hidden_dim': (500, 900),
            'embedding_dim': (100, 200),
            'dropout': (0.4, 0.8),
            'decay': (1e-3, 5e-3)
        },
        'dorsattn': {
            'lr': (5e-5, 5e-4),
            'batch_size': [512, 1024, 2048],
            'hidden_dim': (500, 900),
            'embedding_dim': (100, 200),
            'dropout': (0.4, 0.8),
            'decay': (1e-3, 5e-3)
        },
        'sommot': {
            'lr': (5e-5, 5e-4),
            'batch_size': [512, 1024, 2048],
            'hidden_dim': (600, 1000),
            'embedding_dim': (100, 200),
            'dropout': (0.4, 0.8),
            'decay': (1e-3, 5e-3)
        },
        'multi': {
            'lr': (5e-5, 5e-4),
            'batch_size': [512, 1024, 2048],
            'hidden_dim': (800, 1200),
            'embedding_dim': (150, 350),
            'dropout': (0.4, 0.8),
            'decay': (3e-4, 1e-3)
        }
    }
    
    return search_spaces


def get_current_best_configs():
    """
    Base configs (find with hand trial and errors)
    """
    best_configs = {
        'visual': {
            'lr': 1e-4,
            'batch_size': 1024,
            'hidden_dim': 700,
            'embedding_dim': 150,
            'dropout': 0.6,
            'decay': 2e-3
        },
        'dorsattn': {
            'lr': 1e-4,
            'batch_size': 1024,
            'hidden_dim': 700,
            'embedding_dim': 150,
            'dropout': 0.6,
            'decay': 2e-3
        },
        'sommot': {
            'lr': 1e-4,
            'batch_size': 1024,
            'hidden_dim': 800,
            'embedding_dim': 150,
            'dropout': 0.6,
            'decay': 2e-3
        },
        'multi': {
            'lr': 1e-4,
            'batch_size': 1024,
            'hidden_dim': 1000,
            'embedding_dim': 256,
            'dropout': 0.6,
            'decay': 6e-4
        }
    }
    
    return best_configs


def train_multitask_model_detailed(features_list, fmri_list, val_features_list, val_fmri_list, 
                                  input_dim=7000, output_dim=1000, decay=3e-5, embedding_dim=64, hidden_dim=800,
                                  dropout_rate=0.6, seed=777):
    """
    Training multi-task model that returns both mean correlation and individual subject correlations
    
    Returns:
        tuple: (mean_correlation, subject_correlations_list)
    """
    
    if seed is not None:
        from data_utils import set_seed
        set_seed(seed)
    
    lr = 1e-4
    batch_size = 1024
    num_epochs = 20  # same number for optimization and final training
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
            
            # Collect predictions for subject
            for subj_id in range(4):
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    subject_data[subj_id]['true'].append(subj_fmri.cpu().numpy())
                    subject_data[subj_id]['pred'].append(subj_pred.cpu().numpy())
    
    # Compute correlation for each subject
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
    
    # Mean between subjects
    mean_correlation = np.mean(subject_correlations) if subject_correlations else 0.0
    
    return mean_correlation, subject_correlations


def train_with_hyperparams_optimized_detailed(train_dataset, val_dataset, hyperparams, network_name, seed=777):
    """
    Training ottimizzato che restituisce sia la correlazione media che i dettagli per soggetto
    
    Returns:
        tuple: (final_correlation, subject_correlations_list)
    """
    
    from data_utils import set_seed
    set_seed(seed)
    
    print(f"Training {network_name} with hyperparams (seed={seed}):")
    for key, value in hyperparams.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # Params training
    lr = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    hidden_dim = hyperparams['hidden_dim']
    embedding_dim = hyperparams['embedding_dim']
    dropout_rate = hyperparams['dropout']
    decay = hyperparams['decay']

    num_epochs = 20  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = train_dataset.features.shape[1]
    output_dim = train_dataset.fmri.shape[1]
    
    try:
        # Create DataLoaders (only thing that changes between trials)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for features_batch, fmri_batch, subject_onehot_batch in train_loader:
                features_batch = features_batch.to(device)
                fmri_batch = fmri_batch.to(device)
                subject_onehot_batch = subject_onehot_batch.to(device)
                
                # Forward pass
                backbone_features = model(features_batch, subject_onehot_batch)
                
                # Compute loss per subject
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
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()

        # EVALUATION
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
                
                print(f"  Subject {subj_id}: {subj_mean_corr:.4f} correlation")
        
        # Mean 4 subjects correlation
        final_correlation = np.mean(subject_correlations) if subject_correlations else 0.0
        
        print(f"Network {network_name} - Final correlation (mean of {len(subject_correlations)} subjects): {final_correlation:.4f}")
        print(f"  Individual subjects: {[f'{corr:.4f}' for corr in subject_correlations]}")
        
        return final_correlation, subject_correlations
            
    except Exception as e:
        print(f"Training failed for {network_name}: {e}")
        return 0.0, [0.0, 0.0, 0.0, 0.0]


def train_with_hyperparams_detailed(network_data, hyperparams, network_name, seed=777):
    
    # Set seed 
    from data_utils import set_seed
    set_seed(seed)
    
    print(f"Training {network_name} with hyperparams (seed={seed}):")
    for key, value in hyperparams.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # input/output
    input_dim = network_data['train']['features'][0].shape[1]
    output_dim = network_data['train']['fmri'][0].shape[1]
    
    try:
        correlation, subject_details = train_multitask_model_detailed(
            features_list=network_data['train']['features'],
            fmri_list=network_data['train']['fmri'],
            val_features_list=network_data['val']['features'],
            val_fmri_list=network_data['val']['fmri'],
            input_dim=input_dim,
            output_dim=output_dim,
            decay=hyperparams['decay'],
            embedding_dim=hyperparams['embedding_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            dropout_rate=hyperparams['dropout'],
            seed=seed  # Passa il seed
        )
        
        print(f"Network {network_name} - Final correlation (mean of 4 subjects): {correlation:.4f}")
        print(f"  Individual subjects: {[f'{corr:.4f}' for corr in subject_details]}")
        
        return correlation, subject_details
        
    except Exception as e:
        print(f"Training failed for {network_name}: {e}")
        return 0.0, [0.0, 0.0, 0.0, 0.0]


def train_with_hyperparams_optimized(train_dataset, val_dataset, hyperparams, network_name, seed=777):
    """
    Wrapper for mean correlation
    """
    correlation, _ = train_with_hyperparams_optimized_detailed(train_dataset, val_dataset, hyperparams, network_name, seed)
    return correlation


def train_with_hyperparams(network_data, hyperparams, network_name, seed=777):
    correlation, _ = train_with_hyperparams_detailed(network_data, hyperparams, network_name, seed)
    return correlation


def create_objective_function_optimized(network_name, train_dataset, val_dataset, seed=777):

    search_spaces = get_search_spaces()
    space = search_spaces[network_name]
    
    def objective(trial):
        # Sample hyperparameters
        hyperparams = {
            'lr': trial.suggest_float('lr', space['lr'][0], space['lr'][1], log=True),
            'batch_size': trial.suggest_categorical('batch_size', space['batch_size']),
            'hidden_dim': trial.suggest_int('hidden_dim', space['hidden_dim'][0], space['hidden_dim'][1]),
            'embedding_dim': trial.suggest_int('embedding_dim', space['embedding_dim'][0], space['embedding_dim'][1]),
            'dropout': trial.suggest_float('dropout', space['dropout'][0], space['dropout'][1]),
            'decay': trial.suggest_float('decay', space['decay'][0], space['decay'][1], log=True)
        }
        
        # Train and evaluate with pre-computed datasets 
        correlation, subject_details = train_with_hyperparams_optimized_detailed(train_dataset, val_dataset, hyperparams, network_name, seed)
        
        # Convert numpy floats to Python floats for JSON serialization
        subject_details_serializable = [float(corr) for corr in subject_details]
        
        # Save details
        trial.set_user_attr('subject_correlations', subject_details_serializable)
        
        return correlation
    
    return objective


def create_objective_function(network_name, network_data, seed=777):
   
    search_spaces = get_search_spaces()
    space = search_spaces[network_name]
    
    def objective(trial):
        # Sample hyperparameters
        hyperparams = {
            'lr': trial.suggest_float('lr', space['lr'][0], space['lr'][1], log=True),
            'batch_size': trial.suggest_categorical('batch_size', space['batch_size']),
            'hidden_dim': trial.suggest_int('hidden_dim', space['hidden_dim'][0], space['hidden_dim'][1]),
            'embedding_dim': trial.suggest_int('embedding_dim', space['embedding_dim'][0], space['embedding_dim'][1]),
            'dropout': trial.suggest_float('dropout', space['dropout'][0], space['dropout'][1]),
            'decay': trial.suggest_float('decay', space['decay'][0], space['decay'][1], log=True)
        }
        
        # Train and evaluate 
        correlation, subject_details = train_with_hyperparams_detailed(network_data, hyperparams, network_name, seed)
        
        # Save
        trial.set_user_attr('subject_correlations', subject_details)
        
        return correlation
    
    return objective


def optimize_network_hyperparams(network_name, network_data, n_trials=50, study_name=None, use_optimized=True, seed=777, use_storage=True, storage_path="optimization_history.db", warm_start=True):
    """
    Ottimizza iperparametri per un network specifico
    
    Args:
        network_name: nome del network ('visual', 'dorsattn', 'sommot', 'multi')
        network_data: dati del network 
        n_trials: numero di trial
        study_name: nome dello studio (optional)
        use_optimized: se True, usa versione ottimizzata con dataset pre-computati
        seed: seed per riproducibilità (default 777)
        use_storage: se True, usa storage persistente SQLite
        storage_path: path del database SQLite (solo se use_storage=True)
        warm_start: se True, parte dalle tue configurazioni attuali
        
    Returns:
        dict: migliori iperparametri trovati
    """
    
    if use_storage and warm_start:
        return optimize_with_warm_start(
            network_name=network_name,
            network_data=network_data,
            n_trials=n_trials,
            storage_path=storage_path,
            use_optimized=use_optimized,
            seed=seed
        )
    
    if use_storage:
        return optimize_with_storage(
            network_name=network_name,
            network_data=network_data,
            n_trials=n_trials,
            storage_path=storage_path,
            study_name=study_name,
            use_optimized=use_optimized,
            seed=seed
        )
    
    if study_name is None:
        study_name = f"{network_name}_optimization"
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZING {network_name.upper()} NETWORK HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"N trials: {n_trials}")
    print(f"Study name: {study_name}")
    print(f"Using optimized version: {use_optimized}")
    print(f"Seed: {seed}")
    print(f"Storage: In-memory (temporary)")
    print(f"Warm start: {warm_start}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize correlation
        sampler=TPESampler(seed=seed),  # Usa stesso seed anche per sampler
        study_name=study_name
    )
    
    if warm_start:
        best_configs = get_current_best_configs()
        current_config = best_configs[network_name]
        study.enqueue_trial(current_config)
        print(f"Warm start: using your current {network_name} configuration")
        for key, value in current_config.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value}")
    
    if use_optimized:
        # Pre-compute datasets once
        print("Pre-computing datasets...")
        subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
        
        train_dataset = MultiSubjectDataset(
            network_data['train']['features'], 
            network_data['train']['fmri'], 
            subject_ids
        )
        val_dataset = MultiSubjectDataset(
            network_data['val']['features'], 
            network_data['val']['fmri'], 
            subject_ids
        )
        
        print(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")
        objective = create_objective_function_optimized(network_name, train_dataset, val_dataset, seed)
    else:
        # Use original version (recreates datasets each time) con seed
        objective = create_objective_function(network_name, network_data, seed)
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Results
    print(f"\nOptimization completed!")
    print(f"Best correlation: {study.best_value:.4f}")
    
    if warm_start and len(study.trials) > 0:
        original_trial = study.trials[0]
        print(f"Your original config result: {original_trial.value:.4f}")
        if study.best_value > original_trial.value:
            improvement = study.best_value - original_trial.value
            print(f"Improvement found: +{improvement:.4f}")
    
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # Get best trail details
    best_subject_correlations = None
    if hasattr(study.best_trial, 'user_attrs') and 'subject_correlations' in study.best_trial.user_attrs:
        best_subject_correlations = study.best_trial.user_attrs['subject_correlations']
        print(f"Best trial individual subjects: {[f'{corr:.4f}' for corr in best_subject_correlations]}")
    
    # Save results
    results = {
        'network': network_name,
        'best_correlation': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'study_name': study_name,
        'seed': seed,  
        'best_subject_correlations': best_subject_correlations,  
        'storage_type': 'in_memory',
        'warm_start': warm_start
    }
    
    if warm_start and len(study.trials) > 0:
        original_trial = study.trials[0]
        results['original_correlation'] = original_trial.value
        results['improvement'] = study.best_value - original_trial.value if original_trial.value else None
    
    return results


# ==========================================
# PERSISTENT STORAGE & ADVANCED OPTIMIZATION
# ==========================================

def create_persistent_storage(storage_path="optimization_history.db"):

    storage_url = f"sqlite:///{storage_path}"
    storage = RDBStorage(storage_url)
    print(f"Storage creato/connesso: {storage_path}")
    return storage


def optimize_with_storage(network_name, network_data, n_trials=50, storage_path="optimization_history.db", 
                         study_name=None, use_optimized=True, seed=777):
    
    # Setup storage
    storage = create_persistent_storage(storage_path)
    
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{network_name}_opt_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZING {network_name.upper()} WITH PERSISTENT STORAGE")
    print(f"{'='*60}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"N trials: {n_trials}")
    print(f"Seed: {seed}")
    
    # Create or load existent study
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Studio esistente caricato - {len(study.trials)} trial già completati")
        existing_trials = len(study.trials)
    except KeyError:
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=seed),
            study_name=study_name,
            storage=storage
        )
        print("New study created")
        existing_trials = 0
    
    # Setup objective function
    if use_optimized:
        print("Pre-computing datasets...")
        subject_ids = [0, 1, 2, 3]
        
        train_dataset = MultiSubjectDataset(
            network_data['train']['features'], 
            network_data['train']['fmri'], 
            subject_ids
        )
        val_dataset = MultiSubjectDataset(
            network_data['val']['features'], 
            network_data['val']['fmri'], 
            subject_ids
        )
        
        print(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")
        objective = create_objective_function_optimized(network_name, train_dataset, val_dataset, seed)
    else:
        objective = create_objective_function(network_name, network_data, seed)
    
    # Optimize
    print(f"Starting optimization...")
    study.optimize(objective, n_trials=n_trials)
    
    # Results
    print(f"\nOptimization completed!")
    print(f"Total trials: {len(study.trials)} (new: {len(study.trials) - existing_trials})")
    print(f"Best correlation: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # Get individual subject correlations of best trial
    best_subject_correlations = None
    if hasattr(study.best_trial, 'user_attrs') and 'subject_correlations' in study.best_trial.user_attrs:
        best_subject_correlations = study.best_trial.user_attrs['subject_correlations']
        print(f"Best trial individual subjects: {[f'{corr:.4f}' for corr in best_subject_correlations]}")
    
    # Results 
    results = {
        'network': network_name,
        'study_name': study_name,
        'storage_path': storage_path,
        'best_correlation': study.best_value,
        'best_params': study.best_params,
        'best_subject_correlations': best_subject_correlations,
        'total_trials': len(study.trials),
        'new_trials': len(study.trials) - existing_trials,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def optimize_with_warm_start(network_name, network_data, n_trials=50, storage_path="optimization_history.db", 
                            use_optimized=True, seed=777, include_variants=True):

    # Setup storage
    storage = create_persistent_storage(storage_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{network_name}_warmstart_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"WARM START OPTIMIZATION: {network_name.upper()}")
    print(f"{'='*60}")
    print(f"Starting from your current best configurations...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=seed),
        study_name=study_name,
        storage=storage
    )
    
    best_configs = get_current_best_configs()
    current_config = best_configs[network_name]
    
    print(f"Your current config:")
    for key, value in current_config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    study.enqueue_trial(current_config)
    seed_trials = 1
    
    if include_variants and n_trials > 5:
        print(f"Adding {network_name} variants...")
        
        variants = []
        if network_name in ['visual', 'dorsattn']:
            variants = [
                # Variante 1: lr più basso
                {**current_config, 'lr': 5e-5, 'decay': 1.5e-3},
                # Variante 2: hidden più grande
                {**current_config, 'hidden_dim': 800, 'embedding_dim': 120},
                # Variante 3: dropout diverso
                {**current_config, 'dropout': 0.5, 'batch_size': 2048}
            ]
        elif network_name == 'sommot':
            variants = [
                # Variante 1: configurazione più grande
                {**current_config, 'hidden_dim': 900, 'embedding_dim': 180},
                # Variante 2: lr più basso
                {**current_config, 'lr': 8e-5, 'decay': 1e-3},
                # Variante 3: batch più grande
                {**current_config, 'batch_size': 2048, 'dropout': 0.55}
            ]
        elif network_name == 'multi':
            variants = [
                # Variante 1: embedding più grande
                {**current_config, 'embedding_dim': 300, 'hidden_dim': 1200},
                # Variante 2: decay più basso
                {**current_config, 'decay': 4e-4, 'lr': 8e-5},
                # Variante 3: configurazione diversa
                {**current_config, 'hidden_dim': 800, 'embedding_dim': 200, 'dropout': 0.7}
            ]
        
        # Enqueue 
        for i, variant in enumerate(variants[:3]):  
            study.enqueue_trial(variant)
            seed_trials += 1
            print(f"  Variant {i+1}: hidden_dim={variant['hidden_dim']}, embedding_dim={variant['embedding_dim']}")
    
    print(f"Enqueued {seed_trials} seed configurations")
    print(f"Will explore {n_trials - seed_trials} additional trials")
    
    # Setup objective function
    if use_optimized:
        print("Pre-computing datasets...")
        subject_ids = [0, 1, 2, 3]
        
        train_dataset = MultiSubjectDataset(
            network_data['train']['features'], 
            network_data['train']['fmri'], 
            subject_ids
        )
        val_dataset = MultiSubjectDataset(
            network_data['val']['features'], 
            network_data['val']['fmri'], 
            subject_ids
        )
        
        print(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")
        objective = create_objective_function_optimized(network_name, train_dataset, val_dataset, seed)
    else:
        objective = create_objective_function(network_name, network_data, seed)
    
    # Optimize
    print(f"Starting optimization with warm start...")
    study.optimize(objective, n_trials=n_trials)
    
    # Results
    print(f"\nWarm start optimization completed!")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best correlation: {study.best_value:.4f}")
    
    # Confronta con la tua configurazione originale
    original_trial = study.trials[0]  # Prima trial = tua config
    print(f"Your original config result: {original_trial.value:.4f}")
    
    if study.best_value > original_trial.value:
        improvement = study.best_value - original_trial.value
        print(f"Found improvement: +{improvement:.4f} correlation!")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value}")
   
    
    # Get individual subject correlations del miglior trial
    best_subject_correlations = None
    if hasattr(study.best_trial, 'user_attrs') and 'subject_correlations' in study.best_trial.user_attrs:
        best_subject_correlations = study.best_trial.user_attrs['subject_correlations']
        print(f"Best trial individual subjects: {[f'{corr:.4f}' for corr in best_subject_correlations]}")
    
    # Results 
    results = {
        'network': network_name,
        'study_name': study_name,
        'storage_path': storage_path,
        'best_correlation': study.best_value,
        'best_params': study.best_params,
        'best_subject_correlations': best_subject_correlations,
        'original_correlation': original_trial.value,
        'improvement': study.best_value - original_trial.value if original_trial.value else None,
        'total_trials': len(study.trials),
        'seed_trials': seed_trials,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'warm_start': True
    }
    
    return results


def save_optimization_results(results, filename='optimization_results.json'):
    """
    Salva risultati ottimizzazione (sia in-memory che persistent)
    """
    # Load existing results if file exists
    all_results = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_results = json.load(f)
    
    # Add new results
    all_results[results['network']] = results
    
    # Save
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    storage_type = results.get('storage_type', 'in_memory')
    print(f"Results saved to {filename} (storage: {storage_type})")
    
    if 'storage_path' in results:
        print(f"Persistent storage available at: {results['storage_path']}")


def load_optimization_results(filename='optimization_results.json'):
  
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return {}
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def show_optimization_results(filename='optimization_results.json'):

    results = load_optimization_results(filename)
    
    if not results:
        print("No optimization results found.")
        return
    
    print("="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    
    for network, result in results.items():
        print(f"\n{network.upper()} NETWORK:")
        print(f"  Best correlation: {result['best_correlation']:.4f}")
        
        if 'best_subject_correlations' in result and result['best_subject_correlations'] is not None:
            subj_corrs = result['best_subject_correlations']
            print(f"  Individual subjects: {[f'{corr:.4f}' for corr in subj_corrs]}")
            print(f"    Subject 0: {subj_corrs[0]:.4f}")
            print(f"    Subject 1: {subj_corrs[1]:.4f}")
            print(f"    Subject 2: {subj_corrs[2]:.4f}")
            print(f"    Subject 3: {subj_corrs[3]:.4f}")
        
        print(f"  N trials: {result['n_trials']}")
        
        # Storage info
        storage_type = result.get('storage_type', 'unknown')
        print(f"  Storage: {storage_type}")
        if 'storage_path' in result:
            print(f"  Database: {result['storage_path']}")
        if 'study_name' in result:
            print(f"  Study name: {result['study_name']}")
        
        print(f"  Best parameters:")
        for param, value in result['best_params'].items():
            if isinstance(value, float):
                print(f"    {param:12}: {value:.2e}")
            else:
                print(f"    {param:12}: {value}")
    

def resume_optimization(network_name, study_name, storage_path="optimization_history.db", 
                       n_trials=50, use_optimized=True, seed=777):
    """
    Resume existing optimization
    """
    
    storage = create_persistent_storage(storage_path)
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Studio '{study_name}' caricato - {len(study.trials)} trial esistenti")
        print(f"Best value corrente: {study.best_value:.4f}")
        
        return {
            'network': network_name,
            'study_name': study_name,
            'storage_path': storage_path,
            'current_trials': len(study.trials),
            'current_best': study.best_value,
            'status': 'ready_to_resume'
        }
        
    except KeyError:
        print(f"Studio '{study_name}' not found {storage_path}")
        return None
