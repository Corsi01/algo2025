"""
Utility functions for submission: model loading, ensemble creation, prediction and reconstruction
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from nilearn import  datasets
from data_utils import atlas_schaefer
from MultiSubjectModel_utils import MultiSubjectMLP


def load_optimization_results(filename='optimize_models/optimization_results.json'):
    """Load optimization results if available"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}


def get_optimized_model_configs(models_dir='optimize_models/'):
    """
    Get model configurations using optimized hyperparameters if available,
    otherwise use default configurations
    
    Args:
        models_dir: Directory containing optimization results and models
    """
    
    # Load optimization results from specific directory
    opt_results_path = os.path.join(models_dir, 'optimization_results.json')
    opt_results = load_optimization_results(opt_results_path)
    
    # Get atlas info for parcel counts
    df_atlas = atlas_schaefer()
    
    parcel_list_vis = df_atlas[df_atlas.network_yeo7 == 'Vis'].parcel_id.tolist()
    parcel_list_attn = df_atlas[df_atlas.network_yeo7 == 'DorsAttn'].parcel_id.tolist()
    parcel_list_sommot = df_atlas[df_atlas.network_yeo7 == 'SomMot'].parcel_id.tolist()
    
    parcel_list_limbic = df_atlas[df_atlas.network_yeo7 == 'Limbic'].parcel_id.tolist()
    parcel_list_control = df_atlas[df_atlas.network_yeo7 == 'Cont'].parcel_id.tolist()
    parcel_list_default = df_atlas[df_atlas.network_yeo7 == 'Default'].parcel_id.tolist()
    parcel_list_salvent = df_atlas[df_atlas.network_yeo7 == 'SalVentAttn'].parcel_id.tolist()
    parcel_list_multi = parcel_list_limbic + parcel_list_control + parcel_list_default + parcel_list_salvent
    
    # Determine input dimensions based on models directory
    if 'no_text' in models_dir:
        # Without language features: saliency + audio + audio2 (3 modalities)
        # Base: 7 * 250 * 3 = 5250, Extra saliency: 5 * 250 = 1250, Extra audio: 5 * 250 * 2 = 2500
        visual_dorsattn_input_dim = 5250 + 1250  # 6500
        sommot_input_dim = 5250 + 1250 + 2500    # 9000
        multi_input_dim = 5250                   # 5250
    else:
        # With language features: saliency + audio + audio2 + language_pooled (4 modalities)
        # Base: 7 * 250 * 4 = 7000, Extra saliency: 5 * 250 = 1250, Extra audio: 5 * 250 * 2 = 2500
        visual_dorsattn_input_dim = 7000 + 1250  # 8250
        sommot_input_dim = 7000 + 1250 + 2500    # 10750
        multi_input_dim = 7000                   # 7000
    
    # Base model configurations with default values
    base_configs = {
        'visual': {
            'input_dim': visual_dorsattn_input_dim, 
            'embedding_dim': 150, 
            'hidden_dim': 700, 
            'output_dim': len(parcel_list_vis), 
            'num_subjects': 4,
            'dropout_rate': 0.5
        },
        'dorsattn': {
            'input_dim': visual_dorsattn_input_dim, 
            'embedding_dim': 150, 
            'hidden_dim': 700,
            'output_dim': len(parcel_list_attn), 
            'num_subjects': 4,
            'dropout_rate': 0.5
        },
        'sommot': {
            'input_dim': sommot_input_dim, 
            'embedding_dim': 150, 
            'hidden_dim': 800,
            'output_dim': len(parcel_list_sommot), 
            'num_subjects': 4,
            'dropout_rate': 0.5
        },
        'multi': {
            'input_dim': multi_input_dim, 
            'embedding_dim': 256, 
            'hidden_dim': 1000,
            'output_dim': len(parcel_list_multi), 
            'num_subjects': 4,
            'dropout_rate': 0.5
        }
    }
    
    # Update with optimized parameters if available
    optimized_configs = {}
    for network_name, base_config in base_configs.items():
        config = base_config.copy()
        
        if network_name in opt_results:
            best_params = opt_results[network_name]['best_params']
            
            # Update the parameters that affect model architecture
            config['embedding_dim'] = best_params['embedding_dim']
            config['hidden_dim'] = best_params['hidden_dim']
            config['dropout_rate'] = best_params['dropout']
            
            print(f"Using optimized config for {network_name} ({'no_text' if 'no_text' in models_dir else 'with_text'}):")
            print(f"  input_dim: {config['input_dim']}")
            print(f"  embedding_dim: {config['embedding_dim']}")
            print(f"  hidden_dim: {config['hidden_dim']}")
            print(f"  dropout_rate: {config['dropout_rate']}")
        else:
            print(f"Using default config for {network_name} ({'no_text' if 'no_text' in models_dir else 'with_text'})")
        
        optimized_configs[network_name] = config
    
    # Return the configs along with parcel lists
    return optimized_configs, {
        'vis': parcel_list_vis,
        'attn': parcel_list_attn, 
        'sommot': parcel_list_sommot,
        'multi': parcel_list_multi
    }


class EnsembleNetworkModel(nn.Module):
    def __init__(self, visual_model, dorsattn_model, sommot_model, multi_model,
                 parcel_list_visual, parcel_list_dorsattn, parcel_list_sommot, parcel_list_multi,
                 feature_configs, subject_mappings, total_parcels=1000):
        """
        Ensemble model that combines 4 MultiSubjectMLPs for different neural networks
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
        """Freeze the backbone parameters (optional)"""
        for model in [self.visual_model, self.dorsattn_model, self.sommot_model, self.multi_model]:
            for param in model.parameters():
                param.requires_grad = False
    
    def _prepare_features(self, features_standard, features_different, network_name):
        """Prepare the right features for each network"""
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
            full_prediction: (batch_size, total_parcels) with all predictions reassembled
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
        
        # Separate predictions for each network using the right subject
        visual_pred = self.visual_model.predict_subject(visual_features, visual_subj_id)
        dorsattn_pred = self.dorsattn_model.predict_subject(dorsattn_features, dorsattn_subj_id)
        sommot_pred = self.sommot_model.predict_subject(sommot_features, sommot_subj_id)
        multi_pred = self.multi_model.predict_subject(multi_features, multi_subj_id)
        
        # Reconstruct the complete original order
        full_prediction = torch.zeros(batch_size, self.total_parcels, device=device)
        
        # Put predictions in the correct positions
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


def create_ensemble_from_optimized_models(models_dir='optimize_models/', 
                                        model_configs=None, 
                                        parcel_lists=None,
                                        total_parcels=1000):
    """
    Create ensemble from optimized MultiSubjectMLP saved models
    
    Args:
        models_dir: Directory containing the models (e.g., 'optimize_models/' or 'optimize_models_no_text/')
        model_configs: Model configurations from get_optimized_model_configs()
        parcel_lists: Parcel lists from get_optimized_model_configs()
        total_parcels: Total number of parcels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model paths
    model_paths = {
        'visual': os.path.join(models_dir, 'models', 'full_visual_model.pth'),
        'dorsattn': os.path.join(models_dir, 'models', 'full_dorsattn_model.pth'),
        'sommot': os.path.join(models_dir, 'models', 'full_sommot_model.pth'),
        'multi': os.path.join(models_dir, 'models', 'full_multi_model.pth')
    }
    
    # Load each MultiSubjectMLP with its specific optimized configuration
    models = {}
    for network_name, model_path in model_paths.items():
        print(f"Loading {network_name} model from {model_path}")
        
        # Create model with optimized config
        config = model_configs[network_name]
        model = MultiSubjectMLP(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            dropout_rate=config['dropout_rate'],
            num_subjects=config['num_subjects']
        )
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        models[network_name] = model
        
        print(f"  Model architecture: input_dim={config['input_dim']}, "
              f"embedding_dim={config['embedding_dim']}, hidden_dim={config['hidden_dim']}, "
              f"output_dim={config['output_dim']}")
    
    # Feature configurations - adjust based on directory
    if 'no_text' in models_dir:
        # Without language features: multi uses subset of features (without language)
        feature_configs = {
            'visual': {'type': 'same', 'indices': None},     
            'dorsattn': {'type': 'same', 'indices': None},  
            'sommot': {'type': 'different', 'indices': None}, 
            'multi': {'type': 'subset', 'indices': slice(0, 5250)}  # Only non-language features
        }
    else:
        # With language features: multi uses subset of all features
        feature_configs = {
            'visual': {'type': 'same', 'indices': None},     
            'dorsattn': {'type': 'same', 'indices': None},  
            'sommot': {'type': 'different', 'indices': None}, 
            'multi': {'type': 'subset', 'indices': slice(0, 7000)}  # All features
        }

    # Subject mappings
    subject_mappings = {
        'visual': {1: 0, 2: 1, 3: 2, 5: 3},
        'dorsattn': {1: 0, 2: 1, 3: 2, 5: 3},
        'sommot': {1: 0, 2: 1, 3: 2, 5: 3},
        'multi': {1: 0, 2: 1, 3: 2, 5: 3}
    }
    
    # Create ensemble
    ensemble = EnsembleNetworkModel(
        models['visual'], models['dorsattn'], models['sommot'], models['multi'],
        parcel_lists['vis'], parcel_lists['attn'], parcel_lists['sommot'], parcel_lists['multi'],
        feature_configs, subject_mappings, total_parcels
    )
    
    return ensemble.to(device)


def make_prediction(ensemble, features_standard, features_different, subject_id):
    """
    Make predictions using ensemble model
    
    Args:
        ensemble: the ensemble model
        features_standard: standard features for visual/dorsattn/multi networks
        features_different: different features for sommot network  
        subject_id: subject ID (1, 2, 3, or 5)
    
    Returns:
        predictions: numpy array (batch_size, 1000) with predictions for all parcels
    """
    ensemble.eval()
    
    with torch.no_grad():
        if not isinstance(features_standard, torch.Tensor):
            features_standard = torch.tensor(features_standard, dtype=torch.float32)
        if not isinstance(features_different, torch.Tensor):
            features_different = torch.tensor(features_different, dtype=torch.float32)
        
        device = next(ensemble.parameters()).device
        features_standard = features_standard.to(device)
        features_different = features_different.to(device)
        
        # Predict
        predictions = ensemble(features_standard, features_different, subject_id)
        predictions = predictions.cpu().numpy()
    
    return predictions
