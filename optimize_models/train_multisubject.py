"""
Main script for multi-subject fMRI encoding training with hyperparameter optimization
FIXED: Added direct full training without intermediate steps
"""

import os
import torch
import numpy as np
from utilities import (
    align_features_and_fmri_samples_extended,
    train_multitask_model,
    train_final_model_on_all_data,
    evaluate_multitask_model,
    evaluate_multitask_model_by_network,
    create_ensemble_from_saved_models,
    make_prediction,
    get_network_parcels,
    setup_model_configurations,
    set_seed
)
from optimization_utils import (
    optimize_network_hyperparams,
    save_optimization_results,
    load_optimization_results,
    show_optimization_results
)
from data_utils import load_stimulus_features, load_fmri, atlas_schaefer, compute_encoding_accuracy

class MultiSubjectTrainer:
    
    def __init__(self, root_data_dir='../data', random_seed=777):
        """
        Args:
            root_data_dir: directory containing data
            random_seed: seed for reproducibility
        """
        self.root_data_dir = root_data_dir
        self.random_seed = random_seed
        
        # Setup
        set_seed(self.random_seed)
        
        # Features alignment parameters
        self.modality = "all"
        self.excluded_samples_start = 5
        self.excluded_samples_end = 5
        self.hrf_delay = 2
        self.stimulus_window = 7
        
        # Train / Val list
        self.movies_train = [
            "friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05",
            "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"
        ]
        self.movies_val = ["friends-s06"]
        
        # Subject IDs
        self.subject_ids = [1, 2, 3, 5]
        
        self._load_base_data()
        self._setup_network_parcels()
        self._setup_model_configs()
        
        print("MultiSubjectTrainer initialized successfully!")
    
    def _load_base_data(self):
        """Load base fMRI and features"""
        print("Loading base data...")
        
        # Load stimulus features
        self.features = load_stimulus_features(self.root_data_dir, self.modality)
        print(f"Available feature modalities: {list(self.features.keys())}")
        
        # Load fMRI data
        self.fmri_data = {}
        for subject_id in self.subject_ids:
            self.fmri_data[subject_id] = load_fmri(self.root_data_dir, subject_id)
            print(f"Loaded fMRI data for subject {subject_id}")
        
        print("Base data loaded successfully!")
    
    def _setup_network_parcels(self):
        self.df_atlas = atlas_schaefer()
        self.parcel_lists = get_network_parcels(self.df_atlas)
        
        print(f"Network parcel counts:")
        for network, parcels in self.parcel_lists.items():
            print(f"  {network:10}: {len(parcels):3d} parcels")
    
    def _setup_model_configs(self):
        self.model_configs, self.feature_configs, self.subject_mappings = setup_model_configurations(self.parcel_lists)
    
    def prepare_data_for_network(self, network_name, visual_extra_params=None, audio_extra_params=None):
        """    
        Args:
            network_name:  ('visual', 'dorsattn', 'sommot', 'multi')
            visual_extra_params: extra params for memory visual features
            audio_extra_params: extra params for memory audio features
        
        Returns:
            dict with data of all subjects
        """
        print(f"\nPreparing data for {network_name} network...")
        
        # Subject-specific parameters visual extra
        visual_extra_subject_params = {}
        audio_extra_subject_params = {}
        
        if network_name in ['visual', 'dorsattn'] and visual_extra_params is None:
            # Default visual extra params 
            visual_extra_subject_params = {
                1: {'hrf_delay': 11, 'stimulus_window': 5},
                2: {'hrf_delay': 11, 'stimulus_window': 5},
                3: {'hrf_delay': 10, 'stimulus_window': 5},
                5: {'hrf_delay': 12, 'stimulus_window': 5}
            }
        elif network_name == 'sommot' and visual_extra_params is None and audio_extra_params is None:
            # Default params sommot network
            visual_extra_subject_params = {
                1: {'hrf_delay': 14, 'stimulus_window': 5},
                2: {'hrf_delay': 14, 'stimulus_window': 5},
                3: {'hrf_delay': 14, 'stimulus_window': 5},
                5: {'hrf_delay': 14, 'stimulus_window': 5}
            }
            audio_extra_subject_params = {
                1: {'hrf_delay': 10, 'stimulus_window': 5},
                2: {'hrf_delay': 10, 'stimulus_window': 5},
                3: {'hrf_delay': 10, 'stimulus_window': 5},
                5: {'hrf_delay': 10, 'stimulus_window': 5}
            }
        
        # Select parcels corresponding to network
        if network_name == 'visual':
            target_parcels = self.parcel_lists['vis']
        elif network_name == 'dorsattn':
            target_parcels = self.parcel_lists['dorsattn']
        elif network_name == 'sommot':
            target_parcels = self.parcel_lists['sommot']
        elif network_name == 'multi':
            target_parcels = self.parcel_lists['multi']
        else:
            raise ValueError(f"Unknown network: {network_name}")
        
        train_data = {'features': [], 'fmri': []}
        val_data = {'features': [], 'fmri': []}
        
        for subject_id in self.subject_ids:
            print(f"  Processing subject {subject_id}...")
            
            # Get subject-specific params
            visual_params = visual_extra_subject_params.get(subject_id, visual_extra_params)
            audio_params = audio_extra_subject_params.get(subject_id, audio_extra_params)
            
            # Training data
            features_train, fmri_train = align_features_and_fmri_samples_extended(
                self.features, self.fmri_data[subject_id],
                excluded_samples_start=self.excluded_samples_start,
                excluded_samples_end=self.excluded_samples_end,
                hrf_delay=self.hrf_delay,
                stimulus_window=self.stimulus_window,
                movies=self.movies_train,
                visual_extra_params=visual_params,
                audio_extra_params=audio_params
            )
            
            # Validation data
            features_val, fmri_val = align_features_and_fmri_samples_extended(
                self.features, self.fmri_data[subject_id],
                excluded_samples_start=self.excluded_samples_start,
                excluded_samples_end=self.excluded_samples_end,
                hrf_delay=self.hrf_delay,
                stimulus_window=self.stimulus_window,
                movies=self.movies_val,
                visual_extra_params=visual_params,
                audio_extra_params=audio_params
            )
            
            # Select target parcels
            fmri_train_network = fmri_train[:, target_parcels]
            fmri_val_network = fmri_val[:, target_parcels]
            
            if network_name == 'multi':
                features_train = features_train[:, :7000]
                features_val = features_val[:, :7000]
            
            train_data['features'].append(features_train)
            train_data['fmri'].append(fmri_train_network)
            val_data['features'].append(features_val)
            val_data['fmri'].append(fmri_val_network)
            
            print(f"    Train: {features_train.shape} -> {fmri_train_network.shape}")
            print(f"    Val:   {features_val.shape} -> {fmri_val_network.shape}")
        
        return {
            'train': train_data,
            'val': val_data,
            'target_parcels': target_parcels
        }
    
    def train_full_model_direct(self, network_name, use_best_params=False):
        """
        Train network model directly on all data (train + val) without intermediate training
        """
        print(f"\n{'='*60}")
        print(f"TRAINING FULL {network_name.upper()} MODEL DIRECTLY ON ALL DATA")
        print(f"{'='*60}")
        
        # Prepare data for this network
        network_data = self.prepare_data_for_network(network_name)
        
        # Combine train and val data for each subject
        features_all_dict = {}
        fmri_all_dict = {}
        
        for i, subject_id in enumerate(self.subject_ids):
            # Combine features
            features_combined = np.concatenate([
                network_data['train']['features'][i],
                network_data['val']['features'][i]
            ], axis=0)
            
            # Combine fMRI
            fmri_combined = np.concatenate([
                network_data['train']['fmri'][i],
                network_data['val']['fmri'][i]
            ], axis=0)
            
            features_all_dict[subject_id] = features_combined
            fmri_all_dict[subject_id] = fmri_combined
            
            print(f"Subject {subject_id}: Combined {features_combined.shape} -> {fmri_combined.shape}")
        
        # Get model dimensions
        input_dim = features_all_dict[self.subject_ids[0]].shape[1]
        output_dim = fmri_all_dict[self.subject_ids[0]].shape[1]
        
        # Determine hyperparameters
        if use_best_params:
            # Load best params from optimization
            opt_results = load_optimization_results()
            if network_name in opt_results:
                best_params = opt_results[network_name]['best_params']
                print(f"Using optimized hyperparameters:")
                for key, value in best_params.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2e}")
                    else:
                        print(f"  {key}: {value}")
                
                decay = best_params['decay']
                embedding_dim = best_params['embedding_dim']
                hidden_dim = best_params['hidden_dim']
                dropout_rate = best_params['dropout']
                batch_size = best_params['batch_size']
                lr = best_params['lr']
            else:
                print(f"No optimization results found for {network_name}, using defaults")
                use_best_params = False
        
        if not use_best_params:
            # Use default params based on network
            if network_name in ['visual', 'dorsattn']:
                decay, dropout_rate, embedding_dim, hidden_dim, batch_size, lr = 2e-3, 0.6, 150, 700, 1024, 1e-4
            elif network_name == 'sommot':
                decay, dropout_rate, embedding_dim, hidden_dim, batch_size, lr = 2e-3, 0.6, 150, 800, 1024, 1e-4
            elif network_name == 'multi':
                decay, dropout_rate, embedding_dim, hidden_dim, batch_size, lr = 6e-4, 0.6, 256, 1000, 1024, 1e-4
            else:
                decay, dropout_rate, embedding_dim, hidden_dim, batch_size, lr = 3e-4, 0.6, 150, 700, 1024, 1e-4
            
            print(f"Using default hyperparameters:")
            print(f"  embedding_dim: {embedding_dim}")
            print(f"  hidden_dim: {hidden_dim}")
            print(f"  dropout: {dropout_rate}")
            print(f"  decay: {decay:.2e}")
            print(f"  batch_size: {batch_size}")
            print(f"  lr: {lr:.2e}")
        
        print(f"Model config: input_dim={input_dim}, output_dim={output_dim}")
        
        # Train final model on all data using the new dedicated function
        full_model = train_final_model_on_all_data(
            features_all_dict=features_all_dict,
            fmri_all_dict=fmri_all_dict,
            input_dim=input_dim,
            output_dim=output_dim,
            decay=decay,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            lr=lr,
            seed=self.random_seed
        )
        
        print(f"Full {network_name} model training completed!")
        return full_model
    
    def train_all_full_models_direct(self, use_best_params=False):
        """
        Train all network models directly on all data (train + val) without intermediate steps
        """
        print("="*80)
        print("TRAINING ALL FULL MODELS DIRECTLY ON ALL DATA")
        print("="*80)
        
        full_models = {}
        networks = ['visual', 'dorsattn', 'sommot', 'multi']
        
        for network_name in networks:
            print(f"\n{'-'*60}")
            print(f"Training {network_name} network...")
            print(f"{'-'*60}")
            
            full_model = self.train_full_model_direct(network_name, use_best_params=use_best_params)
            full_models[network_name] = full_model
        
        return full_models
    
    def optimize_network(self, network_name, n_trials=50):
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZING {network_name.upper()} NETWORK")
        print(f"{'='*60}")
        
        # Prepare data
        network_data = self.prepare_data_for_network(network_name)
        
        # Optimize
        results = optimize_network_hyperparams(
            network_name=network_name,
            network_data=network_data,
            n_trials=n_trials,
            study_name=f"{network_name}_optimization"
        )
        
        # Save results
        save_optimization_results(results)
        
        return results
    
    def optimize_all_networks(self, n_trials=50):
       
        print("="*80)
        print("OPTIMIZING ALL NETWORKS")
        print("="*80)
        
        networks = ['visual', 'dorsattn', 'sommot', 'multi']
        all_results = {}
        
        for network_name in networks:
            results = self.optimize_network(network_name, n_trials)
            all_results[network_name] = results
            
            print(f"\n{network_name.upper()} OPTIMIZATION COMPLETED:")
            print(f"Best correlation: {results['best_correlation']:.4f}")
            print(f"Best params: {results['best_params']}")
        
        return all_results
    
    def train_network_model(self, network_name, use_best_params=False):
        """
        Train network model con possibilità di usare best params da ottimizzazione
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {network_name.upper()} NETWORK MODEL")
        print(f"{'='*60}")
        
        network_data = self.prepare_data_for_network(network_name)
        
        # Get dimensions
        input_dim = network_data['train']['features'][0].shape[1]
        output_dim = network_data['train']['fmri'][0].shape[1]
        
        # Determine hyperparameters
        if use_best_params:
            # Load best params from optimization
            opt_results = load_optimization_results()
            if network_name in opt_results:
                best_params = opt_results[network_name]['best_params']
                print(f"Using optimized hyperparameters:")
                for key, value in best_params.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2e}")
                    else:
                        print(f"  {key}: {value}")
                
                # Use parameters from best_params
                decay = best_params['decay']
                embedding_dim = best_params['embedding_dim']
                hidden_dim = best_params['hidden_dim']
                dropout_rate = best_params['dropout']
                batch_size = best_params['batch_size']
                lr = best_params['lr']
            else:
                print(f"No optimization results found for {network_name}, using defaults")
                use_best_params = False
        
        if not use_best_params:
            # Use default params based on network
            model_config = self.model_configs[network_name]
            embedding_dim = model_config['embedding_dim']
            hidden_dim = model_config['hidden_dim']
            
            if network_name in ['visual', 'dorsattn']:
                dropout_rate, decay, batch_size, lr = 0.6, 2e-3, 1024, 1e-4
            elif network_name == 'sommot':
                dropout_rate, decay, batch_size, lr = 0.6, 2e-3, 1024, 1e-4
            elif network_name == 'multi':
                dropout_rate, decay, batch_size, lr = 0.6, 6e-4, 1024, 1e-4
            else:
                dropout_rate, decay, batch_size, lr = 0.6, 3e-4, 1024, 1e-4
            
            print(f"Using default hyperparameters:")
            print(f"  embedding_dim: {embedding_dim}")
            print(f"  hidden_dim: {hidden_dim}")
            print(f"  dropout: {dropout_rate}")
            print(f"  decay: {decay:.2e}")
            print(f"  batch_size: {batch_size}")
            print(f"  lr: {lr:.2e}")
        
        print(f"Model config: input_dim={input_dim}, output_dim={output_dim}")
        
        # Train model with specified hyperparameters
        model = train_multitask_model(
            features_list=network_data['train']['features'],
            fmri_list=network_data['train']['fmri'],
            val_features_list=network_data['val']['features'],
            val_fmri_list=network_data['val']['fmri'],
            input_dim=input_dim,
            output_dim=output_dim,
            decay=decay,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            lr=lr,
            seed=777
        )
        
        # Evaluate
        print(f"\nEvaluating {network_name} model...")
        features_val_dict = {subj_id: network_data['val']['features'][i] 
                           for i, subj_id in enumerate(self.subject_ids)}
        fmri_val_dict = {subj_id: network_data['val']['fmri'][i] 
                        for i, subj_id in enumerate(self.subject_ids)}
        
        evaluate_multitask_model(model, features_val_dict, fmri_val_dict)
        
        return model, network_data
    
    def train_all_networks(self, use_best_params=False):
        """
        Training di tutti i network con opzione best params
        """
        print("Starting training for all networks...")
        
        models = {}
        network_data_all = {}
        
        networks = ['visual', 'dorsattn', 'sommot', 'multi']
        
        for network_name in networks:
            models[network_name], network_data_all[network_name] = self.train_network_model(
                network_name, use_best_params=use_best_params
            )
        
        return models, network_data_all
    
    def train_full_models(self, models, network_data_all, use_best_params=False):
        """
        Trains final models on all data (train + val) with correct configurations
        """
        print("\n" + "="*60)
        print("TRAINING FULL MODELS ON ALL DATA")
        print("="*60)
        
        full_models = {}
        
        for network_name, model in models.items():
            print(f"\nTraining full {network_name} model...")
            
            network_data = network_data_all[network_name]
            
            features_train_dict = {subj_id: network_data['train']['features'][i] 
                                 for i, subj_id in enumerate(self.subject_ids)}
            fmri_train_dict = {subj_id: network_data['train']['fmri'][i] 
                             for i, subj_id in enumerate(self.subject_ids)}
            features_val_dict = {subj_id: network_data['val']['features'][i] 
                               for i, subj_id in enumerate(self.subject_ids)}
            fmri_val_dict = {subj_id: network_data['val']['fmri'][i] 
                           for i, subj_id in enumerate(self.subject_ids)}
            
            # Combine all data for each subject
            features_all_dict = {}
            fmri_all_dict = {}
            
            for subject_id in self.subject_ids:
                features_all_dict[subject_id] = np.concatenate([
                    features_train_dict[subject_id],
                    features_val_dict[subject_id]
                ], axis=0)
                
                fmri_all_dict[subject_id] = np.concatenate([
                    fmri_train_dict[subject_id],
                    fmri_val_dict[subject_id]
                ], axis=0)
            
            model_config = self.model_configs[network_name]
            
            # Determine hyperparameters with batch_size and lr
            if use_best_params:
                opt_results = load_optimization_results()
                if network_name in opt_results:
                    best_params = opt_results[network_name]['best_params']
                    decay = best_params['decay']
                    dropout = best_params['dropout']
                    emb_dim = best_params['embedding_dim']
                    hid_dim = best_params['hidden_dim']
                    batch_size = best_params['batch_size']
                    lr = best_params['lr']
                    print(f"Using optimized params for {network_name}")
                else:
                    use_best_params = False
            
            if not use_best_params:
                if network_name in ['visual', 'dorsattn']:
                    decay, dropout, emb_dim, hid_dim, batch_size, lr = 2e-3, 0.6, 150, 700, 1024, 1e-4
                elif network_name == 'sommot':
                    decay, dropout, emb_dim, hid_dim, batch_size, lr = 2e-3, 0.6, 150, 800, 1024, 1e-4
                elif network_name == 'multi':
                    decay, dropout, emb_dim, hid_dim, batch_size, lr = 6e-4, 0.6, 256, 1000, 1024, 1e-4
                else:
                    decay, dropout, emb_dim, hid_dim, batch_size, lr = 3e-4, 0.6, 150, 700, 1024, 1e-4
                
                print(f"Using default params for {network_name}")
            
            print(f"Config: input_dim={model_config['input_dim']}, embedding_dim={emb_dim}, hidden_dim={hid_dim}")
            print(f"Training params: decay={decay:.2e}, dropout={dropout}, batch_size={batch_size}, lr={lr:.2e}")
            
            # Train with the new dedicated function
            full_model = train_final_model_on_all_data(
                features_all_dict=features_all_dict,
                fmri_all_dict=fmri_all_dict,
                input_dim=model_config['input_dim'],
                output_dim=model_config['output_dim'],
                decay=decay,
                dropout_rate=dropout,
                embedding_dim=emb_dim,
                hidden_dim=hid_dim,
                batch_size=batch_size,
                lr=lr,
                seed=777
            )
            
            full_models[network_name] = full_model
        
        return full_models
    
    def save_models(self, models, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        
        for network_name, model in models.items():
            save_path = os.path.join(save_dir, f'{network_name}_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved {network_name} model to {save_path}")
    
    def create_and_evaluate_ensemble(self, model_paths=None):
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        if model_paths is None:
            model_paths = {
                'visual': 'models/visual_model.pth',
                'dorsattn': 'models/dorsattn_model.pth',
                'sommot': 'models/sommot_model.pth',
                'multi': 'models/multi_model.pth'
            }
        
        # Load best parameters to create correct model configs
        opt_results = load_optimization_results()
        
        # Create updated model configs with best parameters
        updated_model_configs = {}
        
        for network_name in ['visual', 'dorsattn', 'sommot', 'multi']:
            # Start with default config
            base_config = self.model_configs[network_name].copy()
            
            # If we have optimization results, use those
            if network_name in opt_results:
                best_params = opt_results[network_name]['best_params']
                
                # Update model dimensions
                base_config['embedding_dim'] = best_params['embedding_dim']
                base_config['hidden_dim'] = best_params['hidden_dim']
                
                print(f"Updated {network_name} config with optimization results:")
                print(f"  embedding_dim: {best_params['embedding_dim']}")
                print(f"  hidden_dim: {best_params['hidden_dim']}")
            else:
                print(f"Using default config for {network_name} (no optimization results)")
            
            updated_model_configs[network_name] = base_config
        
        ensemble = create_ensemble_from_saved_models(
            model_paths['visual'],
            model_paths['dorsattn'],
            model_paths['sommot'],
            model_paths['multi'],
            self.parcel_lists['vis'],
            self.parcel_lists['dorsattn'],
            self.parcel_lists['sommot'],
            self.parcel_lists['multi'],
            updated_model_configs,
            self.feature_configs,
            self.subject_mappings,
            total_parcels=1000
        )
        
        print("Ensemble model created successfully!")
        
        print("\nPreparing validation data for ensemble evaluation...")
        
        # Standard features (used by visual, dorsattn, multi)
        val_features_standard = {}
        # Different features (used by sommot)
        val_features_different = {}
        val_fmri_full = {}
        
        for subject_id in self.subject_ids:
            print(f"  Preparing subject {subject_id}...")
            
            # Standard features (visual network type)
            visual_params = {
                1: {'hrf_delay': 11, 'stimulus_window': 5},
                2: {'hrf_delay': 11, 'stimulus_window': 5},
                3: {'hrf_delay': 10, 'stimulus_window': 5},
                5: {'hrf_delay': 12, 'stimulus_window': 5}
            }[subject_id]
            
            features_standard, fmri_val = align_features_and_fmri_samples_extended(
                self.features, self.fmri_data[subject_id],
                excluded_samples_start=self.excluded_samples_start,
                excluded_samples_end=self.excluded_samples_end,
                hrf_delay=self.hrf_delay,
                stimulus_window=self.stimulus_window,
                movies=self.movies_val,
                visual_extra_params=visual_params
            )
            
            # Different features (sommot network type)
            sommot_visual_params = {'hrf_delay': 14, 'stimulus_window': 5}
            sommot_audio_params = {'hrf_delay': 10, 'stimulus_window': 5}
            
            features_different, _ = align_features_and_fmri_samples_extended(
                self.features, self.fmri_data[subject_id],
                excluded_samples_start=self.excluded_samples_start,
                excluded_samples_end=self.excluded_samples_end,
                hrf_delay=self.hrf_delay,
                stimulus_window=self.stimulus_window,
                movies=self.movies_val,
                visual_extra_params=sommot_visual_params,
                audio_extra_params=sommot_audio_params
            )
            
            val_features_standard[subject_id] = features_standard
            val_features_different[subject_id] = features_different
            val_fmri_full[subject_id] = fmri_val
        
        # Evaluate ensemble
        print("\nEvaluating ensemble model...")
        
        for subject_id in self.subject_ids:
            print(f"\n--- Subject {subject_id} ---")
            
            y_pred = make_prediction(
                ensemble,
                val_features_standard[subject_id],
                val_features_different[subject_id],
                subject_id
            )
            
            print('gt type', type(val_fmri_full[subject_id]))
            print('gt shape', val_fmri_full[subject_id].shape)
            # Use compute_encoding_accuracy correctly
            # y_pred is already assembled on all 1000 zones from ensemble
            # val_fmri_full[subject_id] are ground truth for all 1000 zones
            compute_encoding_accuracy(
                val_fmri_full[subject_id],  # Ground truth (samples, 1000)
                y_pred,                     # Predictions (samples, 1000) 
                subject_id,                 # Subject ID for plot
                'all'                  # Modality name
            )
        
        return ensemble
    
    def run_complete_pipeline(self, use_best_params=False):
        """
        Pipeline completa con possibilità di usare best params
        """
        print("="*80)
        print("STARTING COMPLETE MULTI-SUBJECT TRAINING PIPELINE")
        print("="*80)
        
        # 1. Train all network models
        print("\nStep 1: Training individual network models...")
        models, network_data_all = self.train_all_networks(use_best_params=use_best_params)
        
        # 2. Save models
        print("\nStep 2: Saving models...")
        self.save_models(models, 'models')
        
        # 3. Train final models on all data
        print("\nStep 3: Training full models on all data...")
        full_models = self.train_full_models(models, network_data_all, use_best_params=use_best_params)
        
        # 4. Save final models
        print("\nStep 4: Saving full models...")
        full_model_names = {f'full_{name}': model for name, model in full_models.items()}
        self.save_models(full_model_names, 'models')
        
        # 5. Create and evaluate ensemble
        print("\nStep 5: Creating and evaluating ensemble...")
        ensemble = self.create_and_evaluate_ensemble()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'models': models,
            'full_models': full_models,
            'ensemble': ensemble,
            'network_data': network_data_all
        }
    
    def run_direct_full_training_pipeline(self, use_best_params=False):
        """
        NEW: Pipeline che addestra direttamente su tutti i dati senza passaggi intermedi
        """
        print("="*80)
        print("STARTING DIRECT FULL TRAINING PIPELINE")
        print("="*80)
        print("This will train models directly on ALL data (train+val) without intermediate steps")
        
        # 1. Train all full models directly
        print("\nStep 1: Training all full models directly on all data...")
        full_models = self.train_all_full_models_direct(use_best_params=use_best_params)
        
        # 2. Save full models with correct naming for ensemble
        print("\nStep 2: Saving full models...")
        # Save both as 'full_X' and 'X' for ensemble compatibility
        all_models_to_save = {}
        for name, model in full_models.items():
            all_models_to_save[f'full_{name}'] = model  # For record keeping
            all_models_to_save[name] = model            # For ensemble loading
        
        self.save_models(all_models_to_save, 'models')
        
        # 3. Create and evaluate ensemble
        print("\nStep 3: Creating and evaluating ensemble...")
        ensemble = self.create_and_evaluate_ensemble()
        
        print("\n" + "="*80)
        print("DIRECT FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'full_models': full_models,
            'ensemble': ensemble
        }


def main():
    """
    Execute training with optimization options
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Subject fMRI Encoding Training with Optimization')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    parser.add_argument('--network', type=str, choices=['visual', 'dorsattn', 'sommot', 'multi', 'all'], 
                       default='all', help='Which network to train')
    parser.add_argument('--full_training', action='store_true', 
                       help='Train on all data (train+val) for final models')
    parser.add_argument('--evaluate_only', action='store_true', 
                       help='Only evaluate existing models')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    
    # Optimization arguments
    parser.add_argument('--optimize', action='store_true', 
                       help='Optimize hyperparameters')
    parser.add_argument('--n_trials', type=int, default=50, 
                       help='Number of optimization trials')
    parser.add_argument('--use_best_params', action='store_true', 
                       help='Use best parameters from optimization')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultiSubjectTrainer(
        root_data_dir=args.data_dir,
        random_seed=args.seed
    )
    
    if args.optimize:
        # Optimization mode
        if args.network == 'all':
            print("Optimizing all networks...")
            results = trainer.optimize_all_networks(n_trials=args.n_trials)
            
            print("\n" + "="*80)
            print("OPTIMIZATION SUMMARY")
            print("="*80)
            for network, result in results.items():
                print(f"{network:10}: {result['best_correlation']:.4f} correlation")
        else:
            print(f"Optimizing {args.network} network...")
            result = trainer.optimize_network(args.network, n_trials=args.n_trials)
            print(f"\nOptimization completed for {args.network}")
            print(f"Best correlation: {result['best_correlation']:.4f}")
    
    elif args.evaluate_only:
        print("Evaluating existing ensemble...")
        ensemble = trainer.create_and_evaluate_ensemble()
        
    elif args.network == 'all':
        # MODIFIED: Choose pipeline based on full_training flag
        if args.full_training:
            print("Using DIRECT FULL TRAINING pipeline...")
            results = trainer.run_direct_full_training_pipeline(use_best_params=args.use_best_params)
        else:
            print("Using STANDARD pipeline (train/val split then full training)...")
            results = trainer.run_complete_pipeline(use_best_params=args.use_best_params)
        
    else:
        print(f"Training {args.network} network only...")
        if args.full_training:
            print("Training FULL model directly on all data...")
            
            # Use the direct full training method
            full_model = trainer.train_full_model_direct(args.network, use_best_params=args.use_best_params)
            trainer.save_models({f'full_{args.network}': full_model}, args.model_dir)
            
        else:
            model, network_data = trainer.train_network_model(args.network, use_best_params=args.use_best_params)
            trainer.save_models({args.network: model}, args.model_dir)


if __name__ == "__main__":
    main()


# Optimization utility functions
def optimize_single_network(network_name, n_trials=50):
    """Quick function to optimize a single network"""
    trainer = MultiSubjectTrainer()
    result = trainer.optimize_network(network_name, n_trials)
    print(f"\nOptimization completed for {network_name}")
    print(f"Best correlation: {result['best_correlation']:.4f}")
    return result


def quick_optimize_all(n_trials=30):
    """Quick optimization of all networks with fewer trials"""
    trainer = MultiSubjectTrainer()
    return trainer.optimize_all_networks(n_trials=n_trials)


def quick_train_with_best():
    """Quick training with best parameters using DIRECT FULL TRAINING"""
    trainer = MultiSubjectTrainer()
    return trainer.run_direct_full_training_pipeline(use_best_params=True)


def train_visual_network_only():
    trainer = MultiSubjectTrainer()
    print("Training Visual Network...")
    visual_model, visual_data = trainer.train_network_model('visual', use_best_params=True)
    torch.save(visual_model.state_dict(), 'models/visual_model.pth')
    
    features_train_dict = {subj_id: visual_data['train']['features'][i] 
                         for i, subj_id in enumerate(trainer.subject_ids)}
    fmri_train_dict = {subj_id: visual_data['train']['fmri'][i] 
                     for i, subj_id in enumerate(trainer.subject_ids)}
    features_val_dict = {subj_id: visual_data['val']['features'][i] 
                       for i, subj_id in enumerate(trainer.subject_ids)}
    fmri_val_dict = {subj_id: visual_data['val']['fmri'][i] 
                   for i, subj_id in enumerate(trainer.subject_ids)}
    
    # Combine all data for final training
    features_all_dict = {}
    fmri_all_dict = {}
    
    for subject_id in trainer.subject_ids:
        features_all_dict[subject_id] = np.concatenate([
            features_train_dict[subject_id],
            features_val_dict[subject_id]
        ], axis=0)
        
        fmri_all_dict[subject_id] = np.concatenate([
            fmri_train_dict[subject_id],
            fmri_val_dict[subject_id]
        ], axis=0)
    
    opt_results = load_optimization_results()
    if 'visual' in opt_results:
        best_params = opt_results['visual']['best_params']
        decay = best_params['decay']
        dropout_rate = best_params['dropout']
        embedding_dim = best_params['embedding_dim']
        hidden_dim = best_params['hidden_dim']
        batch_size = best_params['batch_size']
        lr = best_params['lr']
    else:
        decay, dropout_rate, embedding_dim, hidden_dim, batch_size, lr = 2e-3, 0.6, 150, 700, 1024, 1e-4
    
    full_visual_model = train_final_model_on_all_data(
        features_all_dict=features_all_dict,
        fmri_all_dict=fmri_all_dict,
        input_dim=8250,
        output_dim=len(trainer.parcel_lists['vis']),
        decay=decay,
        dropout_rate=dropout_rate,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        lr=lr,
        seed=777
    )
    
    torch.save(full_visual_model.state_dict(), 'models/full_visual_model.pth')
    print("Visual network training completed!")


def train_direct_full_visual_only():
    """Train only visual network directly on all data"""
    trainer = MultiSubjectTrainer()
    print("Training Visual Network DIRECTLY on all data...")
    full_visual_model = trainer.train_full_model_direct('visual', use_best_params=True)
    trainer.save_models({'full_visual': full_visual_model, 'visual': full_visual_model}, 'models')
    print("Direct full visual network training completed!")


def check_model_files():
    import os
    
    model_dir = 'models'
    expected_files = [
        'visual_model.pth',
        'dorsattn_model.pth', 
        'sommot_model.pth',
        'multi_model.pth',
        'full_visual_model.pth',
        'full_dorsattn_model.pth',
        'full_sommot_model.pth',
        'full_multi_model.pth'
    ]
    
    print("Checking model files...")
    existing = []
    missing = []
    
    for filename in expected_files:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            existing.append(f"{filename} ({size:.1f} MB)")
        else:
            missing.append(filename)
    
    print(f"\nExisting models ({len(existing)}):")
    for model in existing:
        print(f"  ✓ {model}")
    
    print(f"\nMissing models ({len(missing)}):")
    for model in missing:
        print(f"  ✗ {model}")
    
    return existing, missing
