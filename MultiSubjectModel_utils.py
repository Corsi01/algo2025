import os
import random
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


from data_utils import compute_encoding_accuracy, analyze_encoding_accuracy


################# MULTI SUBJECT MODEL ################

class MultiSubjectMLP(nn.Module):
    def __init__(self, input_dim=3750, embedding_dim=64, hidden_dim=800, output_dim=1000, dropout_rate=0.5, num_subjects=4):
        super().__init__()
        
        # Subject embedding trainabile (da one-hot a embedding denso)
        self.subject_embedding = nn.Linear(num_subjects, embedding_dim, bias=False)
        
        # Backbone condiviso
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Heads separati per ogni soggetto
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(num_subjects)
        ])
        
        # Mapping da subject index a one-hot per prediction
        self.num_subjects = num_subjects
    
    def forward(self, x, subject_onehot):
        # Da one-hot a embedding denso (trainabile)
        subj_emb = self.subject_embedding(subject_onehot)  # [batch_size, 4] → [batch_size, 64]
        
        # Concatena features + embedding
        x_augmented = torch.cat([x, subj_emb], dim=1)  # [batch_size, 3814]
        
        # Backbone condiviso
        backbone_features = self.backbone(x_augmented)  # [batch_size, 800]
        
        return backbone_features
    
    def predict_subject(self, x, subject_id):
        """Predizione per un soggetto specifico"""
        batch_size = x.size(0)
        device = x.device
        
        # Crea one-hot per il soggetto specifico
        subject_onehot = torch.zeros(batch_size, self.num_subjects, device=device)
        subject_onehot[:, subject_id] = 1.0
        
        # Backbone features
        backbone_features = self.forward(x, subject_onehot)
        
        # Apply head specifico
        output = self.heads[subject_id](backbone_features)
        
        return output
    
##################### VERSIONE 2


class MultiSubjectMLP2(nn.Module):
    def __init__(self, input_dim=3750, embedding_dim=64, hidden_dim=800, output_dim=1000, dropout_rate=0.5, num_subjects=4):
        super().__init__()

        # Embedding soggetto: da one-hot a embedding denso
        self.subject_embedding = nn.Linear(num_subjects, embedding_dim, bias=False)

        # Backbone condiviso: lavora SOLO su input features
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Heads: prendono [hidden + subject_embedding] → output_dim
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim + embedding_dim, output_dim) for _ in range(num_subjects)
        ])

        self.num_subjects = num_subjects

    def forward(self, x, subject_onehot):
        # Subject embedding: [batch, num_subjects] → [batch, embedding_dim]
        subj_emb = self.subject_embedding(subject_onehot)

        # Backbone: solo su input features
        backbone_features = self.backbone(x)

        # Concatenazione dopo backbone
        combined_features = torch.cat([backbone_features, subj_emb], dim=1)

        return combined_features  

    def predict_subject(self, x, subject_id):
        """Predizione per un soggetto specifico"""
        batch_size = x.size(0)
        device = x.device

        # One-hot per il soggetto
        subject_onehot = torch.zeros(batch_size, self.num_subjects, device=device)
        subject_onehot[:, subject_id] = 1.0

        # Embedding e backbone separati
        subj_emb = self.subject_embedding(subject_onehot)
        backbone_features = self.backbone(x)

        # Concatenazione post-backbone
        combined_features = torch.cat([backbone_features, subj_emb], dim=1)

        # Predizione con head specifica
        output = self.heads[subject_id](combined_features)

        return output

    
    
################# MULTI SUBJECT DATASET ################ 


class MultiSubjectDataset(Dataset):
    def __init__(self, features_list, fmri_list, subject_ids):
        """
        features_list: lista di arrays numpy per ogni soggetto
        fmri_list: lista di arrays numpy per ogni soggetto  
        subject_ids: lista di subject indices [0,1,2,3]
        """
        # Combina tutti i dati
        self.features = np.vstack(features_list)
        self.fmri = np.vstack(fmri_list)
        
        # Crea subject one-hot
        self.subject_onehot = []
        for i, subj_id in enumerate(subject_ids):
            n_samples = len(features_list[i])
            onehot = np.zeros((n_samples, 4))
            onehot[:, subj_id] = 1.0
            self.subject_onehot.append(onehot)
        
        self.subject_onehot = np.vstack(self.subject_onehot)
        
        print(f"Combined dataset: {self.features.shape[0]} samples")
        print(f"Subject distribution: {np.sum(self.subject_onehot, axis=0)}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.fmri[idx], dtype=torch.float32),
            torch.tensor(self.subject_onehot[idx], dtype=torch.float32)
        )
        
        
############### FUNZIONE TRAIN MODELLO MULTI SUBJECT ################


def train_multitask_model(features_list, fmri_list, val_features_list, val_fmri_list, input_dim = 7000, decay=1e-5, embedding_dim=64, hidden_dim=800):
    
    # Parametri training
    lr = 1e-4
    batch_size = 1024
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepara dati training
   # features_list = [features_train_1, features_train_2, features_train_3, features_train_5]
    #fmri_list = [fmri_train_1, fmri_train_2, fmri_train_3, fmri_train_5]
    subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
    
    train_dataset = MultiSubjectDataset(features_list, fmri_list, subject_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Prepara dati validation  
   # val_features_list = [features_val_1, features_val_2, features_val_3, features_val_5]
    #val_fmri_list = [fmri_val_1, fmri_val_2, fmri_val_3, fmri_val_5]
    
    val_dataset = MultiSubjectDataset(val_features_list, val_fmri_list, subject_ids)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modello
    model = MultiSubjectMLP(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=1000,
        dropout_rate=0.5,
        num_subjects=4
    ).to(device)
    
    # Optimizer e loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\nStarting multi-subject training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in trange(num_epochs, desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        
        for features_batch, fmri_batch, subject_onehot_batch in train_loader:
            features_batch = features_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            subject_onehot_batch = subject_onehot_batch.to(device)
            
            # Forward pass
            backbone_features = model(features_batch, subject_onehot_batch)
            
            # Calcola loss per ogni soggetto nel batch
            total_loss = 0.0
            batch_size_total = 0
            
            for subj_id in range(4):
                # Trova campioni di questo soggetto nel batch
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    # Predizioni per questo soggetto
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    # Loss per questo soggetto
                    subj_loss = criterion(subj_pred, subj_fmri)
                    total_loss += subj_loss * subj_features.size(0)
                    batch_size_total += subj_features.size(0)
            
            # Media pesata delle loss
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
        
        tqdm.write(f"Epoch {epoch+1:02d} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
    
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


###################### EVALUATION MODELLO USANDO METRICA CHALLENGE ###############

def evaluate_multitask_model(model, features_val_dict, fmri_val_dict,  subject_mapping={1: 0, 2: 1, 3: 2, 5: 3}):
    """Valuta il modello multi-task su ogni soggetto individualmente"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for original_subj, mapped_subj in subject_mapping.items():
        print(f"\n=== Evaluating Subject {original_subj} (mapped to {mapped_subj}) ===")
        
        # Get validation data for this subject
        features_val = features_val_dict[original_subj]
        fmri_val = fmri_val_dict[original_subj]
        
        # Convert to tensor
        X_test = torch.tensor(features_val, dtype=torch.float32).to(device)
        
        # Predict using subject-specific head
        with torch.no_grad():
            y_pred = model.predict_subject(X_test, mapped_subj)
            y_pred = y_pred.detach().cpu().numpy()
        
        # Compute accuracy
        compute_encoding_accuracy(fmri_val, y_pred, original_subj, 'all')
        analyze_encoding_accuracy(fmri_val, y_pred)
        
    return results
    


def train_multitask_model_all_data(features_train_dict, fmri_train_dict, 
                                   features_val_dict, fmri_val_dict, input_dim = 7000,
                                   decay=4e-4, embedding_dim=256, hidden_dim=1000):
    """
    Train il modello multi-subject su TUTTI i dati disponibili (train + validation)
    Usare solo per il modello finale prima della submission!
    
    Args:
        features_train_dict: Dict con chiavi [1,2,3,5] e valori features_train
        fmri_train_dict: Dict con chiavi [1,2,3,5] e valori fmri_train
        features_val_dict: Dict con chiavi [1,2,3,5] e valori features_val
        fmri_val_dict: Dict con chiavi [1,2,3,5] e valori fmri_val
        decay: Weight decay per l'optimizer
        embedding_dim: Dimensione embedding layer
        hidden_dim: Dimensione hidden layer
    """
    
    # Parametri training
    lr = 1e-4
    batch_size = 1024
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Training on ALL available data (train + validation)...")
    
    # Combina TUTTI i dati disponibili per ogni soggetto
    all_features_list = []
    all_fmri_list = []
    subject_ids = [0, 1, 2, 3]  # Mapping: 1→0, 2→1, 3→2, 5→3
    subject_mapping = {1: 0, 2: 1, 3: 2, 5: 3}
    
    # Per ogni soggetto, combina train + validation
    for original_subj in [1, 2, 3, 5]:
        # Combina features train + val
        features_all = np.vstack([
            features_train_dict[original_subj], 
            features_val_dict[original_subj]
        ])
        
        # Combina fmri train + val
        fmri_all = np.vstack([
            fmri_train_dict[original_subj], 
            fmri_val_dict[original_subj]
        ])
        
        all_features_list.append(features_all)
        all_fmri_list.append(fmri_all)
    
    # Print statistiche dati
    print("\n Combined dataset statistics:")
    for i, original_subj in enumerate([1, 2, 3, 5]):
        mapped_subj = subject_mapping[original_subj]
        train_samples = len(features_train_dict[original_subj])
        val_samples = len(features_val_dict[original_subj])
        total_samples = all_features_list[i].shape[0]
        
        print(f"Subject {original_subj} (mapped to {mapped_subj}): "
              f"{train_samples:,} train + {val_samples:,} val = {total_samples:,} total")
    
    total_samples = sum(len(feat) for feat in all_features_list)
    print(f" Total samples across all subjects: {total_samples:,}")
    
    # Crea dataset combinato
    all_dataset = MultiSubjectDataset(all_features_list, all_fmri_list, subject_ids)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
    
    # Modello
    model = MultiSubjectMLP(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=1000,
        dropout_rate=0.5,
        num_subjects=4
    ).to(device)
    
    # Optimizer e loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()
    
    # Training loop (solo training, no validation)
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
            
            # Calcola loss per ogni soggetto nel batch
            total_loss = 0.0
            batch_size_total = 0
            
            for subj_id in range(4):
                # Trova campioni di questo soggetto nel batch
                subject_mask = subject_onehot_batch[:, subj_id] == 1.0
                
                if subject_mask.any():
                    # Predizioni per questo soggetto
                    subj_features = backbone_features[subject_mask]
                    subj_fmri = fmri_batch[subject_mask]
                    subj_pred = model.heads[subj_id](subj_features)
                    
                    # Loss per questo soggetto
                    subj_loss = criterion(subj_pred, subj_fmri)
                    total_loss += subj_loss * subj_features.size(0)
                    batch_size_total += subj_features.size(0)
            
            # Media pesata delle loss
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