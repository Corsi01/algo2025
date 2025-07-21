import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm, trange


class MultiSubjectMLP(nn.Module):
    def __init__(self, input_dim=3750, embedding_dim=64, hidden_dim=800, output_dim=1000, dropout_rate=0.5, num_subjects=4):
        super().__init__()
        
        # Subject embedding - trainable from one hot encoding
        self.subject_embedding = nn.Linear(num_subjects, embedding_dim, bias=False)
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # SUbject-specific regression heac
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(num_subjects)
        ])
        # Map from subj. index to one hot (for prediction)
        self.num_subjects = num_subjects
    
    def forward(self, x, subject_onehot):
        
        subj_emb = self.subject_embedding(subject_onehot) 
        x_augmented = torch.cat([x, subj_emb], dim=1) 
        backbone_features = self.backbone(x_augmented)  # [batch_size, 800]
        
        return backbone_features
    
    def predict_subject(self, x, subject_id):
        
        batch_size = x.size(0)
        device = x.device
        
        subject_onehot = torch.zeros(batch_size, self.num_subjects, device=device)
        subject_onehot[:, subject_id] = 1.0
        
        backbone_features = self.forward(x, subject_onehot)
        output = self.heads[subject_id](backbone_features)
        
        return output
        

class MultiSubjectDataset(Dataset):
    def __init__(self, features_list, fmri_list, subject_ids):
        """
        features_list: list of numpy arrays for each subject
        fmri_list: list of numpy arrays for each subject
        subject_ids: list of subject indixes [0,1,2,3]
        """
        # Combine all data
        self.features = np.vstack(features_list)
        self.fmri = np.vstack(fmri_list)
        
        # Create subject one-hot
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
