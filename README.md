# Project Structure - Essential Guide

## data/
Placeholder folder - shows where data should be located

## feature_extraction/
- `01_feature_extraction_new.py` - main feature extraction
- `01_feature_extraction_ood.py` - out-of-distribution features  
- `02_pca_new.py` - PCA transformation
- `02_pca_ood.py` - PCA for OOD data
- `feature_extraction_utils_new.py` - utility functions (ID data)
- `feature_extraction_utils_ood.py` - utility functions (OOD data)

## optimize_models/
Contains everything needed to optimize and train the final models; consider that each model was otpimized for 30 rounds, further optimization could lead to even better results

## utils/
- `data_utils.py` - utilities function for loading data and analyzing yeo's network
- `multisubject_utils.py` - model utilities
- `optimization_utils.py` - utilities for optimization and hyperparameters tuning
- `submission_utils_ood.py` - submission helper functions
- `train_utils.py` - model training helper functions

## Submission scripts  
All scripts generates predictions ready for CodaBench upload
- **`submission_id.py`** - generate prediction for model selection phase on ID data (*r = 0.2659 vs baseline r = 0.2033*)
- **`submission_ood.py`** - generate prediction for model evaluation phase on OOD data (*r = 0.1576 vs baseline r = 0.0895*)


