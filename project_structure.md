# Project Structure - Essential Guide

## data/
Placeholder folder - shows where data should be located

## feature_extraction/
- `01_feature_extraction_new.py` - main feature extraction
- `01_feature_extraction_ood.py` - out-of-distribution features  
- `02_pca_new.py` - PCA transformation
- `02_pca_ood.py` - PCA for OOD data
- `feature_extraction_utils_new.py` - utility functions (new data)
- `feature_extraction_utils_new2.py` - utility functions (new data v2)
- `feature_extraction_utils_ood.py` - utility functions (OOD data)

## optimize_models/
Contains everything needed to optimize and train the final model
- Optimization scripts
- Training utilities
- Model training functions

## Submission scripts phase 2 
All scripts generates predictions ready for CodaBench upload; baseline challenge value is *r = 0.0895*
- **`submission_optimized_single_ood.py`** - predict using base model (*optimize_models/*), using a dummy variable to handle chaplin that has no text features (*r = 0.1528*)
- **`submission_optimized_dual_ood.py`** - predict using base model (*optimize_models/*), handle chaplin that has no text features with an apposite model (*optimize_models_no_text/*);  (*r = 0.1524*)
- **`submission_optimized_triple_ood.py`** - predict using base model (*optimize_models/*), handle chaplin that has no text features with an apposite model (*optimize_models_no_text/*) + handle passepartout with a version that use multilingual text features (*optimize_models_language_multi/*; *r = n*)

## Additional utilities
- `MultiSubjectModel_utils.py` - model utilities
- `submission_utils_ood.py` - submission helper functions
