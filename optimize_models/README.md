# Hyperparameter Optimization - Essential Commands

## Installation
```bash
pip install optuna>=3.0.0
```

## Optimization

### All networks
```bash
python train_multisubject.py --network all --optimize --n_trials 30
```

### Single network
```bash
python train_multisubject.py --network visual --optimize --n_trials 50
python train_multisubject.py --network dorsattn --optimize --n_trials 30
python train_multisubject.py --network sommot --optimize --n_trials 40
python train_multisubject.py --network multi --optimize --n_trials 60
```

## Training with optimized parameters

### All networks
```bash
python train_multisubject.py --network all --use_best_params
```

### Single network
```bash
python train_multisubject.py --network visual --use_best_params
```

### Final training (train+validation)
```bash
python train_multisubject.py --network all --use_best_params --full_training
```

## Evaluation
```bash
python train_multisubject.py --evaluate_only
```

## Complete workflow
```bash
# 1. Optimize
python train_multisubject.py --network all --optimize --n_trials 50

# 2. Final training
python train_multisubject.py --network all --use_best_params --full_training

# 3. Evaluation
python train_multisubject.py --evaluate_only
```

## Generated files
- `optimization_results.json` - optimized parameters
- `models/*.pth` - saved models
