# 🎯 Training Guide: Sound Classification with PyTorch Lightning

This guide covers both **normal training** (single experiments) and **hyperparameter optimization** using Optuna with TPE sampling and Hyperband pruning.

## 📋 Prerequisites

### 1. Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Verify installations
python -c "import torch, lightning, optuna, mlflow; print('✅ All packages available')"
```

### 2. Data Preparation
```bash
# Download datasets
python scripts/download_data.py

# Prepare ESC-50 dataset
python scripts/prepare_esc50.py
```

### 3. MLflow Setup
```bash
# Start MLflow UI (in separate terminal)
mlflow ui

# Or use the custom UI launcher
python scripts/mlflow_ui.py  # Includes ngrok for remote access
```

---

## 🏃 Normal Training (Single Experiments)

### Basic Commands

#### **Default Training**
```bash
# Train EnvNet-v2 on ESC-50 fold 0 with default settings
python scripts/train.py
```

#### **Custom Parameters**
```bash
# Train with specific parameters
python scripts/train.py \
  dataset=esc50 \
  dataset.fold=1 \
  model=envnet_v2 \
  optimizer.lr=0.001 \
  batch_size=64 \
  trainer.max_epochs=50

# Quick test run
python scripts/train.py \
  trainer.max_epochs=5 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=10
```

#### **Different Datasets**
```bash
# Train on UrbanSound8K
python scripts/train.py dataset=urbansound8k

# Train on specific ESC-50 fold
python scripts/train.py dataset.fold=2
```

### Creating Custom Experiment Configs

#### **Step 1: Create Experiment Config**
```bash
mkdir -p configs/experiments
cp configs/training.yaml configs/experiments/my_experiment.yaml
```

#### **Step 2: Edit Experiment Config**
```yaml
# configs/experiments/my_experiment.yaml
# @package _global_
defaults:
  - base_config
  - override dataset: esc50
  - override model: envnet_v2
  - _self_

# Experiment identification
experiment_name: "envnet_dropout_study"
seed: 42

# Model settings
model:
  dropout: 0.3

# Training settings
optimizer:
  lr: 0.001
  weight_decay: 1e-4

scheduler:
  T_max: 100

batch_size: 32
trainer:
  max_epochs: 100

# Logging
logging:
  experiment_name: "${experiment_name}_fold_${dataset.fold}"
```

#### **Step 3: Run Custom Experiment**
```bash
python scripts/train.py \
  --config-path=configs/experiments \
  --config-name=my_experiment
```

---

## 🔧 Hyperparameter Optimization

### Configuration Setup

#### **Step 1: Configure Hyperparameter Spaces**

**Model Parameters** (`configs/optimization/hyperparameter_spaces/envnet_v2.yaml`):
```yaml
model:
  dropout:
    type: float
    low: 0.1
    high: 0.7
    log: false
```

**Training Parameters** (`configs/optimization/hyperparameter_spaces/training.yaml`):
```yaml
# Optimizer hyperparameters
optimizer:
  lr:
    type: float
    low: 1e-5
    high: 1e-1
    log: true  # Log scale for learning rate
  
  weight_decay:
    type: float
    low: 1e-6
    high: 1e-2
    log: true

# Training configuration
trainer:
  max_epochs:
    type: int
    low: 50
    high: 200

# Batch size
batch_size:
  type: categorical
  choices: [16, 32, 64, 128]

# Data augmentation
dataset:
  augment:
    type: categorical
    choices: [true, false]
```

**Loss Parameters** (`configs/optimization/hyperparameter_spaces/loss.yaml`):
```yaml
loss:
  label_smoothing:
    type: float
    low: 0.0
    high: 0.3
    log: false
```

#### **Step 2: Configure Optimization Settings**

**Edit Optuna Config** (`configs/optimization/optuna_config.yaml`):
```yaml
# Study configuration
study_name: "envnet_esc50_optimization_v1"
direction: "maximize"  # maximize validation accuracy
storage_path: "optuna_studies.db"

# Optimization settings
n_trials: 100
timeout: null  # or 3600 for 1-hour limit

# TPE Sampler
sampler:
  n_startup_trials: 10
  n_ei_candidates: 24
  seed: 42

# Hyperband Pruner
pruner:
  min_resource: 1
  max_resource: 100
  reduction_factor: 3

# Monitoring
monitor: "val/acc"
mode: "max"
min_epochs: 10

# MLflow
mlflow_experiment_name: "optuna_hyperparameter_optimization"

# Output
output_dir: "outputs/optimization"
best_config_path: "best_config.yaml"
```

### Running Optimization

#### **Basic Optimization**

**Note**: The optimization script uses `configs/optimization.yaml` which combines both training and optuna settings. This allows you to override both training parameters and optuna settings from the command line.

```bash
# Run with default settings
python scripts/optimize_hyperparams.py
python scripts/debug_optimize.py # shows progress bars and detailed logging

# Quick test (5 trials)
python scripts/optimize_hyperparams.py optuna.n_trials=5

# Custom study name
python scripts/optimize_hyperparams.py \
  optuna.study_name="quick_test_v1"
```

#### **Production Optimization**
```bash
# Full optimization run
python scripts/optimize_hyperparams.py \
  optuna.n_trials=200 \
  optuna.study_name="envnet_esc50_production" \
  optuna.timeout=7200  # 2 hours max
```

#### **Resume Existing Study**
```bash
# Add more trials to existing study
python scripts/optimize_hyperparams.py \
  optuna.study_name="envnet_esc50_production" \
  optuna.n_trials=50  # Add 50 more trials
```

### Monitoring Progress

#### **MLflow Dashboard**
1. Open browser to `http://localhost:5000`
2. Look for experiment: `"optuna_hyperparameter_optimization"`
3. Monitor trials in real-time

#### **Command Line Monitoring**
```bash
# List all studies
python scripts/analyze_study.py --list

# Quick study info
python scripts/analyze_study.py \
  --study-name "envnet_esc50_optimization_v1"
```

---

## 📊 Analysis and Results

### Study Analysis

#### **Basic Analysis**
```bash
# Analyze study results
python scripts/analyze_study.py \
  --study-name "envnet_esc50_optimization_v1"

# Generate visualization plots
python scripts/analyze_study.py \
  --study-name "envnet_esc50_optimization_v1" \
  --plots

# Export results to CSV
python scripts/analyze_study.py \
  --study-name "envnet_esc50_optimization_v1" \
  --export "results_v1.csv"
```

#### **Understanding Output**

The analysis provides:
- **Study Summary**: Total trials, completion rates
- **Best Trial**: Optimal hyperparameters and performance
- **Parameter Importance**: Which hyperparameters matter most
- **Pruning Statistics**: Efficiency of early stopping
- **Optimization History**: Performance improvement over time

### Using Best Configuration

#### **Train Final Model**
```bash
# Best config is saved automatically
cat outputs/optimization/best_config.yaml

# Train final model with best hyperparameters
python scripts/train.py \
  --config-path=outputs/optimization \
  --config-name=best_config \
  trainer.max_epochs=200  # Train longer for final model
```

#### **Cross-Validation with Best Config**
```bash
# Test on all folds
for fold in 0 1 2 3 4; do
  python scripts/train.py \
    --config-path=outputs/optimization \
    --config-name=best_config \
    dataset.fold=$fold \
    logging.experiment_name="best_config_fold_${fold}"
done
```

---

## 📈 Best Practices

### Optimization Strategy

#### **1. Start Small**
```bash
# Quick exploration (15-30 minutes)
python scripts/optimize_hyperparams.py \
  optuna.n_trials=20 \
  optuna.study_name="quick_exploration"
```

#### **2. Iterative Refinement**
1. **Broad Search**: Wide ranges, 50-100 trials
2. **Analyze Results**: Check parameter importance
3. **Narrow Search**: Focus on important parameters
4. **Final Optimization**: Many trials in refined space

#### **3. Resource Management**
```yaml
# For overnight runs
n_trials: 100
timeout: 28800  # 8 hours
min_epochs: 5   # Faster pruning
max_resource: 50  # Don't waste time on bad trials
```

### Hyperparameter Space Design

#### **Learning Rate**
```yaml
optimizer:
  lr:
    type: float
    low: 1e-5    # Very conservative
    high: 1e-1   # Aggressive
    log: true    # ALWAYS use log scale for LR
```

#### **Regularization**
```yaml
model:
  dropout:
    type: float
    low: 0.0     # No dropout
    high: 0.8    # Heavy dropout
    
optimizer:
  weight_decay:
    type: float
    low: 1e-6
    high: 1e-2
    log: true    # Log scale for weight decay
```

#### **Architecture**
```yaml
# For discrete choices
batch_size:
  type: categorical
  choices: [16, 32, 64, 128]  # Hardware-friendly sizes

# For continuous ranges
model:
  hidden_size:
    type: int
    low: 64
    high: 512
    log: false  # Linear scale for sizes
```

### Experiment Organization

#### **Naming Convention**
```bash
# Descriptive study names
study_name="envnet_esc50_lr_batch_dropout_v1"
study_name="envnet_esc50_scheduler_comparison_v2"
study_name="envnet_esc50_final_optimization_v3"

# Include dataset and model
study_name="resnet_urbansound8k_optimization"
study_name="transformer_esc50_fold0_v1"
```

#### **Folder Structure**
```
outputs/
├── optimization/
│   ├── envnet_esc50_v1/
│   │   ├── best_config.yaml
│   │   ├── study_summary.json
│   │   └── plots/
│   ├── envnet_esc50_v2/
│   └── resnet_urbansound8k/
├── experiments/
│   ├── single_experiments/
│   └── ablation_studies/
└── final_models/
    ├── envnet_esc50_best/
    └── resnet_urbansound8k_best/
```

---

## 🚨 Troubleshooting

### Common Issues

#### **"No studies found"**
```bash
# Check database location
ls -la optuna_studies.db

# List all studies
python scripts/analyze_study.py --list

# Verify study name
python scripts/optimize_hyperparams.py optuna.study_name="correct_name"
```

#### **"MLflow import error"**
```bash
# Check for naming conflicts
ls scripts/mlflow*

# Should see mlflow_ui.py, NOT mlflow.py
# If mlflow.py exists, rename it:
mv scripts/mlflow.py scripts/mlflow_ui.py
rm -rf scripts/__pycache__
```

#### **"Optimization not improving"**
- Check hyperparameter ranges aren't too narrow
- Verify metric name (`val/acc` vs `val_acc`)
- Increase `n_startup_trials` for more exploration
- Check if data is properly shuffled

#### **"Trials fail immediately"**
```bash
# Test single training run first
python scripts/train.py \
  trainer.max_epochs=2 \
  trainer.limit_train_batches=5

# Check GPU memory
nvidia-smi

# Use smaller batch sizes
python scripts/optimize_hyperparams.py \
  batch_size.choices="[8, 16, 32]"
```

#### **"Too slow"**
```yaml
# Speed up optimization
pruner:
  min_resource: 2     # Prune earlier
  max_resource: 30    # Don't train too long
  
trainer:
  max_epochs: 50      # Reasonable upper bound
```

#### **"Out of memory"**
```bash
# Reduce batch sizes in hyperparameter space
# Edit configs/optimization/hyperparameter_spaces/training.yaml
batch_size:
  type: categorical
  choices: [8, 16, 32]  # Smaller sizes
```

### Performance Tips

#### **Speed Up Training**
```yaml
# In training config
trainer:
  precision: 16-mixed  # Use mixed precision
  devices: 1           # Single GPU
  
num_workers: 8         # Optimize data loading

# For optimization
optuna:
  min_epochs: 5        # Prune early
  max_resource: 50     # Don't overtrain
```

#### **Optimize Resource Usage**
```bash
# Set memory limits
export CUDA_VISIBLE_DEVICES=0

# Monitor resource usage
watch -n 1 nvidia-smi

# Use system monitoring
htop
```

---

## 📚 Examples

### Complete Workflow Example

```bash
# 1. Quick exploration
python scripts/optimize_hyperparams.py \
  optuna.n_trials=20 \
  optuna.study_name="envnet_esc50_explore"

# 2. Analyze results
python scripts/analyze_study.py \
  --study-name "envnet_esc50_explore" \
  --plots

# 3. Refine search based on importance analysis
# Edit hyperparameter spaces to focus on important parameters

# 4. Full optimization
python scripts/optimize_hyperparams.py \
  optuna.n_trials=100 \
  optuna.study_name="envnet_esc50_refined"

# 5. Train final model
python scripts/train.py \
  --config-path=outputs/optimization \
  --config-name=best_config \
  trainer.max_epochs=200

# 6. Cross-validate
for fold in 0 1 2 3 4; do
  python scripts/train.py \
    --config-path=outputs/optimization \
    --config-name=best_config \
    dataset.fold=$fold
done
```

### Research Experiment Pipeline

```bash
# Ablation study: Effect of dropout
python scripts/optimize_hyperparams.py \
  optuna.study_name="ablation_dropout" \
  # Only vary dropout, fix other parameters

# Architecture comparison
python scripts/optimize_hyperparams.py \
  model=envnet_v2 \
  optuna.study_name="envnet_v2_optimization"

python scripts/optimize_hyperparams.py \
  model=resnet \
  optuna.study_name="resnet_optimization"

# Dataset comparison
python scripts/optimize_hyperparams.py \
  dataset=esc50 \
  optuna.study_name="esc50_best"

python scripts/optimize_hyperparams.py \
  dataset=urbansound8k \
  optuna.study_name="urbansound8k_best"
```

---

This guide provides everything you need to run effective experiments and hyperparameter optimization for your sound classification pipeline! 🎵✨ 