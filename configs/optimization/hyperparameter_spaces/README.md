# Hyperparameter Spaces - Modular Structure

This directory contains hyperparameter space configurations organized in a modular hierarchy to prevent conflicts between different model types.

## Directory Structure

```
configs/optimization/hyperparameter_spaces/
├── README.md                 # This documentation
├── training.yaml            # Generic training hyperparameters (always loaded)
├── loss.yaml                # Generic loss function hyperparameters (always loaded)
└── models/                  # Model-specific hyperparameter spaces
    ├── envnet_v2.yaml       # EnvNet-v2 specific parameters
    ├── ast.yaml             # AST specific parameters
    └── ...                  # Add new models here
```

## How It Works

The optimization scripts automatically detect the model type from the `model._target_` field in the configuration and load only the relevant hyperparameter spaces:

1. **Generic spaces** (always loaded):
   - `training.yaml` - learning rate, weight decay, batch size, epochs, etc.
   - `loss.yaml` - label smoothing and other loss function parameters

2. **Model-specific spaces** (loaded based on model type):
   - Extracts model name from `model._target_` (e.g., `src.models.envnet_v2.EnvNetV2` → `envnet_v2`)
   - Loads `models/{model_name}.yaml` if it exists
   - Examples:
     - `models/envnet_v2.yaml` - dropout, waveform augmentation (time_stretch, gain_shift)
     - `models/ast.yaml` - patch_size, patch_stride, transformer-specific parameters

## Adding New Models

To add support for a new model:

1. **Create model-specific hyperparameter file:**
   ```bash
   touch configs/optimization/hyperparameter_spaces/models/your_model.yaml
   ```

2. **Model detection is automatic** - the system extracts the model name from `model._target_`:
   - `src.models.your_model.YourModelClass` → `your_model`
   - `src.models.resnet.ResNet` → `resnet`
   - No code changes needed!

3. **Define hyperparameters** in your model file following the existing format:
   ```yaml
   model:
     your_param:
       type: float
       low: 0.1
       high: 1.0
   ```

## Example Usage

```bash
# Automatically loads: training.yaml, loss.yaml, models/envnet_v2.yaml
python scripts/optimize_hyperparams.py model=envnet_v2 dataset=esc50

# Automatically loads: training.yaml, loss.yaml, models/ast.yaml  
python scripts/optimize_hyperparams.py model=ast dataset=esc50
```

## Benefits

- ✅ **No parameter conflicts** between different model types
- ✅ **Automatic model detection** - no manual file selection needed
- ✅ **Modular design** - easy to add new models
- ✅ **Generic parameters shared** - common training parameters reused
- ✅ **Clear organization** - model-specific parameters isolated

## Troubleshooting

- **Model not detected**: Check that your model's `_target_` follows the pattern `src.models.{model_name}.{ClassName}`
- **Missing model file**: Create `models/{model_name}.yaml` for your specific model (filename must match extracted model name)
- **Generic files missing**: Ensure `training.yaml` and `loss.yaml` exist in the root directory 