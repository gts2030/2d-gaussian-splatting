# Configuration Files

This directory contains various YAML configuration files for different training scenarios.

## Available Configurations

### `quick_test.yaml`
- **Purpose**: Fast testing and development
- **Iterations**: 5,000
- **Features**: Reduced training time, basic quality
- **Use case**: Testing new datasets, debugging, rapid prototyping

### `high_quality.yaml`
- **Purpose**: Best quality results
- **Iterations**: 50,000
- **Features**: Extended training, enhanced regularization
- **Use case**: Final production runs, research papers, demos

## Usage

```bash
# Use quick test config
python train.py -s /path/to/your/dataset -c configs/quick_test.yaml

# Use high quality config
python train.py -s /path/to/your/dataset -c configs/high_quality.yaml

# Use default config (config.yaml in project root)
python train.py -s /path/to/your/dataset
```

## Creating Custom Configurations

You can create your own configuration files by copying one of the existing ones and modifying the parameters. The YAML structure is:

```yaml
dataset:
  # Dataset-related parameters
  
optimization:
  # Training optimization parameters
  
pipeline:
  # Rendering pipeline parameters
  
training:
  # Training control parameters
  
gui:
  # Network GUI settings
  
wandb:
  # Weights & Biases logging settings
```

## Parameter Priority

Command line arguments take precedence over YAML config values:

1. **Command line arguments** (highest priority)
2. **YAML configuration file**
3. **Default values** (lowest priority)

Essential arguments like `source_path` must still be provided via command line.
