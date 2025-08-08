# W&B (Weights & Biases) Configuration Guide

This guide helps you set up W&B logging for 2D Gaussian Splatting training.

## Quick Setup

### 1. Enable W&B in Configuration

Edit your config file (e.g., `config.yaml`) and set:

```yaml
wandb:
  enabled: true              # Enable W&B logging
  project: "your-project"    # Your project name
  entity: null               # Use personal account (recommended)
  run_name: null             # Auto-generated
```

### 2. API Key Setup

Choose one of these methods:

**Option A: Environment Variable (Recommended)**
```bash
export WANDB_API_KEY="your_api_key_here"
python train.py -s /path/to/dataset
```

**Option B: Key File**
```bash
echo "your_api_key_here" > secrets/wandb_api_key.txt
python train.py -s /path/to/dataset
```

## Entity Configuration

### Personal Account (Recommended)
```yaml
wandb:
  entity: null  # Uses your personal W&B account automatically
```

### Organization Account
```yaml
wandb:
  entity: "your-organization-name"  # Must have write permissions
```

## Common Issues & Solutions

### 403 Permission Error

**Problem**: `permission denied` or `PERMISSION_ERROR`

**Solutions**:
1. **Use personal account**: Set `entity: null` in config
2. **Check organization permissions**: Ensure you can create projects in the organization
3. **Verify entity name**: Make sure the organization name is correct

### Entity Access Issues

**Problem**: Cannot access specified entity

**Solutions**:
1. The system will automatically fall back to your personal account
2. Check the console output for entity validation messages
3. Verify you're logged into the correct W&B account

## Configuration Examples

### Personal Development
```yaml
wandb:
  enabled: true
  project: "2dgs-experiments"
  entity: null  # Personal account
```

### Team/Organization
```yaml
wandb:
  enabled: true
  project: "2dgs-production"
  entity: "your-team-name"  # Organization with proper permissions
```

### Disable Logging
```yaml
wandb:
  enabled: false  # No W&B logging
```

## Command Line Usage

```bash
# Train with W&B (if enabled in config)
python train.py -s /path/to/dataset

# Train without W&B (regardless of config)
python train.py -s /path/to/dataset -c config_no_wandb.yaml
```

## Logged Metrics

When enabled, the system logs:

- **Loss metrics**: Total loss, distortion loss, normal loss
- **Training metrics**: Number of points, learning rates
- **Images**: Rendered images, ground truth, depth maps, normal maps
- **Configuration**: All training parameters

## Troubleshooting

### Check W&B Status
1. Look for initialization messages in console output
2. Check for entity validation and fallback messages
3. Verify the run URL is displayed if successful

### Debug Mode
Enable more detailed error messages by checking the console output during W&B initialization.

### Support
- Ensure you have the latest version of `wandb` installed
- Check W&B documentation for account/organization setup
- Verify your API key is valid and has necessary permissions
