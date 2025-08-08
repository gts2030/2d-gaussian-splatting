"""
Configuration utilities for 2D Gaussian Splatting.

This module handles YAML configuration loading and parameter management.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from argparse import Namespace


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {e}")


def create_namespace_from_dict(config_dict: Dict[str, Any], prefix: str = "") -> Namespace:
    """
    Convert nested dictionary to flat namespace object.
    
    Args:
        config_dict (Dict): Configuration dictionary
        prefix (str): Prefix for nested keys
        
    Returns:
        Namespace: Flattened configuration as namespace
    """
    namespace = Namespace()
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # For nested dictionaries, add them as separate namespace attributes
            nested_key = f"{prefix}_{key}" if prefix else key
            setattr(namespace, nested_key, create_namespace_from_dict(value))
        else:
            # For simple values, add directly
            attr_name = f"{prefix}_{key}" if prefix else key
            setattr(namespace, attr_name, value)
    
    return namespace


def merge_config_with_args(config: Dict[str, Any], args: Namespace) -> Namespace:
    """
    Merge YAML configuration with command line arguments.
    Command line arguments take precedence over config file values.
    
    Args:
        config (Dict): Configuration from YAML file
        args (Namespace): Command line arguments
        
    Returns:
        Namespace: Merged configuration
    """
    # Create a new namespace for merged config
    merged = Namespace()
    
    # First, add all config values
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                setattr(merged, key, value)
        else:
            setattr(merged, section, values)
    
    # Override with command line arguments (if they exist and are not None/default)
    args_dict = vars(args)
    for key, value in args_dict.items():
        # Only override if the argument was actually provided (not default)
        if value is not None:
            # Special handling for some arguments that should always override
            if key in ['source_path', 'model_path', 'config', 'quiet'] or value != getattr(merged, key, None):
                setattr(merged, key, value)
    
    return merged


def get_default_config_path() -> str:
    """
    Get the default configuration file path.
    
    Returns:
        str: Path to default config.yaml
    """
    # Look for config.yaml in the project root
    current_dir = Path(__file__).parent.parent  # Go up from utils/
    config_path = current_dir / "config.yaml"
    return str(config_path)


def validate_config(config: Namespace) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config (Namespace): Configuration to validate
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    # Check required parameters
    if not hasattr(config, 'source_path') or not config.source_path:
        raise ValueError("source_path is required")
    
    if not os.path.exists(config.source_path):
        raise ValueError(f"Source path does not exist: {config.source_path}")
    
    # Validate numeric parameters
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    
    if config.position_lr_init <= 0:
        raise ValueError("position_lr_init must be positive")
    
    # Validate lambda parameters
    if not (0 <= config.lambda_dssim <= 1):
        raise ValueError("lambda_dssim must be between 0 and 1")
    
    return True


def save_config_to_yaml(config: Namespace, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config (Namespace): Configuration to save
        output_path (str): Output YAML file path
    """
    # Convert namespace back to nested dictionary
    config_dict = {
        'dataset': {
            'source_path': config.source_path,
            'model_path': config.model_path,
            'images': getattr(config, 'images', 'images'),
            'resolution': getattr(config, 'resolution', -1),
            'white_background': getattr(config, 'white_background', False),
            'data_device': getattr(config, 'data_device', 'cuda'),
            'eval': getattr(config, 'eval', False),
            'sh_degree': getattr(config, 'sh_degree', 3),
            'render_items': getattr(config, 'render_items', ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']),
        },
        'optimization': {
            'iterations': config.iterations,
            'position_lr_init': config.position_lr_init,
            'position_lr_final': config.position_lr_final,
            'position_lr_delay_mult': getattr(config, 'position_lr_delay_mult', 0.01),
            'position_lr_max_steps': getattr(config, 'position_lr_max_steps', 30000),
            'feature_lr': config.feature_lr,
            'opacity_lr': config.opacity_lr,
            'scaling_lr': config.scaling_lr,
            'rotation_lr': config.rotation_lr,
            'lambda_dssim': config.lambda_dssim,
            'lambda_normal': getattr(config, 'lambda_normal', 0.05),
            'lambda_dist': getattr(config, 'lambda_dist', 0.0),
            'densification_interval': config.densification_interval,
            'opacity_reset_interval': config.opacity_reset_interval,
            'densify_from_iter': config.densify_from_iter,
            'densify_until_iter': config.densify_until_iter,
            'densify_grad_threshold': config.densify_grad_threshold,
        },
        'pipeline': {
            'convert_SHs_python': getattr(config, 'convert_SHs_python', False),
            'compute_cov3D_python': getattr(config, 'compute_cov3D_python', False),
            'depth_ratio': getattr(config, 'depth_ratio', 0.0),
            'debug': getattr(config, 'debug', False),
        },
        'training': {
            'test_iterations': getattr(config, 'test_iterations', [7000, 30000]),
            'save_iterations': getattr(config, 'save_iterations', [7000, 30000]),
            'checkpoint_iterations': getattr(config, 'checkpoint_iterations', []),
            'quiet': getattr(config, 'quiet', False),
        },
        'wandb': {
            'enabled': getattr(config, 'use_wandb', False),
            'project': getattr(config, 'wandb_project', '2d-gaussian-splatting'),
            'entity': getattr(config, 'wandb_entity', 'gts2030'),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def print_config(config: Namespace, title: str = "Configuration") -> None:
    """
    Pretty print configuration.
    
    Args:
        config (Namespace): Configuration to print
        title (str): Title for the configuration printout
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    # Group related parameters
    sections = {
        'Dataset': ['source_path', 'model_path', 'white_background', 'sh_degree', 'render_items'],
        'Training': ['iterations', 'position_lr_init', 'feature_lr', 'opacity_lr'],
        'Pipeline': ['convert_SHs_python', 'compute_cov3D_python', 'depth_ratio', 'debug'],
        'Regularization': ['lambda_dssim', 'lambda_normal', 'lambda_dist'],
        'Densification': ['densify_grad_threshold', 'densification_interval', 'densify_from_iter', 'densify_until_iter'],
        'W&B': ['use_wandb', 'wandb_project', 'wandb_entity']
    }
    
    for section_name, keys in sections.items():
        print(f"\n{section_name}:")
        for key in keys:
            if hasattr(config, key):
                value = getattr(config, key)
                print(f"  {key}: {value}")
    
    print(f"\n{'='*50}")
