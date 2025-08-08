"""
W&B (Weights & Biases) utilities for 2D Gaussian Splatting training.

This module handles W&B initialization, API key management, and logging utilities.
Adapted from StreetUnveiler project for 2D Gaussian Splatting.
"""

import os
import wandb
import torch
import numpy as np
from pathlib import Path


def load_wandb_api_key():
    """
    Load W&B API key from various sources in order of priority.
    
    Returns:
        str: W&B API key or None if not found
    """
    # Try different locations for API key
    key_locations = [
        "secrets/wandb_api_key.txt",
        "wandb_api_key.txt", 
        os.path.expanduser("~/.wandb_api_key"),
    ]
    
    # First check environment variable
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        return api_key
    
    # Then check files
    for key_file in key_locations:
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r') as f:
                    lines = f.readlines()
                    # Check each line for valid API key
                    for line in lines:
                        content = line.strip()
                        # Skip empty lines, comment lines, and example content
                        if (content and 
                            not content.startswith('#') and 
                            content != 'your_wandb_api_key_here' and
                            len(content) > 10):  # API keys are typically longer than 10 chars
                            return content
            except Exception as e:
                print(f"Warning: Could not read API key from {key_file}: {e}")
                continue
    
    return None


def init_wandb(project_name="2d-gaussian-splatting", experiment_name=None, config=None, model_path=None, entity=None):
    """
    Initialize W&B for training logging.
    
    Args:
        project_name (str): W&B project name
        experiment_name (str): Experiment name (optional)
        config (dict): Training configuration dictionary
        model_path (str): Model output path
        entity (str): W&B entity (username/organization, optional)
        
    Returns:
        bool: True if W&B was successfully initialized, False otherwise
    """
    try:
        # Load API key
        api_key = load_wandb_api_key()
        if not api_key:
            print("⚠️  W&B API key not found. Logging will be disabled.")
            print("   Please create secrets/wandb_api_key.txt with your API key")
            print("   or set WANDB_API_KEY environment variable")
            return False
        
        # Login to W&B
        wandb.login(key=api_key)
        
        # Generate experiment name if not provided
        if not experiment_name:
            experiment_name = f"2dgs_{wandb.util.generate_id()}"
        
        # Initialize run (simplified approach from StreetUnveiler)
        init_params = {
            "project": project_name,
            "name": experiment_name,
            "config": config,
            "dir": model_path if model_path else ".",
            "reinit": True
        }
        
        # Only add entity if specified and not null
        if entity and entity != "null":
            init_params["entity"] = entity
        
        wandb.init(**init_params)
        
        print(f"✅ W&B initialized: {wandb.run.url}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize W&B: {e}")
        print("   Continuing without W&B logging...")
        return False


def log_scalar(name, value, step=None):
    """
    Log scalar value to W&B.
    
    Args:
        name (str): Metric name
        value (float): Metric value
        step (int): Step number (optional)
    """
    if wandb.run is None:
        return
        
    try:
        if step is not None:
            wandb.log({name: value}, step=step)
        else:
            wandb.log({name: value})
    except Exception as e:
        print(f"Warning: Failed to log scalar {name}: {e}")


def log_image(name, image, step=None, caption=None):
    """
    Log image to W&B.
    
    Args:
        name (str): Image name
        image (torch.Tensor or np.ndarray): Image data
        step (int): Step number (optional)
        caption (str): Image caption (optional)
    """
    if wandb.run is None:
        return
        
    try:
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image
        
        # Handle different image formats
        if image_np.ndim == 4:  # Batch dimension
            image_np = image_np[0]  # Take first image
        
        if image_np.ndim == 3:
            # Check if channels first (C, H, W) -> convert to (H, W, C)
            if image_np.shape[0] <= 4:  # Assume channels first if first dim is small
                image_np = np.transpose(image_np, (1, 2, 0))
        elif image_np.ndim == 2:
            # Handle grayscale images (2D) - no conversion needed, wandb handles 2D arrays
            pass
        else:
            print(f"Warning: Unexpected image dimensions: {image_np.shape}")
            return
        
        # Ensure image values are in valid range [0, 1] for wandb
        if image_np.dtype != np.uint8:  # If not already uint8
            image_np = np.clip(image_np, 0, 1)  # Clamp to [0, 1] for float images
        
        # Create W&B image
        wandb_image = wandb.Image(image_np, caption=caption)
        
        if step is not None:
            wandb.log({name: wandb_image}, step=step)
        else:
            wandb.log({name: wandb_image})
            
    except Exception as e:
        print(f"Warning: Failed to log image {name}: {e}")


def log_histogram(name, data, step=None):
    """
    Log histogram to W&B.
    
    Args:
        name (str): Histogram name
        data (torch.Tensor or np.ndarray): Data for histogram
        step (int): Step number (optional)
    """
    if wandb.run is None:
        return
        
    try:
        # Convert tensor to numpy if needed
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy().flatten()
        else:
            data_np = data.flatten()
        
        wandb_hist = wandb.Histogram(data_np)
        
        if step is not None:
            wandb.log({name: wandb_hist}, step=step)
        else:
            wandb.log({name: wandb_hist})
            
    except Exception as e:
        print(f"Warning: Failed to log histogram {name}: {e}")


def log_metrics(metrics_dict, step=None):
    """
    Log multiple metrics at once.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
        step (int): Step number (optional)
    """
    if wandb.run is None:
        return
        
    try:
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)
    except Exception as e:
        print(f"Warning: Failed to log metrics: {e}")


def finish_wandb():
    """Finish W&B run."""
    if wandb.run is not None:
        try:
            wandb.finish()
            print("✅ W&B run finished")
        except Exception as e:
            print(f"Warning: Error finishing W&B run: {e}")


def is_wandb_available():
    """Check if W&B is available and initialized."""
    return wandb.run is not None


def create_training_config(dataset, opt):
    """
    Create training configuration dictionary for W&B logging.
    
    Args:
        dataset: Dataset parameters object
        opt: Optimization parameters object
        
    Returns:
        dict: Configuration dictionary containing all training parameters
    """
    config = {
        # Dataset parameters
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "sh_degree": dataset.sh_degree,
        "white_background": dataset.white_background,
        
        # Optimization parameters
        "iterations": opt.iterations,
        "position_lr_init": opt.position_lr_init,
        "position_lr_final": opt.position_lr_final,
        "feature_lr": opt.feature_lr,
        "opacity_lr": opt.opacity_lr,
        "scaling_lr": opt.scaling_lr,
        "rotation_lr": opt.rotation_lr,
        "lambda_dssim": opt.lambda_dssim,
        "lambda_normal": opt.lambda_normal,
        "lambda_dist": opt.lambda_dist,
        "densify_grad_threshold": opt.densify_grad_threshold,
        "densification_interval": opt.densification_interval,
        "opacity_reset_interval": opt.opacity_reset_interval,
        "densify_from_iter": opt.densify_from_iter,
        "densify_until_iter": opt.densify_until_iter,
    }
    
    return config


def prepare_output_and_wandb(dataset, opt, args):
    """
    Prepare output directory and initialize W&B logging (StreetUnveiler style).
    
    Args:
        dataset: Dataset parameters object
        opt: Optimization parameters object  
        args: Configuration arguments
        
    Returns:
        bool: True if W&B is enabled and initialized successfully, False otherwise
    """
    # Check if wandb should be enabled
    use_wandb = getattr(args, 'enabled', False)  # From yaml config wandb.enabled
    if not use_wandb:
        use_wandb = getattr(args, 'use_wandb', False)  # Fallback to old arg name
    
    if not use_wandb:
        print("ℹ️  W&B logging disabled in configuration")
        return False
    
    # Check if wandb is available
    try:
        import wandb
    except ImportError:
        print("⚠️ W&B not available, continuing without wandb logging")
        return False
    
    # Create experiment name
    scene_name = dataset.source_path.split('/')[-1] if dataset.source_path else 'unknown'
    experiment_name = f"2dgs_{scene_name}_{opt.iterations}iter"
    
    # Prepare config using utility function
    config = create_training_config(dataset, opt)
    
    # Get wandb settings from config
    wandb_project = getattr(args, 'project', None)
    if not wandb_project:
        wandb_project = getattr(args, 'wandb_project', '2d-gaussian-splatting')
    
    # Handle entity more carefully - null/None means use personal account
    wandb_entity = getattr(args, 'entity', None)
    if wandb_entity == 'null':  # YAML null becomes string 'null'
        wandb_entity = None
    elif not wandb_entity:
        wandb_entity = getattr(args, 'wandb_entity', None)
    
    print(f"🚀 Initializing W&B logging...")
    print(f"📁 Project: {wandb_project}")
    print(f"👤 Entity: {wandb_entity if wandb_entity else 'personal account'}")
    print(f"🔖 Experiment: {experiment_name}")
    
    # Initialize W&B
    wandb_enabled = init_wandb(
        project_name=wandb_project,
        experiment_name=experiment_name,
        config=config,
        model_path=dataset.model_path,
        entity=wandb_entity
    )
    
    return wandb_enabled


def log_training_images(rendered_image, gt_image, render_pkg, iteration, scene_name="training"):
    """
    Log training images every 1000 iterations with enhanced visualizations.
    
    Args:
        rendered_image (torch.Tensor): Rendered image
        gt_image (torch.Tensor): Ground truth image
        render_pkg (dict): Render package containing depth, normal maps, etc.
        iteration (int): Current iteration
        scene_name (str): Scene identifier for logging
    """
    if wandb.run is None or iteration % 1000 != 0:
        return
        
    try:
        from utils.general_utils import colormap
        
        # Log rendered and ground truth images
        log_image(f"{scene_name}/rendered", rendered_image, step=iteration, caption=f"Rendered at iter {iteration}")
        log_image(f"{scene_name}/ground_truth", gt_image, step=iteration, caption=f"Ground Truth at iter {iteration}")
        
        # Log depth maps with colormap
        if 'surf_depth' in render_pkg:
            depth = render_pkg['surf_depth']
            depth_norm = depth / depth.max() if depth.max() > 0 else depth
            depth_colored = colormap(depth_norm.detach().cpu().numpy().squeeze(), cmap='jet')
            log_image(f"{scene_name}/depth", depth_colored, step=iteration, caption=f"Depth map at iter {iteration}")
        
        # Log normal maps
        if 'rend_normal' in render_pkg:
            rend_normal = render_pkg['rend_normal'] * 0.5 + 0.5
            log_image(f"{scene_name}/rendered_normal", rend_normal, step=iteration, caption=f"Rendered Normal at iter {iteration}")
        
        if 'surf_normal' in render_pkg:
            surf_normal = render_pkg['surf_normal'] * 0.5 + 0.5
            log_image(f"{scene_name}/surface_normal", surf_normal, step=iteration, caption=f"Surface Normal at iter {iteration}")
        
        # Log alpha channel
        if 'rend_alpha' in render_pkg:
            alpha = render_pkg['rend_alpha']
            log_image(f"{scene_name}/alpha", alpha, step=iteration, caption=f"Alpha at iter {iteration}")
        
        # Log distribution map
        if 'rend_dist' in render_pkg:
            dist = render_pkg['rend_dist']
            dist_colored = colormap(dist.detach().cpu().numpy().squeeze(), cmap='jet')
            log_image(f"{scene_name}/distribution", dist_colored, step=iteration, caption=f"Distribution at iter {iteration}")
        
        print(f"📸 Logged images to W&B at iteration {iteration}")
        
    except Exception as e:
        print(f"⚠️ Failed to log training images: {e}")