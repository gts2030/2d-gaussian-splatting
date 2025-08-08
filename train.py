#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from alive_progress import alive_bar
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from utils.wandb_utils import (
        init_wandb, log_scalar, log_image, log_histogram, 
        log_metrics, finish_wandb, is_wandb_available, log_training_images,
        create_training_config, prepare_output_and_wandb
    )
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
    print("Warning: wandb utils not available. Install with: uv add wandb")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Initialize W&B logging
    wandb_enabled = prepare_output_and_wandb(dataset, opt, args) if WANDB_FOUND else False
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    first_iter += 1
    total_iterations = opt.iterations - first_iter + 1
    
    with alive_bar(total_iterations, title="🚀 Training 2D Gaussian Splatting", 
                   unit=" iters", enrich_print=False, spinner="waves") as bar:
        for iteration in range(first_iter, opt.iterations + 1):        
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # regularization
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # loss
            total_loss = loss + dist_loss + normal_loss
            
            total_loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


                if iteration % 10 == 0:
                    # Update progress bar with beautiful loss information
                    loss_text = f"Loss: {ema_loss_for_log:.5f} | Distort: {ema_dist_for_log:.5f} | Normal: {ema_normal_for_log:.5f} | Points: {len(gaussians.get_xyz)}"
                    bar.text = loss_text
                    bar(10)  # Advance by 10 iterations
                    
                    # Log metrics to Wandb
                    if wandb_enabled:
                        log_metrics({
                            "loss/total": ema_loss_for_log,
                            "loss/distortion": ema_dist_for_log,
                            "loss/normal": ema_normal_for_log,
                            "metrics/num_points": len(gaussians.get_xyz),
                            "metrics/iteration": iteration,
                            "training/lr_position": gaussians.optimizer.param_groups[0]['lr'],
                            "training/lr_features": gaussians.optimizer.param_groups[1]['lr'],
                            "training/lr_opacity": gaussians.optimizer.param_groups[2]['lr'],
                            "training/lr_scaling": gaussians.optimizer.param_groups[3]['lr'],
                            "training/lr_rotation": gaussians.optimizer.param_groups[4]['lr']
                        }, step=iteration)

                # Log and save
                if tb_writer is not None:
                    tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                
                # Log images to Wandb every 1000 iterations
                if wandb_enabled and iteration % 1000 == 0 and iteration > 0:
                    log_training_images(image, gt_image, render_pkg, iteration, "training")

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)


                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            with torch.no_grad():        
                if network_gui.conn == None:
                    network_gui.try_connect(dataset.render_items)
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                            net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        metrics_dict = {
                            "#": gaussians.get_opacity.shape[0],
                            "loss": ema_loss_for_log
                            # Add more metrics as needed
                        }
                        # Send the data
                        network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                        if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        # raise e
                        network_gui.conn = None

    # Finish Wandb logging
    if wandb_enabled:
        finish_wandb()


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up simplified command line argument parser
    parser = ArgumentParser(description="2D Gaussian Splatting Training")
    
    # Essential arguments only
    parser.add_argument("-s", "--source_path", type=str, required=True, 
                       help="Path to the source directory containing images")
    parser.add_argument("-m", "--model_path", type=str, default="", 
                       help="Path to save model outputs (auto-generated if empty)")
    parser.add_argument("-c", "--config", type=str, default="config.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress verbose output")
    parser.add_argument("--detect_anomaly", action="store_true", 
                       help="Enable PyTorch anomaly detection")
    
    # Parse command line arguments
    cmd_args = parser.parse_args(sys.argv[1:])
    
    # Load and merge configuration
    try:
        from utils.config_utils import (
            load_yaml_config, merge_config_with_args, 
            validate_config, print_config, get_default_config_path
        )
        
        # Use default config path if not specified
        config_path = cmd_args.config
        if not os.path.exists(config_path):
            config_path = get_default_config_path()
            print(f"Using default config: {config_path}")
        
        # Load YAML configuration
        yaml_config = load_yaml_config(config_path)
        
        # Merge with command line arguments
        config = merge_config_with_args(yaml_config, cmd_args)
        
        # Validate configuration
        validate_config(config)
        
        # Print configuration
        if not config.quiet:
            print_config(config, "2D Gaussian Splatting Configuration")
        
        # Create parameter objects from merged config
        lp = ModelParams(ArgumentParser())
        op = OptimizationParams(ArgumentParser()) 
        pp = PipelineParams(ArgumentParser())
        
        # Extract parameters using the merged config
        dataset_params = lp.extract(config)
        opt_params = op.extract(config)
        pipe_params = pp.extract(config)
        
        # Ensure save_iterations includes final iteration
        if hasattr(config, 'save_iterations'):
            if config.iterations not in config.save_iterations:
                config.save_iterations.append(config.iterations)
        else:
            config.save_iterations = [config.iterations]
        
        print(f"🚀 Starting training: {config.source_path}")
        print(f"📁 Output directory: {dataset_params.model_path}")

        # Initialize system state (RNG)
        safe_state(config.quiet)

        # Start GUI server, configure and run training
        gui_ip = getattr(config, 'ip', '127.0.0.1')
        gui_port = getattr(config, 'port', 6009)
        network_gui.init(gui_ip, gui_port)
        torch.autograd.set_detect_anomaly(config.detect_anomaly)
        
        training(
            dataset_params, opt_params, pipe_params, 
            config.test_iterations, config.save_iterations, 
            getattr(config, 'checkpoint_iterations', []), 
            getattr(config, 'start_checkpoint', None), 
            config
        )
        
    except Exception as e:
        print(f"❌ Error during training setup: {e}")
        print(f"💡 Make sure your config file exists and source path is valid")
        sys.exit(1)

    # All done
    print("\nTraining complete.")
