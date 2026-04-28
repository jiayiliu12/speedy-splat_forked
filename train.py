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
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from compute_scene_metrics import scene_metrics
import wandb
import random

def score_func(view, gaussians, pipeline, background, scores):

    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True

    image = render(view, gaussians, pipeline, background,
                   scores=img_scores)['render']

    # Backward computes and stores grad squared values
    # in img_scores's grad
    image.sum().backward()

    scores += img_scores.grad


def prune(scene, gaussians, pipe, background, prune_ratio):

    start_prune = torch.cuda.Event(enable_timing = True)
    end_prune = torch.cuda.Event(enable_timing = True)
    torch.cuda.reset_peak_memory_stats()

    start_prune.record()

    with torch.enable_grad():
        pbar = tqdm(
            total=len(scene.getTrainCameras()),
            desc='Computing Pruning Scores')
        scores = torch.zeros_like(gaussians.get_opacity) # one score for each Gaussian in the model!

        random_list = random.sample(range(0, len(scene.getTrainCameras()) - 1), int(len(scene.getTrainCameras()) * 0.1)) # sample() samples without replacement! (only unique numbers in list)

        for i, view in enumerate(scene.getTrainCameras()): # TODO: Maybe random sample views?? Whats the quality after this change??
            # if i in random_list:
            score_func(view, gaussians, pipe, background,
                scores)
            pbar.update(1)
        pbar.close()

    gaussians.prune_gaussians(prune_ratio, scores)

    end_prune.record()
    
    # Track peak memory usage (in bytes) and convert to MB
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    pruning_time_ms = start_prune.elapsed_time(end_prune)

    return {
        "peak_memory_allocated" : peak_memory_allocated,
        "peak_memory_reserved" : peak_memory_reserved,
        "time_ms" : pruning_time_ms
    }

def training(dataset, opt, pipe, testing_iterations, visualize_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    wandb.init(project="Speedy-Splat", config={**vars(dataset),**vars(opt)})

    # CUDA timing events
    start_whole = torch.cuda.Event(enable_timing=True)
    end_whole = torch.cuda.Event(enable_timing=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_render = torch.cuda.Event(enable_timing=True)
    end_render = torch.cuda.Event(enable_timing=True)
    start_backward = torch.cuda.Event(enable_timing=True)
    end_backward = torch.cuda.Event(enable_timing=True)
    start_dens = torch.cuda.Event(enable_timing=True)
    end_dens = torch.cuda.Event(enable_timing=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_time_ms = 0

    prune_peak_memory_allocated = 0
    prune_peak_memory_reserved = 0

    # --- Benchmark accumulators (mirrors LiteGS) ---
    # _all_iter_ms tracks pure render+loss+backward time, excluding densification
    # and testing — so it measures the same quantity as LiteGS's per-iter timer.
    _WARMUP_ITERS = min(100, max(10, opt.iterations // 300))
    _all_iter_ms: list[float] = []
    _all_densify_ms: list[float] = []
    # ------------------------------------------------

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        torch.cuda.reset_peak_memory_stats()

        start_whole.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        start.record()
        start_render.record()
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        end_render.record()
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        start_backward.record()
        loss.backward()
        end_backward.record()

        end.record()

        # ---------------------------------------------------------------
        # Everything below is torch.no_grad. Order matches LiteGS:
        #   1. progress bar / saving
        #   2. optimizer step
        #   3. compute pure training timings  <- no densification yet
        #   4. wandb log (training metrics)   <- clean measurement
        #   5. training_report / evaluation   <- clean measurement
        #   6. densification + pruning
        #   7. wandb log (densification + total-with-pruning timing)
        # ---------------------------------------------------------------
        with torch.no_grad():

            # 1. Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 2. Optimizer step (before densification, matching LiteGS order)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # 3. Compute pure training timings (render + loss + backward only).
            #    elapsed_time() blocks until both events are recorded, so no
            #    explicit synchronize is needed here.
            time = start.elapsed_time(end)
            time_render = start_render.elapsed_time(end_render)
            time_bwd = start_backward.elapsed_time(end_backward)
            train_time_ms += time
            _all_iter_ms.append(time)

            # 4. Log pure training metrics — densification has NOT run yet,
            #    so these numbers reflect only the actual training kernel.
            wandb.log({
                "train/total_loss": loss.item(),
                "train/L1": Ll1.item(),
                "gaussians/count": scene.gaussians.get_xyz.shape[0],
                "time/render [ms]": time_render,
                "time/backward [ms]": time_bwd,
                "time/total_iteration [ms]": time,
                "time/train_accumulated [ms]": train_time_ms,
            }, iteration)

            # 5. Evaluation / visualisation — before densification (mirrors LiteGS)
            training_report(
                tb_writer, iteration,
                train_time_ms,
                testing_iterations, visualize_iterations,
                scene, render, (pipe, background), is_final=(iteration == opt.iterations))

            # 6. Densification + pruning
            prune_time_ms = 0

            if iteration < opt.densify_until_iter:
                start_dens.record()
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                end_dens.record()

                # --- Soft Pruning ---
                if (iteration >= opt.prune_from_iter) and \
                    (iteration < opt.prune_until_iter) and \
                    (iteration % opt.prune_interval == 0):

                    prune_pkg = prune(
                        scene, gaussians, pipe, background,
                        opt.densify_prune_ratio)
                    prune_time_ms += prune_pkg['time_ms']
                    prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
                    prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # --- Hard Pruning ---
            if (iteration >= opt.densify_until_iter) and \
                (iteration >= opt.prune_from_iter) and \
                (iteration < opt.prune_until_iter) and \
                (iteration % opt.prune_interval == 0):

                prune_pkg = prune(
                    scene, gaussians, pipe, background,
                    opt.after_densify_prune_ratio)
                prune_time_ms += prune_pkg['time_ms']
                prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
                prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']

            # end_whole covers the full iteration including densification
            end_whole.record()
            end_whole.synchronize()

            # 7. Densification timing logged separately — training metrics above
            #    are already committed and stay uncontaminated.
            time_whole = start_whole.elapsed_time(end_whole)
            time_dens = start_dens.elapsed_time(end_dens) + prune_time_ms

            if time_dens > 0:
                _all_densify_ms.append(time_dens)

            wandb.log({
                "time/densification_pruning [ms]": time_dens,
                "time/total_iteration_with_pruning [ms]": time_whole,
            }, iteration)

    # --- Benchmark Summary (mirrors LiteGS) ---
    _bench_iters = (
        np.array(_all_iter_ms[_WARMUP_ITERS:])
        if len(_all_iter_ms) > _WARMUP_ITERS
        else np.array(_all_iter_ms)
    )
    _densify_arr = np.array(_all_densify_ms) if _all_densify_ms else np.zeros(1)
    _total_iter_s    = np.array(_all_iter_ms).sum() / 1000
    _total_densify_s = _densify_arr.sum() / 1000

    _gpu_name = torch.cuda.get_device_name(0)
    _bench_scalars = {
        "benchmark/iter_mean_ms":         float(np.mean(_bench_iters)),
        "benchmark/iter_median_ms":       float(np.median(_bench_iters)),
        "benchmark/iter_std_ms":          float(np.std(_bench_iters)),
        "benchmark/densify_mean_ms":      float(np.mean(_densify_arr)),
        "benchmark/densify_total_s":      round(_total_densify_s, 3),
        "benchmark/total_training_s":     round(_total_iter_s, 3),
        "benchmark/total_with_densify_s": round(_total_iter_s + _total_densify_s, 3),
    }
    wandb.log({
        **_bench_scalars,
        "benchmark/iter_time_histogram":    wandb.Histogram(np.array(_bench_iters)),
        "benchmark/densify_time_histogram": wandb.Histogram(_densify_arr),
    })
    wandb.summary.update({
        "benchmark/gpu":            _gpu_name,
        "benchmark/warmup_iters":   _WARMUP_ITERS,
        "benchmark/measured_iters": len(_bench_iters),
        **_bench_scalars,
    })
    print("\n=== Training Benchmark ===")
    print(f"  GPU:            {_gpu_name}")
    print(f"  Iters measured: {len(_bench_iters):,}  (excl. {_WARMUP_ITERS} warmup)")
    print(f"  Iter time:      mean {np.mean(_bench_iters):.2f} ms  |  median {np.median(_bench_iters):.2f} ms  |  std {np.std(_bench_iters):.2f} ms")
    print(f"  Densification:  mean {np.mean(_densify_arr):.2f} ms  |  total {_total_densify_s:.2f} s  ({len(_all_densify_ms)} steps)")
    print(f"  Pure training:  {_total_iter_s:.1f} s")
    print(f"  Incl. densify:  {_total_iter_s + _total_densify_s:.1f} s")
    print("==========================\n")
    # ------------------------------------------


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
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
        tb_writer, iteration,
        train_time_ms,
        testing_iterations, visualize_iterations,
        scene: Scene, renderFunc, renderArgs, is_final):
    """Evaluate on test/train cameras and log quality metrics.

    Timing parameters for densification/pruning have been removed — those are
    logged separately in the main loop AFTER densification runs, keeping this
    function's output uncontaminated by densification overhead.
    """

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        print("\n[ITER {}] Training Time: {:.1f} minutes".format(
            iteration, train_time_ms / 60_000))

        validation_configs = (
            {'name': 'Testset',     'cameras': scene.getTestCameras()},
            {'name': 'Trainingset', 'cameras': [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            name, cameras = config['name'], config['cameras']
            if cameras and len(cameras) > 0:
                if is_final: # Calculate LPIPS at last iteration
                    lpips_metrics = scene_metrics(iteration, name, cameras,
                    scene, renderFunc, renderArgs, True)
                    wandb.log({
                        f"test/l1_loss_{name}":        lpips_metrics[0],
                        f"test/psnr_{name}":           lpips_metrics[1],
                        f"test/ssim_{name}":           lpips_metrics[2],
                        f"test/lpips_{name}":          lpips_metrics[3],
                        f"time/inference_{name} [ms]": lpips_metrics[4] * 1000,
                    }, step=iteration)
                else:
                    metrics = scene_metrics(iteration, name, cameras,
                        scene, renderFunc, renderArgs, False)
                    wandb.log({
                        f"test/l1_loss_{name}":        metrics[0],
                        f"test/psnr_{name}":           metrics[1],
                        f"test/ssim_{name}":           metrics[2],
                        f"time/inference_{name} [ms]": metrics[3] * 1000,
                    }, step=iteration)
                    if tb_writer:
                        tb_writer.add_scalar(f'metrics_{name}/L1 Loss', metrics[0], iteration)
                        tb_writer.add_scalar(f'metrics_{name}/PSNR',    metrics[1], iteration)
                        tb_writer.add_scalar(f'metrics_{name}/SSIM',    metrics[2], iteration)
                        tb_writer.add_scalar(f'metrics_{name}/FPS',     metrics[3], iteration)

                # --- Wandb image logging for Testset (mirrors LiteGS) ---
                if name == "Testset":
                    logged_images = []
                    num_log_images = 6
                    log_indices = set(np.linspace(0, len(cameras) - 1, num_log_images, dtype=int).tolist())
                    for batch_i, viewpoint in enumerate(cameras):
                        if batch_i in log_indices:
                            rendered = torch.clamp(
                                renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
                            gt_np       = gt.permute(1, 2, 0).cpu().numpy()
                            panel       = np.concatenate([rendered_np, gt_np], axis=1)
                            panel_uint8 = (panel * 255).astype(np.uint8)
                            logged_images.append(wandb.Image(
                                panel_uint8,
                                caption=f"iter {iteration} | Testset | frame {batch_i} | left: render  right: GT"
                            ))
                    wandb.log({"test/renders_Testset": logged_images}, step=iteration)
                # ---------------------------------------------------------

    # if (iteration in visualize_iterations) and tb_writer:
    #     validation_configs = (
    #         {'name': 'Testset',     'cameras': scene.getTestCameras()},
    #         {'name': 'Trainingset', 'cameras': [
    #             scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
    #             for idx in range(5, 30, 5)]}
    #     )
    #     for config in validation_configs:
    #         for viewpoint in config['cameras'][:5]:
    #             image = torch.clamp(
    #                 renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
    #                 0.0, 1.0)
    #             gt_image = torch.clamp(
    #                 viewpoint.original_image.to("cuda"),
    #                 0.0, 1.0)
    #             tb_writer.add_images(
    #                 config['name'] + "_view_{}/render".format(viewpoint.image_name),
    #                 image[None], global_step=iteration)
    #             tb_writer.add_images(
    #                 config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
    #                 gt_image[None], global_step=iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(1000, 31000, 1000)))
    parser.add_argument("--visualize_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.visualize_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")