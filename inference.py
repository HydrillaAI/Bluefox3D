import os
import argparse
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
_cache_dir = os.path.dirname(os.path.abspath(__file__))
_gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(_cache_dir, f'autotune_cache_{_gpu_tag}.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

import json

from bluefox3d.config import DEFAULT_MODEL_PATH, PRODUCT_NAME
from bluefox3d.pipelines import BlueFox3DImageTo3DPipeline
from bluefox3d.utils.mesh_cleanup import cleanup_mesh
import o_voxel

PRESETS = {
    "baseline": {},
    "quality": {
        "resolution": 1536,
        "cleanup": True,
        "smooth": False,
        "keep_alpha": True,
        "ss_sampling_steps": 12,
        "shape_slat_sampling_steps": 12,
        "tex_slat_sampling_steps": 12,
    },
    "fast": {
        "resolution": 1024,
        "cleanup": True,
        "smooth": False,
        "keep_alpha": True,
        "ss_sampling_steps": 8,
        "shape_slat_sampling_steps": 8,
        "tex_slat_sampling_steps": 8,
    },
}

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = DEFAULT_MODEL_PATH

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from bluefox3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False):
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = BlueFox3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def save_mesh_stills(mesh, stills_dir: str, resolution: int = 512):
    from bluefox3d.representations.mesh.base import Mesh
    from bluefox3d.utils.render_utils import render_frames, yaw_pitch_r_fov_to_extrinsics_intrinsics

    os.makedirs(stills_dir, exist_ok=True)
    preview = Mesh(mesh.vertices, mesh.faces)
    yaws = [np.pi / 2, np.pi / 2 + np.pi / 4]
    pitches = [0.2, 0.25]
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, 2.0, 40)
    frames = render_frames(preview, extr, intr, {"resolution": resolution, "bg_color": (255, 255, 255)}, verbose=False)
    color = frames.get("color") or frames.get("normal") or next(iter(frames.values()))
    names = ("front.png", "three_quarter.png")
    saved = []
    for name, frame in zip(names, color):
        path = os.path.join(stills_dir, name)
        Image.fromarray(frame).save(path)
        saved.append(path)
    return saved


def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    resolution: int = -1,
    decimation_target: int = 1000000,
    texture_size: int = 4096,
    cleanup: bool = False,
    smooth: bool = False,
    keep_alpha: bool = False,
    save_preprocessed: str = "",
    render_stills: str = "",
    meta_path: str = "",
    remesh: bool = True,
):
    started = time.time()
    # Load models
    pipeline = init_pipeline(model_path, low_vram=low_vram)

    # Preprocess image first — rembg loads to GPU for this call, then offloads.
    # MoGe is loaded afterwards so both never occupy VRAM at the same time.
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img, keep_alpha=keep_alpha)
    if save_preprocessed:
        os.makedirs(os.path.dirname(os.path.abspath(save_preprocessed)) or ".", exist_ok=True)
        image_preprocessed.save(save_preprocessed)
        print(f"[Inference] Saved preprocessed image: {save_preprocessed}")
    image_for_model = image_preprocessed.convert("RGB")

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)) or ".", f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_for_model.save(tmp_path)

    # Camera estimation
    if manual_fov > 0:
        # Use manually specified FOV (in radians)
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cuda")
        print("[Inference] Estimating camera parameters...")
        camera_params = get_camera_params_wild_moge(
            tmp_path, moge_model, device="cuda",
            mesh_scale=mesh_scale, extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        # MoGe is only needed for camera estimation; free its VRAM for inference.
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()
    os.remove(tmp_path)

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_for_model,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]
    cleanup_stats = None
    if cleanup:
        print("[Inference] Mesh cleanup (drop islands + light smooth)...")
        cleanup_stats = cleanup_mesh(mesh, drop_islands=True, smooth=smooth)

    stills = []

    # Extract GLB (do this before any extra GPU renders; those can poison the CUDA context)
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target, texture_size=texture_size,
        remesh=remesh, remesh_band=1, remesh_project=0, use_tqdm=True,
    )

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    if render_stills:
        try:
            print(f"[Inference] Rendering stills to {render_stills} ...")
            stills = save_mesh_stills(mesh, render_stills)
        except Exception as exc:
            print(f"[Inference] Still render skipped: {exc}")
    elapsed = time.time() - started
    print(f"[Done] GLB saved to: {output_path} ({elapsed:.1f}s)")
    if meta_path:
        os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "image": image_path,
                "output": output_path,
                "seed": seed,
                "model_path": model_path,
                "low_vram": low_vram,
                "resolution": resolution if resolution > 0 else (1024 if low_vram else 1536),
                "pipeline_type": pipeline_type,
                "cleanup": cleanup,
                "keep_alpha": keep_alpha,
                "decimation_target": decimation_target,
                "texture_size": texture_size,
                "remesh": remesh,
                "ss_sampling_steps": ss_sampling_steps,
                "shape_slat_sampling_steps": shape_slat_sampling_steps,
                "tex_slat_sampling_steps": tex_slat_sampling_steps,
                "cleanup_stats": cleanup_stats,
                "stills": stills,
                "preprocessed": save_preprocessed or None,
                "seconds": elapsed,
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{PRODUCT_NAME} Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM from ~18GB to ~10-12GB at the cost of slower inference.")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")
    parser.add_argument("--preset", type=str, default="baseline", choices=sorted(PRESETS),
                        help="baseline = Pixal3D-equivalent defaults. quality/fast enable wrapper cleanup.")
    parser.add_argument("--decimation", type=int, default=1000000, help="GLB decimation target")
    parser.add_argument("--texture-size", type=int, default=4096, dest="texture_size", help="GLB texture size")
    parser.add_argument("--no-remesh", action="store_true", dest="no_remesh",
                        help="Skip o_voxel remesh (use when 1536 remesh OOMs)")
    parser.add_argument("--cleanup", action="store_true", help="Drop small disconnected mesh islands")
    parser.add_argument("--smooth", action="store_true", help="Optional Laplacian smooth after island drop")
    parser.add_argument("--keep-alpha", action="store_true", dest="keep_alpha",
                        help="Save/return premultiplied RGBA preprocess (model still sees RGB)")
    parser.add_argument("--save-preprocessed", type=str, default="", dest="save_preprocessed",
                        help="Path to write the preprocessed PNG")
    parser.add_argument("--render-stills", type=str, default="", dest="render_stills",
                        help="Directory for front / three-quarter stills")
    parser.add_argument("--meta", type=str, default="", help="Write a JSON sidecar with settings and runtime")
    parser.add_argument("--ss-steps", type=int, default=None, dest="ss_steps")
    parser.add_argument("--shape-steps", type=int, default=None, dest="shape_steps")
    parser.add_argument("--tex-steps", type=int, default=None, dest="tex_steps")

    args = parser.parse_args()
    preset = dict(PRESETS[args.preset])
    resolution = args.resolution if args.resolution > 0 else preset.get("resolution", -1)
    cleanup = args.cleanup or bool(preset.get("cleanup", False))
    smooth = args.smooth or bool(preset.get("smooth", False))
    keep_alpha = args.keep_alpha or bool(preset.get("keep_alpha", False))
    ss_steps = args.ss_steps if args.ss_steps is not None else preset.get("ss_sampling_steps", 12)
    shape_steps = args.shape_steps if args.shape_steps is not None else preset.get("shape_slat_sampling_steps", 12)
    tex_steps = args.tex_steps if args.tex_steps is not None else preset.get("tex_slat_sampling_steps", 12)

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=resolution,
        ss_sampling_steps=ss_steps,
        shape_slat_sampling_steps=shape_steps,
        tex_slat_sampling_steps=tex_steps,
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        cleanup=cleanup,
        smooth=smooth,
        keep_alpha=keep_alpha,
        save_preprocessed=args.save_preprocessed,
        render_stills=args.render_stills,
        meta_path=args.meta,
        remesh=not args.no_remesh,
    )
