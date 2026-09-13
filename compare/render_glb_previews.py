#!/usr/bin/env python3
"""Render front and 3/4 stills. Each GLB runs in a fresh Python process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Inverse of the Blender/glTF export rotation in inference.py (R^2 = I).
GLB_TO_MESH = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def _as_rgb_float(image) -> np.ndarray:
    if hasattr(image, "convert"):
        rgb = image.convert("RGB")
    else:
        rgb = Image.fromarray(np.asarray(image)).convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr


def _color_factor(material) -> list[float]:
    factor = np.ones(3, dtype=np.float32)
    raw_factor = getattr(material, "baseColorFactor", None)
    if raw_factor is not None:
        factor = np.asarray(raw_factor, dtype=np.float32).reshape(-1)[:3]
        if float(np.max(factor)) > 1.5:
            factor = factor / 255.0
    return [float(x) for x in factor]


def load_textured_mesh(glb_path: Path):
    import torch
    import trimesh
    from bluefox3d.representations.mesh.base import Mesh, MeshWithPbrMaterial, PbrMaterial, Texture

    loaded = trimesh.load(str(glb_path), force="scene")
    geom = next(iter(loaded.geometry.values())) if hasattr(loaded, "geometry") else loaded
    verts = np.asarray(geom.vertices, dtype=np.float32) @ GLB_TO_MESH
    # Generation / glTF export leaves the mesh inverted for the Z-up camera.
    verts[:, 1] *= -1.0
    verts[:, 2] *= -1.0
    faces = np.asarray(geom.faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError("empty mesh")

    center = (verts.max(axis=0) + verts.min(axis=0)) * 0.5
    verts = verts - center
    scale = float(np.linalg.norm(verts, axis=1).max()) or 1.0
    verts = verts / scale * 0.85

    uvs = getattr(getattr(geom, "visual", None), "uv", None)
    material = getattr(getattr(geom, "visual", None), "material", None)
    raw = getattr(material, "baseColorTexture", None) if material is not None else None
    if raw is None and material is not None:
        raw = getattr(material, "image", None)

    if uvs is not None and raw is not None:
        tex_img = _as_rgb_float(raw)
        if tex_img.shape[0] > 1024 or tex_img.shape[1] > 1024:
            tex_img = np.asarray(
                Image.fromarray((np.clip(tex_img, 0, 1) * 255).astype(np.uint8)).resize(
                    (1024, 1024), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        face_uvs = np.asarray(uvs, dtype=np.float32)[faces]
        texture = Texture(torch.from_numpy(np.ascontiguousarray(tex_img)).cuda())
        pbr = PbrMaterial(base_color_texture=texture, base_color_factor=_color_factor(material))
        return MeshWithPbrMaterial(
            torch.from_numpy(verts).cuda(),
            torch.from_numpy(faces).cuda(),
            torch.zeros(len(faces), dtype=torch.int32, device="cuda"),
            torch.from_numpy(np.ascontiguousarray(face_uvs)).cuda(),
            [pbr],
        ).cuda()

    return Mesh(torch.from_numpy(verts).cuda(), torch.from_numpy(faces).cuda())


def to_hwc(t) -> np.ndarray:
    arr = t.detach().float().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def composite_still(result, bg=(248, 248, 248)) -> np.ndarray:
    mask = to_hwc(result["mask"])[..., 0] if "mask" in result else None
    if "base_color" in result or "attr" in result:
        albedo = to_hwc(result["base_color"] if "base_color" in result else result["attr"]).astype(np.float32)
        if albedo.shape[-1] > 3:
            albedo = albedo[..., :3]
        if float(np.nanmax(albedo)) > 1.5:
            albedo = albedo / 255.0
        albedo = np.clip(np.nan_to_num(albedo, nan=0.0), 0.0, 1.0)
    else:
        albedo = np.full((*to_hwc(result["normal"]).shape[:2], 3), 0.82, dtype=np.float32)

    if "normal" in result:
        normal = to_hwc(result["normal"])
        n = normal * 2.0 - 1.0
        light = np.array([0.28, 0.18, 0.94], dtype=np.float32)
        light /= np.linalg.norm(light)
        lambert = np.clip((n * light).sum(axis=-1, keepdims=True), 0.0, 1.0)
        rgb = np.clip(np.power(np.clip(albedo, 0.0, 1.0), 0.85) * (0.55 + 0.45 * lambert) * 1.25, 0.0, 1.0)
    else:
        rgb = albedo

    frame = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    if mask is not None:
        canvas = np.full_like(frame, bg)
        visible = mask > 0.5
        canvas[visible] = frame[visible]
        return canvas
    return frame


def render_one(glb_path: Path, out_dir: Path, resolution: int = 512) -> None:
    from bluefox3d.renderers import MeshRenderer
    from bluefox3d.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    mesh = load_textured_mesh(glb_path)
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 2

    yaws = [np.pi / 2, np.pi / 2 + np.pi / 4]
    pitches = [0.22, 0.28]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, 2.0, 40)
    out_dir.mkdir(parents=True, exist_ok=True)
    from bluefox3d.representations.mesh.base import MeshWithPbrMaterial

    textured = isinstance(mesh, MeshWithPbrMaterial) or mesh.vertex_attrs is not None
    return_types = ["attr", "normal", "mask"] if textured else ["normal", "mask"]
    for name, extr, intr in zip(("front.png", "three_quarter.png"), extrinsics, intrinsics):
        result = renderer.render(mesh, extr, intr, return_types=return_types)
        Image.fromarray(composite_still(result)).save(out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("/home/dharani/work/compare/runs"))
    parser.add_argument("--one-glb", type=Path, default=None)
    parser.add_argument("--stills", type=Path, default=None)
    args = parser.parse_args()

    if args.one_glb is not None:
        dest = args.stills or (args.one_glb.parent / "stills")
        render_one(args.one_glb, dest)
        print("ok", args.one_glb)
        return

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "/home/dharani/work/BlueFox3D")
    for glb in sorted(args.runs.glob("*/*/output.glb")):
        stills = glb.parent / "stills"
        cmd = [py, str(Path(__file__).resolve()), "--one-glb", str(glb), "--stills", str(stills)]
        print("render", glb)
        code = subprocess.call(cmd, env=env)
        if code != 0:
            print("fail", glb, "exit", code)


if __name__ == "__main__":
    main()
