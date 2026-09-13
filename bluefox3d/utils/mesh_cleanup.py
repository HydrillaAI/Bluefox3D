"""Optional mesh cleanup used by the BlueFox3D wrapper (not the Pixal3D baseline)."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from trimesh import smoothing


def _as_numpy(mesh):
    verts = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().int().cpu().numpy()
    return verts, faces


def drop_small_islands(vertices: np.ndarray, faces: np.ndarray, min_face_ratio: float = 0.02):
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    parts = tm.split(only_watertight=False)
    if len(parts) <= 1:
        return vertices, faces
    total = max(len(faces), 1)
    keep = [p for p in parts if len(p.faces) >= min_face_ratio * total]
    if not keep:
        keep = [max(parts, key=lambda p: len(p.faces))]
    merged = trimesh.util.concatenate(keep)
    return np.asarray(merged.vertices, dtype=np.float32), np.asarray(merged.faces, dtype=np.int32)


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 2, lamb: float = 0.3):
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    smoothing.filter_laplacian(tm, lamb=lamb, iterations=iterations)
    return np.asarray(tm.vertices, dtype=np.float32), np.asarray(tm.faces, dtype=np.int32)


def cleanup_mesh(mesh, drop_islands: bool = True, smooth: bool = True, min_face_ratio: float = 0.02):
    """Rewrite mesh.vertices / mesh.faces in place. Voxel attrs stay on the grid."""
    device = mesh.vertices.device
    verts, faces = _as_numpy(mesh)
    n_faces_before = int(faces.shape[0])
    if drop_islands:
        verts, faces = drop_small_islands(verts, faces, min_face_ratio=min_face_ratio)
    if smooth and faces.shape[0] > 0:
        verts, faces = laplacian_smooth(verts, faces)
    mesh.vertices = torch.from_numpy(np.ascontiguousarray(verts)).to(device=device, dtype=torch.float32)
    mesh.faces = torch.from_numpy(np.ascontiguousarray(faces)).to(device=device, dtype=torch.int32)
    return {
        "faces_before": n_faces_before,
        "faces_after": int(faces.shape[0]),
    }
