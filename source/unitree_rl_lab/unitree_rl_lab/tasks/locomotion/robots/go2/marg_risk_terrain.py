from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def _lerp(v0: float, v1: float, ratio: float) -> float:
    return float(v0 + ratio * (v1 - v0))


def _make_box_xy(
    *,
    size_x: float,
    size_y: float,
    top_z: float,
    height: float,
    center_x: float,
    center_y: float,
) -> trimesh.Trimesh:
    z_center = top_z - 0.5 * height
    dims = (size_x, size_y, height)
    transform = trimesh.transformations.translation_matrix((center_x, center_y, z_center))
    return trimesh.creation.box(dims, transform)


def _level_from_difficulty(difficulty: float, level_count: int) -> int:
    clamped = float(np.clip(difficulty, 0.0, 1.0))
    return int(np.floor(clamped * level_count)) + 1


def _rng_from_seed(seed: int | None, difficulty: float, level_id: int) -> np.random.Generator:
    base = 0 if seed is None else int(seed)
    diff_term = int(round(float(difficulty) * 1_000_000.0))
    hashed = (base * 1_000_003 + level_id * 7_919 + diff_term) & 0xFFFFFFFF
    return np.random.default_rng(hashed)


def _terrain_style_for_level(level_id: int) -> Literal[
    "single_gap",
    "stones_everywhere",
    "stones_2rows",
    "stones_balance",
    "beams_balance",
    "air_beams_balance",
]:
    level_styles = {
        1: "single_gap",
        2: "stones_everywhere",
        3: "stones_2rows",
        4: "stones_balance",
        5: "beams_balance",
        6: "air_beams_balance",
        7: "beams_balance",
        8: "air_beams_balance",
    }
    return level_styles[level_id]


def marg_risk_terrain(
    difficulty: float, cfg: "MargriskTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate risk-inspired MARG terrain from curriculum level.

    The level is inferred from difficulty. Higher level -> larger gaps, narrower stones/beams, and higher obstacles.
    """

    sx, sy = cfg.size
    level_id = _level_from_difficulty(difficulty=difficulty, level_count=cfg.level_count)
    ratio = (level_id - 1) / max(1, cfg.level_count - 1)
    rng = _rng_from_seed(seed=getattr(cfg, "seed", None), difficulty=difficulty, level_id=level_id)

    gap_size = _lerp(cfg.gap_range[0], cfg.gap_range[1], ratio)
    stone_size = _lerp(cfg.stone_size_range[0], cfg.stone_size_range[1], ratio)
    beam_width = _lerp(cfg.beam_width_range[0], cfg.beam_width_range[1], ratio)
    max_height = _lerp(cfg.height_range[0], cfg.height_range[1], ratio)
    style = _terrain_style_for_level(level_id)

    meshes: list[trimesh.Trimesh] = []

    # Shared spawn region keeps resets stable while still forcing traversal to harder zones.
    spawn = _make_box_xy(
        size_x=1.4,
        size_y=1.4,
        top_z=0.0,
        height=cfg.base_thickness,
        center_x=0.5 * sx,
        center_y=0.5 * sy,
    )
    meshes.append(spawn)

    if style == "single_gap":
        gap_center_x = 0.7 * sx
        left_len = max(1.2, gap_center_x - 0.5 * gap_size)
        right_start = gap_center_x + 0.5 * gap_size
        right_len = max(1.2, sx - right_start)

        meshes.append(
            _make_box_xy(
                size_x=left_len,
                size_y=sy,
                top_z=0.0,
                height=cfg.base_thickness,
                center_x=0.5 * left_len,
                center_y=0.5 * sy,
            )
        )
        meshes.append(
            _make_box_xy(
                size_x=right_len,
                size_y=sy,
                top_z=0.0,
                height=cfg.base_thickness,
                center_x=right_start + 0.5 * right_len,
                center_y=0.5 * sy,
            )
        )

    elif style == "stones_everywhere":
        pitch = stone_size + gap_size
        x_start = 0.7
        x_end = sx - 0.7
        y_start = 0.4
        y_end = sy - 0.4
        x_vals = np.arange(x_start, x_end, pitch)
        y_vals = np.arange(y_start, y_end, pitch)
        for cx in x_vals:
            for cy in y_vals:
                top_z = rng.uniform(cfg.height_range[0], max_height)
                meshes.append(
                    _make_box_xy(
                        size_x=stone_size,
                        size_y=stone_size,
                        top_z=top_z,
                        height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                        center_x=float(cx),
                        center_y=float(cy),
                    )
                )

    elif style == "stones_2rows":
        pitch = stone_size + gap_size
        x_vals = np.arange(0.8, sx - 0.8, pitch)
        y_offsets = (-0.3, 0.3)
        for cx in x_vals:
            for offset in y_offsets:
                top_z = rng.uniform(cfg.height_range[0], max_height)
                meshes.append(
                    _make_box_xy(
                        size_x=stone_size,
                        size_y=stone_size,
                        top_z=top_z,
                        height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                        center_x=float(cx),
                        center_y=float(0.5 * sy + offset),
                    )
                )

    elif style == "stones_balance":
        stone_x = stone_size
        stone_y = max(0.20, stone_size * 0.45)
        pitch = stone_x + gap_size
        x_vals = np.arange(0.7, sx - 0.7, pitch)
        for cx in x_vals:
            top_z = rng.uniform(cfg.height_range[0], max_height)
            meshes.append(
                _make_box_xy(
                    size_x=stone_x,
                    size_y=stone_y,
                    top_z=top_z,
                    height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                    center_x=float(cx),
                    center_y=0.5 * sy,
                )
            )

    elif style == "beams_balance":
        beam_len = 0.9
        pitch = beam_width + gap_size
        x_vals = np.arange(0.7, sx - 0.7, pitch)
        for cx in x_vals:
            top_z = rng.uniform(cfg.height_range[0], max_height)
            meshes.append(
                _make_box_xy(
                    size_x=beam_width,
                    size_y=beam_len,
                    top_z=top_z,
                    height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                    center_x=float(cx),
                    center_y=0.5 * sy,
                )
            )

    else:  # air_beams_balance
        beam_len = 0.8
        pitch = beam_width + gap_size
        x_vals = np.arange(0.8, sx - 0.8, pitch)
        for cx in x_vals:
            top_z = rng.uniform(max(cfg.height_range[0] + 0.02, 0.02), max_height)
            y_jitter = rng.uniform(-0.15, 0.15)
            meshes.append(
                _make_box_xy(
                    size_x=beam_width,
                    size_y=beam_len,
                    top_z=top_z,
                    height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                    center_x=float(cx),
                    center_y=float(0.5 * sy + y_jitter),
                )
            )

    origin = np.array([0.5 * sx, 0.5 * sy, 0.0])
    return meshes, origin


@configclass
class MargRiskTerrainCfg(SubTerrainBaseCfg):
    function = marg_risk_terrain

    level_count: int = 8
    base_thickness: float = 0.08

    gap_range: tuple[float, float] = (0.10, 0.60)
    stone_size_range: tuple[float, float] = (0.80, 0.24)
    beam_width_range: tuple[float, float] = (0.30, 0.12)
    height_range: tuple[float, float] = (0.00, 0.44)


MARG_RISK_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "marg_risk": MargRiskTerrainCfg(proportion=1.0),
    },
)
