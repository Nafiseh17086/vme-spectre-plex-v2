#!/usr/bin/env python3
"""
02_segment_and_type.py
======================

Segmentation and per-cell marker quantification for SPECTRE-Plex multiplex
images, adapted for the VME gut reservoir study.

Key adaptations vs. the original Anderson et al. duodenum pipeline
------------------------------------------------------------------
1. Boundary image is built from DAPI + Na/K-ATPase + EpCAM + Pan-CK
   (same as SPECTRE-Plex), but the cell-typing panel is expanded to
   include VME-relevant markers:
     - SIV Gag          (viral antigen)
     - p-eIF2alpha      (integrated stress response)
     - CD3, CD4, CD8    (T-cell lineages)
     - FoxP3            (Treg)
     - CD117            (mast cells)
     - CD20 / CD138     (B / plasma cells)
     - CD68             (myeloid)
     - BSG / CD147      (top VME regressor from the manuscript)
     - CA2, CA12        (hypoxia / tumor-like VME)
     - ATF4             (ISR driver)

2. Per-cell calls are exported with an explicit column for VME category
   so downstream R scripts can build SIV-positive / SIV-neighbor /
   SIV-negative labels analogous to the Visium spot categories.

Usage
-----
python 02_segment_and_type.py \
    --stitched /data/animalA/cycle01/processed/A7005_colon_section3/stitched.ome.tif \
    --channels channels.yaml \
    --sample-id A7005_colon_section3 \
    --group LAEA \
    --output-dir /data/animalA/cycle01/celltable

`channels.yaml` maps channel indices to marker names, e.g.:
    0: DAPI
    1: NaKATPase
    2: EpCAM
    3: PanCK
    4: SIVGag
    5: p-eIF2a
    ...
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("spectreplex.vme.seg")

BOUNDARY_MARKERS = ("NaKATPase", "EpCAM", "PanCK")
NUCLEAR_MARKER = "DAPI"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_channel_map(path: Path) -> Dict[int, str]:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {int(k): str(v) for k, v in raw.items()}


def load_stitched(path: Path) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(
            f"Expected CYX stitched image, got shape {arr.shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# Normalization and boundary image construction
# ---------------------------------------------------------------------------

def minmax(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_pct, hi_pct))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def build_boundary_image(
    channels: np.ndarray, ch_map: Dict[int, str]
) -> np.ndarray:
    """Average Na/K-ATPase + EpCAM + Pan-CK after min-max normalization."""
    inv = {v: k for k, v in ch_map.items()}
    missing = [m for m in BOUNDARY_MARKERS if m not in inv]
    if missing:
        LOG.warning(
            "Boundary markers missing from channel map: %s — "
            "falling back to whichever are present.",
            missing,
        )
    present = [m for m in BOUNDARY_MARKERS if m in inv]
    if not present:
        raise RuntimeError(
            "No boundary markers found; cannot build segmentation input."
        )
    stack = np.stack([minmax(channels[inv[m]]) for m in present], axis=0)
    return stack.mean(axis=0)


# ---------------------------------------------------------------------------
# Cellpose segmentation (cyto3, flow_threshold = 0.0 per SPECTRE-Plex)
# ---------------------------------------------------------------------------

def segment_cells(
    boundary: np.ndarray, nuclei: np.ndarray, gpu: bool = False
) -> np.ndarray:
    try:
        from cellpose import models
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "cellpose>=3.0.7 is required: pip install cellpose"
        ) from e

    model = models.Cellpose(gpu=gpu, model_type="cyto3")
    # Stack into a 2-channel image: [cyto, nucleus]
    img = np.stack([boundary, minmax(nuclei)], axis=-1)
    masks, _, _, _ = model.eval(
        img, channels=[1, 2], flow_threshold=0.0, diameter=None,
    )
    return masks.astype(np.int32)


# ---------------------------------------------------------------------------
# Per-cell marker intensity extraction + Otsu-based positivity calls
# ---------------------------------------------------------------------------

def per_cell_intensities(
    channels: np.ndarray, masks: np.ndarray, ch_map: Dict[int, str]
) -> pd.DataFrame:
    from skimage.measure import regionprops_table

    rows: List[Dict] = []
    for ch_idx, marker in ch_map.items():
        props = regionprops_table(
            masks,
            intensity_image=channels[ch_idx],
            properties=("label", "centroid", "area", "mean_intensity"),
        )
        df = pd.DataFrame(props).rename(
            columns={
                "centroid-0": "y",
                "centroid-1": "x",
                "mean_intensity": f"{marker}_mean",
            }
        )
        rows.append(df.set_index("label")[[f"{marker}_mean"]])

    # Merge per-marker intensity tables on cell label
    merged = pd.concat(rows, axis=1)
    # Add centroid/area from any one channel's regionprops pass
    ref = pd.DataFrame(
        regionprops_table(
            masks, properties=("label", "centroid", "area"),
        )
    ).rename(columns={"centroid-0": "y", "centroid-1": "x"})
    merged = ref.set_index("label").join(merged).reset_index()
    return merged


def call_positivity(df: pd.DataFrame, markers: List[str]) -> pd.DataFrame:
    """Apply Otsu's threshold per marker to assign +/-."""
    from skimage.filters import threshold_otsu

    df = df.copy()
    for m in markers:
        col = f"{m}_mean"
        if col not in df.columns:
            continue
        vals = df[col].values
        # Guard against degenerate channels (all zero or constant)
        if np.unique(vals).size < 2:
            df[f"{m}_pos"] = False
            continue
        try:
            thr = threshold_otsu(vals)
        except ValueError:
            thr = np.percentile(vals, 95)
        df[f"{m}_pos"] = df[col] > thr
    return df


# ---------------------------------------------------------------------------
# VME-specific category assignment
# ---------------------------------------------------------------------------

def assign_vme_category(
    df: pd.DataFrame, neighbor_radius_um: float = 100.0,
    px_per_um: float = 1.0,
) -> pd.DataFrame:
    """Mirror the Visium SIV+/SIV-neighbor/SIV- classification at single-cell
    resolution.

    - SIV_positive  : cell is SIVGag_pos == True
    - SIV_neighbor  : cell within neighbor_radius_um of any SIV_positive cell
    - SIV_negative  : otherwise
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:  # pragma: no cover
        raise ImportError("scipy is required") from e

    df = df.copy()
    df["vme_category"] = "SIV_negative"

    if "SIVGag_pos" not in df.columns:
        LOG.warning("No SIVGag_pos column — all cells labelled SIV_negative.")
        return df

    pos_mask = df["SIVGag_pos"].values.astype(bool)
    df.loc[pos_mask, "vme_category"] = "SIV_positive"

    if pos_mask.sum() == 0:
        return df

    xy = df[["x", "y"]].to_numpy() / px_per_um  # convert to microns
    pos_xy = xy[pos_mask]
    tree = cKDTree(pos_xy)
    dists, _ = tree.query(xy, k=1)
    neighbor_mask = (dists <= neighbor_radius_um) & (~pos_mask)
    df.loc[neighbor_mask, "vme_category"] = "SIV_neighbor"
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    stitched_path: Path,
    channels_yaml: Path,
    sample_id: str,
    group: str,
    output_dir: Path,
    px_per_um: float = 1.0,
    neighbor_radius_um: float = 100.0,
    gpu: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ch_map = load_channel_map(channels_yaml)
    channels = load_stitched(stitched_path)

    boundary = build_boundary_image(channels, ch_map)
    nuclei_idx = {v: k for k, v in ch_map.items()}[NUCLEAR_MARKER]
    masks = segment_cells(boundary, channels[nuclei_idx], gpu=gpu)

    LOG.info("Segmented %d cells", int(masks.max()))

    df = per_cell_intensities(channels, masks, ch_map)
    markers = list(ch_map.values())
    df = call_positivity(df, markers)
    df = assign_vme_category(
        df, neighbor_radius_um=neighbor_radius_um, px_per_um=px_per_um,
    )

    df["sample_id"] = sample_id
    df["group"] = group

    out_csv = output_dir / f"{sample_id}_celltable.csv"
    df.to_csv(out_csv, index=False)

    manifest = {
        "sample_id": sample_id,
        "group": group,
        "n_cells": int(len(df)),
        "n_SIV_positive": int((df["vme_category"] == "SIV_positive").sum()),
        "n_SIV_neighbor": int((df["vme_category"] == "SIV_neighbor").sum()),
        "n_SIV_negative": int((df["vme_category"] == "SIV_negative").sum()),
        "channels": ch_map,
        "stitched_source": str(stitched_path),
        "neighbor_radius_um": neighbor_radius_um,
    }
    with open(output_dir / f"{sample_id}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    LOG.info("Wrote %s (%d cells)", out_csv, len(df))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stitched", type=Path, required=True)
    p.add_argument("--channels", type=Path, required=True, help="YAML channel map")
    p.add_argument("--sample-id", type=str, required=True)
    p.add_argument("--group", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--px-per-um", type=float, default=1.0)
    p.add_argument("--neighbor-radius-um", type=float, default=100.0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run(
        stitched_path=args.stitched,
        channels_yaml=args.channels,
        sample_id=args.sample_id,
        group=args.group,
        output_dir=args.output_dir,
        px_per_um=args.px_per_um,
        neighbor_radius_um=args.neighbor_radius_um,
        gpu=args.gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
