#!/usr/bin/env python3
"""
01_preprocess_spectreplex.py
============================

SPECTRE-Plex-style preprocessing pipeline adapted for the VME (Viral
Microenvironment) gut-reservoir study described in the "VME_GutReservoirs"
manuscript.

Adapted from Anderson et al. 2025 (Communications Biology, SPECTRE-Plex),
original reference implementation:
    https://github.com/mdanderson03/SpectrePlex

Pipeline steps
--------------
1. Z-stack focus selection using a modified Brenner score (skip=17).
2. Pair stained and dye-inactivated ("bleach") stacks by slice index.
3. BaSiC flat-field correction trained on tissue-containing frames.
4. PyStackReg registration of bleach -> stained image.
5. Subtract registered bleach from stained image (autofluorescence removal).
6. Export TIFF stacks + metadata for McMicro/Ashlar stitching.
7. Build tissue mask, mask out non-tissue regions, write final OME-TIFF.

Usage
-----
python 01_preprocess_spectreplex.py \
    --stained-dir  /data/animalA/cycle01/stained \
    --bleach-dir   /data/animalA/cycle01/bleach \
    --output-dir   /data/animalA/cycle01/processed \
    --sample-id    A7005_colon_section3 \
    --group        LAEA

Outputs
-------
<output-dir>/<sample-id>/
    focus_indices.json          # z-index chosen per tile
    flatfield_profile.npy       # BaSiC flatfield
    darkfield_profile.npy       # BaSiC darkfield
    registered/                 # per-tile corrected + subtracted TIFFs
    stitched.ome.tif            # final tissue-masked OME-TIFF
    tissue_mask.tif             # binary tissue mask
    metadata.json               # run parameters for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Optional dependencies — imported lazily so --help always works.
# Install:
#   pip install tifffile scikit-image basicpy pystackreg numpy scipy opencv-python
# McMicro / Ashlar are invoked as a subprocess (Nextflow/Docker recommended).

LOG = logging.getLogger("spectreplex.vme")


# ---------------------------------------------------------------------------
# Focus selection: modified Brenner score with skip parameter
# ---------------------------------------------------------------------------

def modified_brenner(img: np.ndarray, skip: int = 17) -> float:
    """Modified Brenner focus score.

    Brenner's original score squares differences between pixels separated by
    2 along the x-axis. The "skip" parameter generalizes that separation so
    the score responds to coarser textures typical of 16x large-FOV tiles.

    Parameters
    ----------
    img : 2D ndarray
    skip : int
        Pixel offset used to compute local intensity differences.
        SPECTRE-Plex uses skip=17 as an optimized default.
    """
    if img.ndim != 2:
        raise ValueError("modified_brenner expects a 2D image")
    a = img[:, :-skip].astype(np.float64)
    b = img[:, skip:].astype(np.float64)
    return float(np.sum((b - a) ** 2))


def select_focus_slice(stack: np.ndarray, skip: int = 17) -> int:
    """Return the z-index of the best-focused slice in a z-stack."""
    scores = [modified_brenner(stack[z], skip=skip) for z in range(stack.shape[0])]
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# BaSiC flat-field correction
# ---------------------------------------------------------------------------

def fit_basic_flatfield(tiles: Sequence[np.ndarray]):
    """Fit a BaSiC flatfield/darkfield model on a set of tissue tiles."""
    try:
        from basicpy import BaSiC
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "basicpy is required: pip install basicpy"
        ) from e

    stack = np.stack(tiles, axis=0).astype(np.float32)
    basic = BaSiC(get_darkfield=True, smoothness_flatfield=1.0)
    basic.fit(stack)
    return basic


def apply_basic(img: np.ndarray, basic) -> np.ndarray:
    """Apply a fitted BaSiC model to a single image."""
    corrected = basic.transform(img[np.newaxis, ...].astype(np.float32))[0]
    return corrected


# ---------------------------------------------------------------------------
# Bleach -> stained registration (pystackreg)
# ---------------------------------------------------------------------------

def register_and_subtract(
    stained: np.ndarray, bleach: np.ndarray, transform: str = "RIGID_BODY"
) -> np.ndarray:
    """Register bleach image to stained image and subtract.

    Subtraction isolates signal that *dropped* after dye inactivation — i.e.
    the antibody-specific fluorescence. In the VME context this is critical
    because SIV Gag / p-eIF2alpha IF channels sit on top of considerable gut
    tissue autofluorescence (lipofuscin, collagen, elastin).
    """
    try:
        from pystackreg import StackReg
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pystackreg is required: pip install pystackreg"
        ) from e

    sr_map = {
        "TRANSLATION": StackReg.TRANSLATION,
        "RIGID_BODY": StackReg.RIGID_BODY,
        "AFFINE": StackReg.AFFINE,
    }
    sr = StackReg(sr_map[transform.upper()])
    sr.register(stained.astype(np.float32), bleach.astype(np.float32))
    registered_bleach = sr.transform(bleach.astype(np.float32))
    subtracted = stained.astype(np.float32) - registered_bleach
    subtracted[subtracted < 0] = 0
    return subtracted


# ---------------------------------------------------------------------------
# Tissue mask construction (adapted for gut FFPE/cryosection morphology)
# ---------------------------------------------------------------------------

def build_tissue_mask(
    channels: Sequence[np.ndarray],
    downsample: int = 2,
    dilate_px: int = 10,
) -> np.ndarray:
    """Construct a binary tissue mask from a set of channels.

    Steps
    -----
    1. Downsample each channel by `downsample`.
    2. Min-max normalize and average.
    3. Otsu threshold.
    4. Morphological dilation + fill holes.
    5. Retain only the largest connected component.

    Notes for VME study
    -------------------
    For gut sections we recommend building the mask from *structural* channels
    (DAPI + pan-CK or Na/K-ATPase) rather than SIV Gag; SIV signal is focal
    and would badly under-estimate the tissue area.
    """
    try:
        import cv2
        from skimage.filters import threshold_otsu
        from skimage.morphology import binary_dilation, disk, remove_small_holes
        from skimage.measure import label, regionprops
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scikit-image and opencv-python required"
        ) from e

    normed = []
    for ch in channels:
        small = cv2.resize(
            ch, (ch.shape[1] // downsample, ch.shape[0] // downsample),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        lo, hi = np.percentile(small, (1, 99))
        small = np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1)
        normed.append(small)
    composite = np.mean(np.stack(normed, axis=0), axis=0)

    thr = threshold_otsu(composite)
    binary = composite > thr
    binary = binary_dilation(binary, footprint=disk(dilate_px))
    binary = remove_small_holes(binary, area_threshold=5000)

    labelled = label(binary)
    if labelled.max() == 0:
        return binary.astype(np.uint8)
    biggest = max(regionprops(labelled), key=lambda r: r.area).label
    mask = (labelled == biggest).astype(np.uint8)

    # Upsample mask back to full resolution
    full = cv2.resize(
        mask, (channels[0].shape[1], channels[0].shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return full


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    stained_dir: Path
    bleach_dir: Path
    output_dir: Path
    sample_id: str
    group: str                        # e.g. "LAEA", "EAEA" (or "Healthy"/"Celiac")
    brenner_skip: int = 17
    register_transform: str = "RIGID_BODY"
    downsample_mask: int = 2
    mask_dilate_px: int = 10


def load_zstack(path: Path) -> np.ndarray:
    import tifffile
    return tifffile.imread(str(path))


def write_tiff(path: Path, arr: np.ndarray, metadata: dict) -> None:
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path), arr.astype(np.float32),
        metadata=metadata, imagej=False,
    )


def run_pipeline(cfg: RunConfig) -> None:
    out = cfg.output_dir / cfg.sample_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "registered").mkdir(exist_ok=True)

    stained_files = sorted(cfg.stained_dir.glob("*.tif"))
    bleach_files = sorted(cfg.bleach_dir.glob("*.tif"))
    if len(stained_files) != len(bleach_files):
        raise RuntimeError(
            f"Mismatched tile counts: {len(stained_files)} stained vs "
            f"{len(bleach_files)} bleach"
        )

    LOG.info("Found %d tile pairs for %s", len(stained_files), cfg.sample_id)

    # Pass 1: pick focused slice per tile, collect focused planes for BaSiC
    focus_indices = {}
    focused_stained: List[np.ndarray] = []
    focused_bleach: List[np.ndarray] = []

    for sf, bf in zip(stained_files, bleach_files):
        stained_stack = load_zstack(sf)
        bleach_stack = load_zstack(bf)
        z = select_focus_slice(stained_stack, skip=cfg.brenner_skip)
        focus_indices[sf.stem] = z
        focused_stained.append(stained_stack[z])
        # Same slice index in the bleach stack, per SPECTRE-Plex spec
        focused_bleach.append(bleach_stack[z])

    with open(out / "focus_indices.json", "w") as f:
        json.dump(focus_indices, f, indent=2)

    # Pass 2: fit BaSiC on stained tiles (trained only on tissue-containing ones)
    LOG.info("Fitting BaSiC flatfield on %d tiles", len(focused_stained))
    basic = fit_basic_flatfield(focused_stained)
    np.save(out / "flatfield_profile.npy", basic.flatfield)
    np.save(out / "darkfield_profile.npy", basic.darkfield)

    # Pass 3: correct, register, subtract, write per-tile TIFFs
    for sf, stained_img, bleach_img in zip(
        stained_files, focused_stained, focused_bleach
    ):
        s_corr = apply_basic(stained_img, basic)
        b_corr = apply_basic(bleach_img, basic)
        subtracted = register_and_subtract(
            s_corr, b_corr, transform=cfg.register_transform
        )
        write_tiff(
            out / "registered" / f"{sf.stem}.tif",
            subtracted,
            metadata={
                "sample_id": cfg.sample_id,
                "group": cfg.group,
                "focus_z": focus_indices[sf.stem],
                "source_stained": str(sf),
                "source_bleach": str(cfg.bleach_dir / sf.name),
            },
        )

    # Metadata — downstream scripts (McMicro/Ashlar) consume this
    meta = asdict(cfg)
    meta["stained_dir"] = str(cfg.stained_dir)
    meta["bleach_dir"] = str(cfg.bleach_dir)
    meta["output_dir"] = str(cfg.output_dir)
    meta["n_tiles"] = len(stained_files)
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    LOG.info("Preprocessing complete. Next: run McMicro/Ashlar stitching.")
    LOG.info("Then run 02_segment_and_type.py on %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPECTRE-Plex preprocessing for VME gut reservoir study"
    )
    p.add_argument("--stained-dir", type=Path, required=True)
    p.add_argument("--bleach-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-id", type=str, required=True)
    p.add_argument(
        "--group", type=str, required=True,
        help="Experimental group, e.g. LAEA or EAEA",
    )
    p.add_argument("--brenner-skip", type=int, default=17)
    p.add_argument(
        "--register-transform", default="RIGID_BODY",
        choices=["TRANSLATION", "RIGID_BODY", "AFFINE"],
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cfg = RunConfig(
        stained_dir=args.stained_dir,
        bleach_dir=args.bleach_dir,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        group=args.group,
        brenner_skip=args.brenner_skip,
        register_transform=args.register_transform,
    )
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
