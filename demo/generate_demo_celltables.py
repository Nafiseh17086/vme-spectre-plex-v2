#!/usr/bin/env python3
"""
generate_demo_celltables.py
===========================

Generate synthetic cell tables mimicking what 02_segment_and_type.py would
output from real SPECTRE-Plex images. Each sample contains:

 - thousands of cells with (x, y) centroids on a ~2000x2000 pixel "tissue"
 - per-cell mean intensities for all markers in channels.yaml
 - Otsu-based positivity flags (_pos)
 - vme_category: SIV_positive / SIV_neighbor / SIV_negative

Two groups are simulated:
 - LAEA (n=3): persistent reservoirs — many small SIV foci, enriched with
               FoxP3+ Treg and CD117+ mast cells in the neighborhood
 - EAEA (n=4): transient reservoirs — fewer SIV foci, enriched with
               CD8 and CD4 T-cells in the neighborhood

This mirrors the biology reported in Hope, Crentsil et al. — purely for
demonstrating the pipeline's output. NOT real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)

OUT = Path("/home/claude/demo/celltables")
OUT.mkdir(parents=True, exist_ok=True)

MARKERS = [
    "DAPI", "NaKATPase", "EpCAM", "PanCK",
    "SIVGag", "p-eIF2a", "ATF4",
    "CD3", "CD4", "CD8", "FoxP3",
    "CD117", "CD68",
    "CD20", "CD138",
    "BSG", "CA2", "CA12",
]

TISSUE_SIZE = 2000  # pixels
N_CELLS_PER_SAMPLE = 4000

SAMPLES = [
    # (sample_id, group, n_SIV_foci, cells_per_focus, neighbor_profile)
    ("A7005_colon_s1", "LAEA", 12, 45, "Treg_mast"),
    ("A7005_colon_s2", "LAEA", 10, 50, "Treg_mast"),
    ("A7006_colon_s1", "LAEA",  9, 40, "Treg_mast"),
    ("E2001_colon_s1", "EAEA",  4, 60, "CD8_Tfh"),
    ("E2002_colon_s1", "EAEA",  3, 55, "CD8_Tfh"),
    ("E2003_colon_s1", "EAEA",  5, 50, "CD8_Tfh"),
    ("E2004_colon_s1", "EAEA",  4, 45, "CD8_Tfh"),
]


def make_sample(sample_id, group, n_foci, cells_per_focus, profile):
    n_bg = N_CELLS_PER_SAMPLE - n_foci * cells_per_focus - 200  # leave room for neighbors
    if n_bg < 0:
        n_bg = 500

    # 1. Background cells scattered uniformly
    bg_x = rng.uniform(0, TISSUE_SIZE, n_bg)
    bg_y = rng.uniform(0, TISSUE_SIZE, n_bg)
    bg_vme = np.array(["SIV_negative"] * n_bg)

    # 2. SIV foci — clusters of cells around random centers
    focus_centers = rng.uniform(100, TISSUE_SIZE - 100, size=(n_foci, 2))
    siv_x, siv_y, siv_vme = [], [], []
    for cx, cy in focus_centers:
        # SIV_positive core
        n_core = rng.integers(3, 8)
        cx_jitter = rng.normal(cx, 15, n_core)
        cy_jitter = rng.normal(cy, 15, n_core)
        siv_x.extend(cx_jitter)
        siv_y.extend(cy_jitter)
        siv_vme.extend(["SIV_positive"] * n_core)
        # SIV_neighbor halo (within ~100 um)
        n_halo = cells_per_focus - n_core
        halo_r = rng.uniform(30, 100, n_halo)
        halo_theta = rng.uniform(0, 2 * np.pi, n_halo)
        siv_x.extend(cx + halo_r * np.cos(halo_theta))
        siv_y.extend(cy + halo_r * np.sin(halo_theta))
        siv_vme.extend(["SIV_neighbor"] * n_halo)

    x = np.concatenate([bg_x, siv_x])
    y = np.concatenate([bg_y, siv_y])
    vme = np.concatenate([bg_vme, siv_vme])

    # Clip to tissue
    x = np.clip(x, 0, TISSUE_SIZE)
    y = np.clip(y, 0, TISSUE_SIZE)
    n_cells = len(x)

    # 3. Generate per-marker intensities
    df = pd.DataFrame({
        "x": x, "y": y,
        "area": rng.integers(80, 250, n_cells),
    })

    # DAPI — all cells
    df["DAPI_mean"] = rng.normal(6000, 1500, n_cells)

    # Structural — mostly epithelial cells (randomly 40% of tissue)
    is_epi = rng.random(n_cells) < 0.40
    df["NaKATPase_mean"] = np.where(is_epi, rng.normal(4500, 900, n_cells), rng.normal(800, 300, n_cells))
    df["EpCAM_mean"]     = np.where(is_epi, rng.normal(3800, 800, n_cells), rng.normal(600, 200, n_cells))
    df["PanCK_mean"]     = np.where(is_epi, rng.normal(3200, 700, n_cells), rng.normal(500, 200, n_cells))

    # SIV Gag — high in SIV_positive, near-zero elsewhere
    df["SIVGag_mean"] = np.where(
        vme == "SIV_positive",
        rng.normal(5500, 1200, n_cells),
        rng.normal(200, 100, n_cells),
    )

    # p-eIF2a (ISR marker) — elevated in LAEA SIV+ cells (persistent reservoir signature)
    peif_base = np.where(vme == "SIV_positive",
                         rng.normal(3500 if group == "LAEA" else 1500, 800, n_cells),
                         rng.normal(400, 150, n_cells))
    df["p-eIF2a_mean"] = peif_base

    # ATF4 — co-varies with p-eIF2a in LAEA
    df["ATF4_mean"] = np.where(
        (vme == "SIV_positive") & (group == "LAEA"),
        rng.normal(3000, 700, n_cells),
        rng.normal(500, 200, n_cells),
    )

    # BSG/CD147 — top VME regressor from manuscript; elevated near SIV foci in LAEA
    df["BSG_mean"] = np.where(
        (vme != "SIV_negative") & (group == "LAEA"),
        rng.normal(3200, 800, n_cells),
        rng.normal(600, 250, n_cells),
    )

    # CA2/CA12 — hypoxia; LAEA-specific
    df["CA2_mean"]  = np.where((vme == "SIV_positive") & (group == "LAEA"),
                               rng.normal(2800, 700, n_cells), rng.normal(400, 150, n_cells))
    df["CA12_mean"] = np.where((vme != "SIV_negative") & (group == "LAEA"),
                               rng.normal(2500, 600, n_cells), rng.normal(350, 140, n_cells))

    # T-cell lineages — baseline random, but neighbor-enriched per group
    # LAEA: Treg + mast dominant near SIV foci. EAEA: CD8 + CD4 dominant.
    is_siv_region = (vme != "SIV_negative")

    if profile == "Treg_mast":
        # LAEA: neighbors have elevated FoxP3 (Treg) and CD117 (mast)
        df["CD3_mean"]   = rng.normal(np.where(is_siv_region, 1200, 800),   400)
        df["CD4_mean"]   = rng.normal(np.where(is_siv_region, 1800, 500),   500)
        df["CD8_mean"]   = rng.normal(np.where(is_siv_region, 400,  600),   300)
        df["FoxP3_mean"] = rng.normal(np.where(is_siv_region, 2500, 400),   600)
        df["CD117_mean"] = rng.normal(np.where(is_siv_region, 2200, 300),   500)
    else:  # CD8_Tfh — EAEA
        df["CD3_mean"]   = rng.normal(np.where(is_siv_region, 2200, 700),   500)
        df["CD4_mean"]   = rng.normal(np.where(is_siv_region, 1500, 500),   400)
        df["CD8_mean"]   = rng.normal(np.where(is_siv_region, 2800, 400),   700)
        df["FoxP3_mean"] = rng.normal(np.where(is_siv_region, 500,  300),   200)
        df["CD117_mean"] = rng.normal(np.where(is_siv_region, 400,  400),   200)

    df["CD68_mean"]  = rng.normal(700, 300, n_cells)
    df["CD20_mean"]  = rng.normal(600, 250, n_cells)
    df["CD138_mean"] = rng.normal(500, 200, n_cells)

    # Ensure non-negative
    for col in [c for c in df.columns if c.endswith("_mean")]:
        df[col] = np.clip(df[col], 0, None)

    # 4. Otsu-style positivity — simple threshold at 75th percentile per marker
    from skimage.filters import threshold_otsu
    for m in MARKERS:
        col = f"{m}_mean"
        vals = df[col].values
        if np.unique(vals).size < 2:
            df[f"{m}_pos"] = False
            continue
        thr = threshold_otsu(vals)
        df[f"{m}_pos"] = df[col] > thr

    # 5. VME category column
    df["vme_category"] = vme
    df["sample_id"] = sample_id
    df["group"] = group

    return df


manifest = []
for sid, grp, n_foci, cpf, profile in SAMPLES:
    df = make_sample(sid, grp, n_foci, cpf, profile)
    out_file = OUT / f"{sid}_celltable.csv"
    df.to_csv(out_file, index=False)
    manifest.append({
        "sample_id": sid, "group": grp,
        "n_cells": len(df),
        "n_SIV_positive": int((df.vme_category == "SIV_positive").sum()),
        "n_SIV_neighbor": int((df.vme_category == "SIV_neighbor").sum()),
        "n_SIV_negative": int((df.vme_category == "SIV_negative").sum()),
    })
    print(f"  wrote {out_file.name}: {len(df)} cells, "
          f"{manifest[-1]['n_SIV_positive']} SIV+, "
          f"{manifest[-1]['n_SIV_neighbor']} SIV-neighbor")

print("\nDone. Cell tables in", OUT)
