#!/usr/bin/env python3
"""
run_analysis_and_figures.py
===========================

End-to-end demo: loads all synthetic cell tables, runs the same analyses
specified in 03_spatial_analysis.R (UMAP, HDBSCAN, DBSCAN busyness, KNN
neighborhoods, t-tests), and writes figures to /home/claude/demo/figures/.

All figures are stamped "DEMO — synthetic data, not real results".
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import umap

CELLTABLES = Path("/home/claude/demo/celltables")
FIG_DIR    = Path("/home/claude/demo/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEMO_STAMP = "DEMO — synthetic data, not real results"

VME_COLORS = {
    "SIV_positive": "#d7191c",
    "SIV_neighbor": "#fdae61",
    "SIV_negative": "#2b83ba",
}
GROUP_COLORS = {"LAEA": "#8c2d04", "EAEA": "#2c7fb8"}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
files = sorted(CELLTABLES.glob("*_celltable.csv"))
cells = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Loaded {len(cells)} cells from {len(files)} samples")
print(cells.groupby(["group", "vme_category"]).size().unstack(fill_value=0))

mean_cols = [c for c in cells.columns if c.endswith("_mean")]

def stamp(ax):
    ax.text(0.99, 0.01, DEMO_STAMP, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="gray", style="italic")

# ---------------------------------------------------------------------------
# Figure 1 — Point maps per sample (VME categories)
# ---------------------------------------------------------------------------
print("Making point-map figure...")
samples = cells.sample_id.unique()
n_cols = 4
n_rows = int(np.ceil(len(samples) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                         squeeze=False)
for ax, sid in zip(axes.flat, samples):
    sub = cells[cells.sample_id == sid]
    for cat, col in VME_COLORS.items():
        m = sub.vme_category == cat
        ax.scatter(sub.x[m], -sub.y[m], s=2, c=col, alpha=0.7, label=cat)
    ax.set_aspect("equal")
    ax.set_title(f"{sid}\n({sub.group.iloc[0]})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    stamp(ax)
# hide empty axes
for ax in axes.flat[len(samples):]:
    ax.axis("off")
axes.flat[0].legend(loc="upper left", bbox_to_anchor=(0, 1.3), fontsize=8,
                    ncol=3, frameon=False)
fig.suptitle("Per-sample point maps — VME categories (synthetic demo)",
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_vme_pointmaps.png", dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2 — UMAP on per-cell marker intensities
# ---------------------------------------------------------------------------
print("Running UMAP on per-cell intensities...")
X = cells[mean_cols].values
X_scaled = MinMaxScaler().fit_transform(X)

rng = np.random.default_rng(42)
if len(X_scaled) > 12000:
    idx = rng.choice(len(X_scaled), 12000, replace=False)
else:
    idx = np.arange(len(X_scaled))

reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
emb = reducer.fit_transform(X_scaled[idx])
umap_df = pd.DataFrame({
    "UMAP1": emb[:, 0], "UMAP2": emb[:, 1],
    "group": cells.group.iloc[idx].values,
    "vme":   cells.vme_category.iloc[idx].values,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, grp in zip(axes, ["LAEA", "EAEA"]):
    sub = umap_df[umap_df.group == grp]
    for cat, col in VME_COLORS.items():
        m = sub.vme == cat
        ax.scatter(sub.UMAP1[m], sub.UMAP2[m], s=3, c=col, alpha=0.6, label=cat)
    ax.set_title(f"{grp} (n={len(sub)} cells)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    stamp(ax)
axes[0].legend(loc="best", fontsize=8)
fig.suptitle("UMAP of per-cell marker intensities (synthetic demo)",
             fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_umap_vme.png", dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3 — HDBSCAN cluster sizes for SIV+ cells per group
# ---------------------------------------------------------------------------
print("Running HDBSCAN on SIV+ cells per sample...")
cluster_rows = []
for sid, sub in cells.groupby("sample_id"):
    siv = sub[sub.vme_category == "SIV_positive"]
    if len(siv) < 5:
        continue
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(siv[["x", "y"]].values)
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        cluster_rows.append({
            "sample_id": sid,
            "group": sub.group.iloc[0],
            "cluster_size": int((labels == lbl).sum()),
        })
cluster_df = pd.DataFrame(cluster_rows)

fig, ax = plt.subplots(figsize=(8, 5))
for grp, col in GROUP_COLORS.items():
    sub = cluster_df[cluster_df.group == grp]
    ax.hist(sub.cluster_size, bins=np.arange(2, 20),
            alpha=0.6, color=col, label=f"{grp} (n={len(sub)} clusters)",
            edgecolor="white")
ax.set_xlabel("HDBSCAN cluster size (n cells)")
ax.set_ylabel("Number of clusters")
ax.set_title("Cluster-size distribution for SIV+ cells (synthetic demo)")
ax.legend()
stamp(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_siv_cluster_size.png", dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4 — Tissue busyness (DBSCAN-based local density, bandwidth=300)
# ---------------------------------------------------------------------------
print("Computing local density / busyness...")
busyness_rows = []
for sid, sub in cells.groupby("sample_id"):
    pts = sub[["x", "y"]].values
    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, r=300, return_length=True)
    busyness_rows.append({
        "sample_id": sid,
        "group": sub.group.iloc[0],
        "mean_local_density": float(np.mean(counts)),
        "median_local_density": float(np.median(counts)),
    })
busy_df = pd.DataFrame(busyness_rows)

fig, ax = plt.subplots(figsize=(7, 5))
for i, grp in enumerate(["LAEA", "EAEA"]):
    sub = busy_df[busy_df.group == grp]
    x = np.full(len(sub), i) + rng.normal(0, 0.05, len(sub))
    ax.scatter(x, sub.mean_local_density, s=60, c=GROUP_COLORS[grp],
               label=grp, edgecolor="white", linewidth=1)
    ax.hlines(sub.mean_local_density.mean(), i-0.2, i+0.2,
              colors=GROUP_COLORS[grp], linewidth=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["LAEA", "EAEA"])
ax.set_ylabel("Mean local cell density (cells within 300 px)")
ax.set_title("Tissue busyness per sample (synthetic demo)")
stamp(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_tissue_busyness.png", dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 5 — KNN neighborhood composition around SIV+ cells
# ---------------------------------------------------------------------------
print("Running KNN neighborhood analysis...")

def lineage_call(row):
    if row["FoxP3_pos"] and row["CD4_pos"]:              return "Treg"
    if row["CD8_pos"]:                                    return "CD8_T"
    if row["CD4_pos"] and row["CD3_pos"]:                 return "CD4_T"
    if row["CD3_pos"]:                                    return "T_other"
    if row["CD117_pos"]:                                  return "Mast"
    if row["CD68_pos"]:                                   return "Myeloid"
    if row["CD20_pos"]:                                   return "B_cell"
    if row["CD138_pos"]:                                  return "Plasma"
    if row["EpCAM_pos"] or row["PanCK_pos"]:              return "Epithelial"
    return "Other"

cells["lineage"] = cells.apply(lineage_call, axis=1)

knn_rows = []
for sid, sub in cells.groupby("sample_id"):
    pts = sub[["x", "y"]].values
    siv_idx = np.where(sub.vme_category.values == "SIV_positive")[0]
    if len(siv_idx) == 0:
        continue
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts[siv_idx], k=11)  # 10 neighbors + self
    nn_idx = nn_idx[:, 1:]  # drop self
    neighbor_lineages = sub.lineage.values[nn_idx.flatten()]
    lineage_counts = pd.Series(neighbor_lineages).value_counts()
    total = lineage_counts.sum()
    for lineage, cnt in lineage_counts.items():
        knn_rows.append({
            "sample_id": sid,
            "group": sub.group.iloc[0],
            "lineage": lineage,
            "fraction": cnt / total,
        })
knn_df = pd.DataFrame(knn_rows)

# Keep lineages observed in at least 3 samples
lineage_counts = knn_df.groupby("lineage").sample_id.nunique()
keep = lineage_counts[lineage_counts >= 3].index.tolist()
knn_df = knn_df[knn_df.lineage.isin(keep)]

lineages = sorted(knn_df.lineage.unique())
n_lin = len(lineages)
n_cols = 4
n_rows = int(np.ceil(n_lin / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2*n_cols, 3*n_rows),
                         squeeze=False)

pvals = {}
for ax, lin in zip(axes.flat, lineages):
    sub = knn_df[knn_df.lineage == lin]
    data = [sub[sub.group == "LAEA"].fraction.values,
            sub[sub.group == "EAEA"].fraction.values]
    bp = ax.boxplot(data, labels=["LAEA", "EAEA"], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, col in zip(bp["boxes"],
                          [GROUP_COLORS["LAEA"], GROUP_COLORS["EAEA"]]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    for i, d in enumerate(data):
        x_j = np.full(len(d), i + 1) + rng.normal(0, 0.06, len(d))
        ax.scatter(x_j, d, s=20, c="black", zorder=3)

    if len(data[0]) >= 2 and len(data[1]) >= 2:
        tstat, p = stats.ttest_ind(data[0], data[1], equal_var=False)
    else:
        p = np.nan
    pvals[lin] = p
    ax.set_title(f"{lin}\np = {p:.3g}" if not np.isnan(p) else lin,
                 fontsize=10)
    ax.set_ylabel("fraction around SIV+")
    stamp(ax)

for ax in axes.flat[n_lin:]:
    ax.axis("off")
fig.suptitle("KNN (k=10) neighborhood composition around SIV+ cells "
             "(synthetic demo)", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_knn_neighborhood_boxplot.png",
            dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 6 — Group comparison bar plots + t-tests
# ---------------------------------------------------------------------------
print("Group-level summaries and t-tests...")
summary = cells.groupby(["sample_id", "group"]).agg(
    total_cells=("x", "size"),
    pct_SIV_positive=("vme_category",
                      lambda v: (v == "SIV_positive").mean() * 100),
    pct_SIV_neighbor=("vme_category",
                      lambda v: (v == "SIV_neighbor").mean() * 100),
).reset_index()

metrics = ["total_cells", "pct_SIV_positive", "pct_SIV_neighbor"]
titles  = ["Total cells", "% SIV+ cells", "% SIV-neighbor cells"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ttests = {}
for ax, m, t in zip(axes, metrics, titles):
    vals = [summary[summary.group == g][m].values for g in ["LAEA", "EAEA"]]
    bp = ax.boxplot(vals, labels=["LAEA", "EAEA"], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, col in zip(bp["boxes"],
                          [GROUP_COLORS["LAEA"], GROUP_COLORS["EAEA"]]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    for i, d in enumerate(vals):
        x_j = np.full(len(d), i + 1) + rng.normal(0, 0.06, len(d))
        ax.scatter(x_j, d, s=60, c="black", zorder=3)
    tstat, p = stats.ttest_ind(vals[0], vals[1], equal_var=False)
    ttests[m] = p
    ax.set_title(f"{t}\nUnpaired t-test p = {p:.3g}")
    ax.set_ylabel(t)
    stamp(ax)
fig.suptitle("LAEA vs. EAEA group comparisons (synthetic demo)",
             fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_group_comparison_barplots.png",
            dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Save tables too
# ---------------------------------------------------------------------------
summary.to_csv(FIG_DIR / "per_sample_summary.csv", index=False)
pd.DataFrame([{"metric": k, "p_value": v} for k, v in ttests.items()]).to_csv(
    FIG_DIR / "group_ttest_results.csv", index=False)
knn_df.to_csv(FIG_DIR / "knn_neighborhood_composition.csv", index=False)
cluster_df.to_csv(FIG_DIR / "siv_positive_cluster_sizes.csv", index=False)

print("\n" + "=" * 60)
print("Figures written to", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print(" -", f.name)
print("\nT-test p-values (LAEA vs EAEA):")
for m, p in ttests.items():
    print(f"  {m:22s} p = {p:.4g}")
