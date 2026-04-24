# VME-SPECTRE-Plex: Spatial analysis of SIV gut reservoirs using SPECTRE-Plex multiplex imaging

This repository adapts the **SPECTRE-Plex** pipeline (Anderson et al., 2025, *Communications Biology*; [mdanderson03/SpectrePlex](https://github.com/mdanderson03/SpectrePlex)) for the **viral microenvironment (VME)** study of SIV persistent reservoirs in gut-associated lymphoid tissue (GALT) described in *VME_GutReservoirs_ST_manuscript_Final*.

The original SPECTRE-Plex pipeline was developed for the **healthy vs. celiac duodenum** comparison. Here we keep the core image-processing mechanics (focus scoring, BaSiC flat-field correction, bleach-stack subtraction via PyStackReg, McMicro/Ashlar stitching, Cellpose cyto3 segmentation, DBSCAN/HDBSCAN/KNN spatial analysis) but re-target the biology to **LAEA (late ART, persistent reservoir) vs. EAEA (early ART, transient reservoir)** macaque gut tissues, with a panel optimized for the VME features identified in the manuscript (SIV Gag, p-eIF2α, FoxP3+ Treg, CD117+ mast cells, BSG/CD147, CA2/CA12, ATF4).

## Why adapt SPECTRE-Plex?

The VME manuscript used 10x Visium spatial transcriptomics paired with IF, and categorized Visium spots as `SIV_positive`, `SIV_neighbor`, or `SIV_negative`. SPECTRE-Plex gives you the same three-tier spatial categorization at **single-cell resolution** with a much larger protein panel per cycle, at a much lower per-sample cost, and with enough cycles to resolve the immune architecture (Treg / mast / ILC cores) that machine learning flagged as the signature of persistent reservoirs.

## Pipeline overview

```
raw z-stacks (stained + bleach)
    │
    ▼
01_preprocess_spectreplex.py
  • Brenner-skip=17 focus selection
  • BaSiC flat-field correction
  • PyStackReg bleach→stained registration
  • Bleach subtraction → autofluorescence-corrected tiles
    │
    ▼
McMicro / Ashlar (external)
  • Tile stitching → stitched.ome.tif
    │
    ▼
02_segment_and_type.py
  • Cellpose cyto3 segmentation (flow_threshold=0.0)
  • Per-cell marker intensities
  • Otsu-based positivity calls
  • VME category assignment (SIV+ / SIV-neighbor / SIV-)
    │
    ▼
03_spatial_analysis.R
  • Point-maps per sample
  • UMAP on normalized intensities
  • HDBSCAN cluster sizes of SIV+ cells
  • DBSCAN local density ("busyness")
  • KNN neighborhood composition around SIV+ cells
  • Unpaired two-tailed t-tests, LAEA vs. EAEA
```

## Repository layout

```
vme_spectre/
├── README.md
├── scripts/
│   ├── 01_preprocess_spectreplex.py
│   ├── 02_segment_and_type.py
│   └── channels.yaml              # Panel → channel-index map
├── R/
│   └── 03_spatial_analysis.R
└── environment.yml                # conda / mamba environment
```

## Quick start

```bash
# 1. Create environment
mamba env create -f environment.yml
mamba activate vme-spectre

# 2. Preprocess one cycle
python scripts/01_preprocess_spectreplex.py \
    --stained-dir  /data/A7005/cycle01/stained \
    --bleach-dir   /data/A7005/cycle01/bleach \
    --output-dir   /data/A7005/processed \
    --sample-id    A7005_colon_s3 \
    --group        LAEA

# 3. Stitch (via McMicro; see https://mcmicro.org)
nextflow run labsyspharm/mcmicro \
    --in /data/A7005/processed/A7005_colon_s3 \
    --start-at registration

# 4. Segment + type
python scripts/02_segment_and_type.py \
    --stitched /data/A7005/processed/A7005_colon_s3/stitched.ome.tif \
    --channels scripts/channels.yaml \
    --sample-id A7005_colon_s3 \
    --group LAEA \
    --output-dir /data/A7005/celltables

# 5. Downstream spatial analysis (all samples together)
Rscript R/03_spatial_analysis.R \
    --celltable-dir /data/all_celltables \
    --groups LAEA,EAEA \
    --output-dir    /data/vme_spatial_out
```

## Key methodological changes vs. Anderson et al.

| Step | SPECTRE-Plex original | VME adaptation |
| --- | --- | --- |
| Tissue | FFPE human pediatric duodenum | Cryo (OCT) rhesus macaque colon / jejunum |
| Comparison | Healthy vs. active celiac | EAEA (transient) vs. LAEA (persistent) reservoir |
| Key channels | Na/K-ATPase, Lactase, CHGA, PCNA, SMA | Keeps boundary set; adds SIV Gag, p-eIF2α, FoxP3, CD117, BSG, CA2/CA12, ATF4 |
| Spatial classes | Cell-type compositional analysis | Adds `SIV_positive` / `SIV_neighbor` / `SIV_negative` three-tier classification mirroring Visium |
| Neighborhood analysis | pEGFR+ tuft cells | SIV Gag+ cells, with Treg / mast / ILC enrichment tests |
| Statistics | Unpaired two-tailed t-test | Same (preserved for comparability) |

## Attribution

- **SPECTRE-Plex** (imaging system, BaSiC flat-fielding, PyStackReg subtraction, Cellpose cyto3 segmentation, HDBSCAN/DBSCAN spatial pipeline): Anderson MD, Plone A, La J, Wong M, Raghunathan K, Silvester JA, Thiagarajah JR. *Communications Biology* 8:636 (2025). doi:10.1038/s42003-025-08052-5. Code: [mdanderson03/SpectrePlex](https://github.com/mdanderson03/SpectrePlex).
- **VME framework** (SIV Gag+, p-eIF2α ISR, Treg-centered CCI networks, BSG/CA2/CA12/ATF4 as VME markers): Hope TJ, Crentsil EU, et al. *A Tissue Virus Microenvironment with Activated Stress Responses Underlies Durable SIV Persistence* (manuscript).

## Dependencies

Python ≥3.10: `numpy`, `scipy`, `scikit-image`, `opencv-python`, `tifffile`, `pandas`, `pyyaml`, `basicpy`, `pystackreg`, `cellpose>=3.0.7`.

R ≥4.2: `readr`, `ggplot2`, `reshape2`, `tidyverse`, `hrbrthemes`, `viridis`, `readxl`, `janitor`, `dplyr`, `corrplot`, `dbscan`, `umap`, `nabor`, `stringr`, `optparse`.

External: McMicro/Ashlar (via Nextflow), ideally GPU for Cellpose on large sections.
