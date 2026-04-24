#!/usr/bin/env Rscript
# 03_spatial_analysis.R
# =====================
#
# Downstream spatial analysis of SPECTRE-Plex-derived single-cell tables for
# the VME (Viral Microenvironment) gut reservoir study.
#
# Inputs
# ------
#   --celltable-dir   directory of *_celltable.csv files from 02_segment_and_type.py
#   --groups          comma-separated group names to compare (e.g. LAEA,EAEA)
#   --output-dir      where to write plots, tables, and RDS summaries
#
# What this does
# --------------
# 1. Loads all per-sample cell tables and merges into one data frame.
# 2. Point-map visualisation of cell-type spatial distribution per sample.
# 3. UMAP on normalized per-cell marker intensities.
# 4. HDBSCAN cluster-size analysis for selected cell populations (e.g. SIV+,
#    Treg, mast cells).
# 5. DBSCAN-based local point density ("busyness") with bandwidth=300 px.
# 6. KNN neighborhood composition around SIV+ cells (analogous to Fig. 2j of
#    Anderson et al. and Fig. 4 of the VME manuscript).
# 7. Between-group comparisons with unpaired two-tailed t-tests, following
#    the SPECTRE-Plex Statistics and Reproducibility section.
#
# CRAN packages used (matching Anderson et al.):
#   readr, ggplot2, reshape2, tidyverse, hrbrthemes, viridis, readxl, janitor,
#   dplyr, corrplot, dbplyr, dbscan, umap, plotly, spatialcluster, spdep,
#   geosphere, usedist, nabor, stringr, lessR
#
# Install once:
#   install.packages(c("optparse","readr","ggplot2","reshape2","tidyverse",
#     "hrbrthemes","viridis","dplyr","corrplot","dbscan","umap","nabor",
#     "stringr"))

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(viridis)
  library(umap)
  library(dbscan)
  library(nabor)
  library(stringr)
})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

opt_list <- list(
  make_option("--celltable-dir", type = "character",
              help = "Directory containing *_celltable.csv files"),
  make_option("--groups", type = "character", default = "LAEA,EAEA",
              help = "Comma-separated group names to compare [default %default]"),
  make_option("--output-dir", type = "character", default = "./vme_spatial_out",
              help = "Output directory [default %default]"),
  make_option("--hdb-minpts", type = "integer", default = 5,
              help = "HDBSCAN minPts parameter [default %default]"),
  make_option("--dbscan-bandwidth", type = "numeric", default = 300,
              help = "Bandwidth (px) for tissue busyness density [default %default]"),
  make_option("--knn-k", type = "integer", default = 10,
              help = "K for KNN neighborhood analysis around SIV+ cells [default %default]")
)

opt <- parse_args(OptionParser(option_list = opt_list))
groups <- strsplit(opt$groups, ",")[[1]]
dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Load all cell tables
# ---------------------------------------------------------------------------

files <- list.files(opt$`celltable-dir`, pattern = "_celltable\\.csv$",
                    full.names = TRUE)
if (length(files) == 0) stop("No *_celltable.csv files found.")

message(sprintf("Loading %d cell tables...", length(files)))
cells <- bind_rows(lapply(files, function(f) {
  df <- suppressMessages(read_csv(f, show_col_types = FALSE))
  df
}))

cells <- cells %>% filter(group %in% groups)
message(sprintf("Total cells: %d across %d samples",
                nrow(cells), length(unique(cells$sample_id))))

# Identify marker intensity / positivity columns
mean_cols <- grep("_mean$", names(cells), value = TRUE)
pos_cols  <- grep("_pos$",  names(cells), value = TRUE)

# ---------------------------------------------------------------------------
# Assign a dominant cell type per cell
# ---------------------------------------------------------------------------
# Priority order: SIV-infected tag is kept separate (vme_category) from lineage.
# We use a lightweight rule-based assignment that mirrors the manuscript's
# cell types. Users should extend this block to match their antibody panel.

lineage_rules <- function(row) {
  # Specific markers override general ones
  if (isTRUE(row[["FoxP3_pos"]]) && isTRUE(row[["CD4_pos"]]))     return("Treg")
  if (isTRUE(row[["CD8_pos"]]))                                   return("CD8_T")
  if (isTRUE(row[["CD4_pos"]]) && isTRUE(row[["CD3_pos"]]))       return("CD4_T")
  if (isTRUE(row[["CD3_pos"]]))                                   return("T_other")
  if (isTRUE(row[["CD117_pos"]]))                                 return("Mast")
  if (isTRUE(row[["CD68_pos"]]))                                  return("Myeloid")
  if (isTRUE(row[["CD20_pos"]]))                                  return("B_cell")
  if (isTRUE(row[["CD138_pos"]]))                                 return("Plasma")
  if (isTRUE(row[["EpCAM_pos"]]) || isTRUE(row[["PanCK_pos"]]))   return("Epithelial")
  return("Other")
}

# Only run if the lineage columns exist; otherwise leave as NA and let the
# user plug in their own rules.
needed <- c("FoxP3_pos","CD4_pos","CD8_pos","CD3_pos","CD117_pos",
            "CD68_pos","CD20_pos","CD138_pos","EpCAM_pos","PanCK_pos")
if (all(needed %in% names(cells))) {
  cells$lineage <- apply(cells[, needed, drop = FALSE], 1, function(r) {
    lineage_rules(as.list(r))
  })
} else {
  cells$lineage <- NA_character_
  warning("Lineage columns missing; skipping rule-based cell-type assignment.")
}

# ---------------------------------------------------------------------------
# 1. Point-map visualisation per sample
# ---------------------------------------------------------------------------

pointmap_dir <- file.path(opt$`output-dir`, "pointmaps")
dir.create(pointmap_dir, showWarnings = FALSE)

for (s in unique(cells$sample_id)) {
  sub <- cells %>% filter(sample_id == s)
  p <- ggplot(sub, aes(x = x, y = -y, colour = vme_category)) +
    geom_point(size = 0.3, alpha = 0.7) +
    scale_colour_manual(values = c(
      SIV_positive = "#d7191c",
      SIV_neighbor = "#fdae61",
      SIV_negative = "#2b83ba"
    )) +
    coord_equal() +
    theme_minimal(base_size = 10) +
    labs(title = sprintf("%s (%s)", s, unique(sub$group)))
  ggsave(file.path(pointmap_dir, sprintf("%s_vme_pointmap.png", s)),
         p, width = 6, height = 6, dpi = 200)
}

# ---------------------------------------------------------------------------
# 2. UMAP on normalized marker intensities
# ---------------------------------------------------------------------------

umap_input <- cells[, mean_cols, drop = FALSE] %>%
  mutate(across(everything(), ~ (. - min(.)) / max(1e-9, max(.) - min(.))))

# Subsample if > 50k cells — UMAP on raw R is slow otherwise
set.seed(42)
idx <- if (nrow(umap_input) > 50000) sample.int(nrow(umap_input), 50000) else seq_len(nrow(umap_input))

message("Running UMAP on ", length(idx), " cells...")
u <- umap::umap(as.matrix(umap_input[idx, ]), n_neighbors = 30, min_dist = 0.3)

umap_df <- data.frame(
  UMAP1 = u$layout[, 1],
  UMAP2 = u$layout[, 2],
  group = cells$group[idx],
  vme   = cells$vme_category[idx],
  lineage = cells$lineage[idx]
)

p_umap <- ggplot(umap_df, aes(UMAP1, UMAP2, colour = vme)) +
  geom_point(size = 0.2, alpha = 0.6) +
  facet_wrap(~ group) +
  theme_minimal(base_size = 10) +
  scale_colour_manual(values = c(
    SIV_positive = "#d7191c",
    SIV_neighbor = "#fdae61",
    SIV_negative = "#2b83ba"
  ))
ggsave(file.path(opt$`output-dir`, "umap_vme.png"),
       p_umap, width = 10, height = 5, dpi = 200)

# ---------------------------------------------------------------------------
# 3. HDBSCAN cluster-size analysis for SIV+ cells (per sample)
# ---------------------------------------------------------------------------

cluster_summary <- cells %>%
  filter(vme_category == "SIV_positive") %>%
  group_by(sample_id, group) %>%
  group_modify(~ {
    if (nrow(.x) < opt$`hdb-minpts`) {
      return(tibble(cluster_size = integer(0)))
    }
    h <- dbscan::hdbscan(as.matrix(.x[, c("x", "y")]),
                         minPts = opt$`hdb-minpts`)
    tibble(cluster_size = as.integer(table(h$cluster[h$cluster > 0])))
  }) %>%
  ungroup()

write_csv(cluster_summary,
          file.path(opt$`output-dir`, "siv_positive_cluster_sizes.csv"))

# ---------------------------------------------------------------------------
# 4. Busyness (DBSCAN point density, bandwidth = 300 px)
# ---------------------------------------------------------------------------

busyness <- cells %>%
  group_by(sample_id, group) %>%
  group_modify(~ {
    pts <- as.matrix(.x[, c("x", "y")])
    nn <- nabor::knn(pts, pts, k = min(50, nrow(pts)))
    # A cell's density = number of neighbours within bandwidth
    counts <- rowSums(nn$nn.dists <= opt$`dbscan-bandwidth`)
    tibble(
      mean_local_density = mean(counts),
      median_local_density = median(counts),
      n_cells = nrow(.x)
    )
  }) %>%
  ungroup()

write_csv(busyness, file.path(opt$`output-dir`, "tissue_busyness.csv"))

# ---------------------------------------------------------------------------
# 5. KNN neighborhood composition around SIV+ cells
# ---------------------------------------------------------------------------
# For each SIV+ cell, what fraction of its k nearest neighbours are each
# lineage? This mirrors the VME manuscript's observation that Treg + mast +
# ILC cluster near SIV foci in LAEA tissues.

if (!all(is.na(cells$lineage))) {
  knn_results <- cells %>%
    group_by(sample_id, group) %>%
    group_modify(~ {
      pts <- as.matrix(.x[, c("x", "y")])
      siv_idx <- which(.x$vme_category == "SIV_positive")
      if (length(siv_idx) == 0) return(tibble())
      nn <- nabor::knn(pts, pts[siv_idx, , drop = FALSE],
                       k = min(opt$`knn-k` + 1, nrow(pts)))
      # Drop self-match (first column)
      nbr_ids <- nn$nn.idx[, -1, drop = FALSE]
      lineages <- .x$lineage[as.vector(nbr_ids)]
      tibble(
        lineage = lineages,
        siv_cell_id = rep(siv_idx, each = ncol(nbr_ids))
      )
    }) %>%
    ungroup() %>%
    count(sample_id, group, lineage) %>%
    group_by(sample_id) %>%
    mutate(fraction = n / sum(n)) %>%
    ungroup()

  write_csv(knn_results,
            file.path(opt$`output-dir`, "knn_neighborhood_composition.csv"))

  # Per-lineage fraction box plot, grouped
  p_knn <- ggplot(knn_results, aes(x = group, y = fraction, fill = group)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.15, size = 1) +
    facet_wrap(~ lineage, scales = "free_y") +
    theme_minimal(base_size = 10) +
    labs(y = "Fraction of KNN around SIV+ cells",
         x = NULL,
         title = "Neighborhood composition around SIV+ cells")
  ggsave(file.path(opt$`output-dir`, "knn_neighborhood_boxplot.png"),
         p_knn, width = 10, height = 7, dpi = 200)
}

# ---------------------------------------------------------------------------
# 6. Group comparisons (unpaired two-tailed t-tests, matching SPECTRE-Plex)
# ---------------------------------------------------------------------------

group_stats <- cells %>%
  group_by(sample_id, group) %>%
  summarise(
    total_cells = n(),
    pct_SIV_positive = mean(vme_category == "SIV_positive") * 100,
    pct_SIV_neighbor = mean(vme_category == "SIV_neighbor") * 100,
    .groups = "drop"
  )

# Only run t-tests when at least two groups have >= 2 samples
safe_ttest <- function(metric) {
  tryCatch({
    vals_g1 <- group_stats[[metric]][group_stats$group == groups[1]]
    vals_g2 <- group_stats[[metric]][group_stats$group == groups[2]]
    if (length(vals_g1) < 2 || length(vals_g2) < 2) return(NA_real_)
    t.test(vals_g1, vals_g2, paired = FALSE, alternative = "two.sided")$p.value
  }, error = function(e) NA_real_)
}

ttest_results <- tibble(
  metric = c("total_cells", "pct_SIV_positive", "pct_SIV_neighbor"),
  p_value = sapply(c("total_cells", "pct_SIV_positive", "pct_SIV_neighbor"),
                   safe_ttest)
)

write_csv(group_stats, file.path(opt$`output-dir`, "per_sample_summary.csv"))
write_csv(ttest_results, file.path(opt$`output-dir`, "group_ttest_results.csv"))

saveRDS(
  list(
    cells = cells, umap = umap_df,
    clusters = cluster_summary, busyness = busyness,
    ttests = ttest_results
  ),
  file.path(opt$`output-dir`, "vme_spatial_results.rds")
)

message("Done. Outputs written to ", opt$`output-dir`)
