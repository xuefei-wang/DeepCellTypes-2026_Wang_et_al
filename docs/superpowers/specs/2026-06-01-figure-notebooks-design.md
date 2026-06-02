# Design: Per-theme reproduction notebooks from the research workspace

**Date:** 2026-06-01
**Repo:** `DeepCellTypes-2024_Wang_et_al` (branch `new-figures`)
**Source:** `~/Projects/deepcell-types-research-workspace`

## Goal

Replace this repo's two notebooks (`results.ipynb`, `data_statistics.ipynb`)
with a **per-theme set of self-contained notebooks** that reproduce the
research workspace's v10 paper figures, each with **embedded outputs**,
driven by derived data copied into `data/` (plus the local zarr archives for
the image panels).

The audience is the public figure-reproduction repo for Wang et al. (2024):
a reader should open a theme notebook and see exactly how each figure was
produced, with the rendered figure embedded inline.

## Context

- **This repo** is the clean, public-facing reproduction repo. Today it has
  two notebooks built from small derived files in `data/` (CSV/npz/YAML),
  some checked in, larger ones pulled from a public S3 bucket
  (`deepcelltypes-2024-wang-et-al`). Plotting is plain matplotlib/seaborn,
  Science style. `requirements.txt` is currently empty.
- **The workspace** is the full research companion: ~50 `analysis/plot_*.py`
  click CLIs producing 83 registered panels (`figures/figures.yaml`), sharing
  helpers `analysis/style.py`, `celltype_colors.py`, `_score_csv.py`,
  `hierarchical_accuracy.py`, `ct_abstention_iqr.py`. The headline
  hierarchical-eval + IQR-fence abstention numbers come from
  `deepcell_types.training.{config,utils,abstention}`.
- **Local data availability** (the original research box): nearly every
  *derived* input (`output/*.csv|*.npz|*.json`, `embeddings/`, `features/`,
  `splits/`, `config/`) exists, **and** the full archives are present:
  `/data/xwang3/expanded-tissuenet.zarr`, `/data/xwang3/gold_standard.zarr`.
  The true gating dependency is **GPU model inference at plot time**, not
  the zarr.

## Decisions (locked)

1. **Scope:** all panels runnable from derived data; include the heavier
   zarr-reading image panels (zarr is local, no GPU). Exclude GPU-inference
   and hand-authored diagrams (see Out of scope).
2. **Layout:** one notebook per theme.
3. **Data:** copy derived inputs into this repo's `data/` (gitignored).
4. **Conversion:** hybrid — shared helpers vendored into a
   `notebooks/dct_figures/` package; per-panel plotting inlined in cells.
5. **Existing notebooks:** supersede with the per-theme set; preserve any
   content unique to the originals (zero-shot comparison, S3 download).

## Layout

```
notebooks/
  dct_figures/                 # shared helper package (vendored from workspace)
    __init__.py
    style.py                   # Science-Advances mpl style (from analysis/style.py;
                               #   save_fig replaced by show/return)
    colors.py                  # celltype + modality + lineage color maps
    scoring.py                 # vendored pure-Python primitives: CELL_TYPE_HIERARCHY,
                               #   adjust_conf_mat_hierarchy, compute_iqr_fence,
                               #   hierarchical collapse, IQR-fence CT abstention,
                               #   per-slice macro-F1
    paths.py                   # resolves data/ + DATA_DIR / GOLD_ZARR env
                               #   (sane local defaults; documented overrides)
  data_statistics.ipynb        # data_stats theme
  classification.ipynb
  calibration.ipynb
  latent.ipynb
  marker_positivity.ipynb
  ssl.ipynb
  fov_exemplars.ipynb
  few_shot.ipynb
data/                          # gitignored; derived inputs + split JSONs + config YAMLs
requirements.txt               # populated
docs/superpowers/specs/2026-06-01-figure-notebooks-design.md   # this file
```

## Notebook → panel mapping

All panel names below are entries in the workspace `figures/figures.yaml`.

### `data_statistics.ipynb` (data_stats theme)
Reads `expanded-tissuenet.zarr` + split JSONs + `embeddings/svd_512.npz` +
cached `channel_intensity_stats.npz`. Heavy but runnable (no GPU).
- `01_celltype_abundance` … `12_channel_stats` (`plot_dataset_stats.py`)
- `13_modality_marker_coverage` … `17_marker_celltype_auc`
  (`plot_channel_stats_extra.py`), incl. `16_marker_embedding_2d`.

### `classification.ipynb`
CSV-driven (`output/final_frozen_cls_test_prediction.csv` + baselines).
- `class_support_distribution`, `confusion_matrix` (`plot_experiment_results.py`)
- `confidence_distribution`, `top_confusion_pairs`, `per_modality_f1`,
  `dann_head` (`plot_model_analysis.py`)
- `macro_f1_test` — DCT vs XGBoost/MAPS/CellSighter/plain-XGB
  (`plot_macro_f1_test.py`)
- `macro_f1_by_modality|tissue|lineage` (`macro_f1_by_slice.py`)
- `per_class_pr_scatter` (`plot_per_class_pr.py`)
- `confusion_matrix_lineage` (`plot_lineage_confusion.py`)
- **Folded in:** the original `results.ipynb` zero-shot baseline comparison
  (DCT vs XGBoost vs MAPS, static-vs-dynamic) so nothing published is lost.

### `calibration.ipynb`
- `calibration_reliability`, `calibration_confidence_by_correctness`,
  `iqr_coverage_macro_f1_pareto`, `iqr_per_class_abstention_rate`,
  `iqr_per_group_fence`, `calibration_reliability_by_lineage`
  (`plot_calibration_and_abstention.py`)
- `iqr_pareto_macro_f1`, `iqr_pareto_weighted_f1` (`plot_iqr_sweep.py`,
  from `output/iqr_sweep_frozen_cls.json`)
- `fov_abstention_exemplar` (`plot_abstention_exemplar.py`, zarr crop)

### `latent.ipynb`
- `latent_celltype|modality|tissue_final_frozen_cls_nca_tsne`
  (`plot_tsne.py`, from `cls_token_embedding_final_frozen_cls.npz` +
  `cls_token_meta_final_frozen_cls.npz`).

### `marker_positivity.ipynb`
- `mp_summary_comparison`, `mp_gold_per_dataset`, `mp_per_marker_waterfall`,
  `mp_gold_paired_bars`, `mp_gold_delta_bars` (`plot_mp_analysis.py`)
- `film_mp_decision_cd45|cd206|sma`, `film_mp_threshold_histogram`
  (`plot_film_mp_real_curves.py`)
- `mp_fov_exemplar*` (`plot_mp_fov_exemplar.py`, `gold_standard.zarr`)

### `ssl.ipynb`
- `per_class_pft_vs_frozen`, `per_class_delta_pft_vs_frozen`
  (`plot_finetune_vs_frozen.py`)
- `ssl_pretrain_schematic`, `ssl_finetune_schematic`, `ssl_metric_bars`
  (`plot_ssl_ablation.py`)
- `ssl_pretext_task` (`plot_ssl_pretext.py`)

### `fov_exemplars.ipynb` (reads local zarr, no GPU)
- `fov_examples` (`real_fov_examples.py`)
- `fov_traditional_workflow`, `fov_dct_prediction`, `fov_ground_truth`
  (`plot_workflow_comparison_real.py`)
- `attention_single_cell_examples` (`plot_attention_single_cell.py`, npz)

### `few_shot.ipynb`
- `fewshot_efficiency_keren` (`plot_fewshot_curve.py`, from
  `output/few_shot/keren/metrics.csv`). Scoring step (`score_fewshot.py`,
  GPU) is out of scope; notebook plots only from the saved metrics.

### Optional diagnostics (include if cheap)
- `xgb_tune_v10_0_progress` (`plot_xgb_tune_progress.py`, from
  `output/tuning/xgb_tune_v10_0_history.csv`) — fold into `classification.ipynb`
  or `ssl.ipynb` as a small diagnostic section.
- `schematic_thresholding` (`plot_schematic_thresholding.py`) — matplotlib
  illustration; include in `marker_positivity.ipynb` if it adds value.

## Out of scope

- **GPU inference at plot time:** `inference_time` (benchmark),
  `unannotated_*` (fresh inference on unannotated FOVs), `score_fewshot.py`.
- **Hand-authored / HTML-rendered diagrams:** all `architecture_*`,
  `schematic_hitl_workflow`, `figure_workflow_comparison`,
  `deepcell_label_interface`. These are not generated from data.

## Conversion recipe (per panel)

1. Lift the figure-building body out of the click CLI: drop `@click.command`
   decorators and arg parsing; replace `save_fig(...)` with returning the
   figure / `plt.show()` so the output embeds.
2. Repoint every input path to `data/` via `dct_figures.paths`.
3. Pull shared style/colors/scoring from `dct_figures`; keep the per-panel
   plotting inline in notebook cells (readable, faithful to existing style).
4. Notebook structure: markdown title → setup cell (imports, style, paths) →
   one section per panel (markdown caption sourced from `figures.yaml` +
   code cell + embedded figure output).

## Data flow

- **Derived data (~2.2 GB):** a documented copy step (mirroring the existing
  `data/required_datasets.yaml` + public-S3 pattern) brings the ~25
  CSV/npz/json inputs + split JSONs + config YAMLs into `data/`. We prepare
  the file set + an updated `required_datasets.yaml` manifest; the actual S3
  upload is the maintainer's step (cannot upload here).
- **Zarr image panels:** read `/data/xwang3/expanded-tissuenet.zarr` and
  `/data/xwang3/gold_standard.zarr` via `DATA_DIR` / `GOLD_ZARR` env vars,
  defaulting to those local paths. A markdown note explains public users
  obtain the archive from the paper release.

## Vendoring the scoring primitives

`_score_csv.py`, `hierarchical_accuracy.py`, `ct_abstention_iqr.py` import
from `deepcell_types.training.{config,utils,abstention}`:
`CELL_TYPE_HIERARCHY`, `adjust_conf_mat_hierarchy`, `compute_iqr_fence`,
`TissueNetConfig`. These drive the headline hierarchical / IQR-abstention
numbers.

**Plan:** vendor these pure-Python primitives into
`notebooks/dct_figures/scoring.py` with provenance comments pointing at the
upstream `deepcell_types` source. **Verify they are torch-free** during
build. If any is entangled with torch, fall back to listing
`deepcell-types @ git+https://github.com/vanvalenlab/deepcell-types` as an
optional dependency in `requirements.txt` rather than vendoring.

## Execution & verification

- Execute each notebook end-to-end on this box
  (`jupyter nbconvert --to notebook --execute --inplace`) so all outputs
  embed.
- Cross-check headline numbers against the workspace README: the macro-F1
  table (DCT + IQR(k=0.2): macro_acc 91.72, macro_F1 83.84, weighted_acc
  95.34, weighted_F1 95.40) and the IQR operating point (k=0.2). A material
  mismatch means the vendored scoring diverged from upstream — fix before
  declaring done.
- All in-scope notebooks run on CPU + local zarr; no GPU required.

## Risks / open items

- **Vendoring vs. dependency** (above) — resolved during build by the
  torch-free check.
- **Supersede risk:** the original `results.ipynb` zero-shot analysis and S3
  auto-download cell are preserved (zero-shot folded into
  `classification.ipynb`; download pattern into the data manifest). The
  originals are removed only after the per-theme set executes cleanly; git
  history retains them.
- **Notebook execution time:** zarr panels (`data_statistics`,
  `fov_exemplars`, MP/abstention FOV exemplars) are slow. Acceptable; noted
  in each notebook header.

## Success criteria

- 8 theme notebooks under `notebooks/`, each executed with embedded outputs.
- `notebooks/dct_figures/` helper package importable and torch-free (or
  documented optional dep).
- All in-scope `figures.yaml` panels reproduced (CSV/npz/json + local-zarr).
- Headline macro-F1 / IQR numbers match the workspace README.
- `data/required_datasets.yaml` manifest + populated `requirements.txt`.
- Original published content (zero-shot, basic data stats) preserved.
