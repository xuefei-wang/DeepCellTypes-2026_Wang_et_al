"""Scoring helpers for the DeepCell Types reproduction notebooks.

Pure numpy / pandas / (optional) zarr. Reproduces the headline cell-type
numbers exactly as the DeepCell Types training code does:

    load prediction CSV -> per-cell argmax over the 51 CT columns ->
    optional per-(tissue, modality) IQR-fence abstention at k ->
    confusion matrix -> ``adjust_conf_mat_hierarchy(CELL_TYPE_HIERARCHY)`` ->
    has-support macro / weighted accuracy + F1.

The scoring pulls three primitives from ``deepcell_types``:
    - ``deepcell_types.training.config.CELL_TYPE_HIERARCHY``
      (re-exported from ``deepcell_types.training.hierarchy``)
    - ``deepcell_types.training.utils.adjust_conf_mat_hierarchy``
      (defined in ``deepcell_types.training.metrics``)
    - ``deepcell_types.training.abstention.compute_iqr_fence``
      (defined in ``deepcell_types.abstention``)
plus the ordered class list of 51 cell types. All four are reproduced
below; each is pure-Python (numpy / dict).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vendored from deepcell_types.training.hierarchy.CELL_TYPE_HIERARCHY
# (re-exported as deepcell_types.training.config.CELL_TYPE_HIERARCHY).
# Predictions of a child type count as correct when the GT label is the parent.
# ---------------------------------------------------------------------------
CELL_TYPE_HIERARCHY = {
    "Tcell": ["CD4T", "CD8T", "Treg", "NKT"],
    "Stromal": ["Fibroblast", "Pericyte"],
}


# ---------------------------------------------------------------------------
# Ordered class list: the 51 standardized cell types in their canonical
# training-index order (matching the DeepCell Types training code). The
# integer values are an arbitrary-but-fixed bijection; only the name->index
# mapping consistency matters for the confusion-matrix / hierarchy math.
# ---------------------------------------------------------------------------
CT2IDX = {
    "AlphaCell": 0, "Astrocyte": 1, "Basophil": 2, "Bcell": 3, "BetaCell": 4,
    "BloodVesselEndothelial": 5, "CD4T": 6, "CD8T": 7, "Club": 8,
    "CollectingDuct": 9, "Dendritic": 10, "EVT": 11, "Endocrine": 12,
    "Endothelial": 13, "Enterocyte": 14, "Eosinophil": 15, "Epithelial": 16,
    "Erythrocyte": 17, "Fibroblast": 18, "Foveolar": 19, "Glial": 20,
    "Goblet": 21, "HSEC": 22, "Hepatocyte": 23, "ICC": 24, "Keratocyte": 25,
    "Langerhans": 26, "LittoralCell": 27, "LymphaticEndothelial": 28,
    "Macrophage": 29, "Mast": 30, "Melanocyte": 31, "Mesangial": 32,
    "Microglial": 33, "Monocyte": 34, "NK": 35, "NKT": 36, "Neuron": 37,
    "Neutrophil": 38, "Paneth": 39, "Pericyte": 40, "Photoreceptor": 41,
    "Plasma": 42, "Podocyte": 43, "SmoothMuscle": 44, "Stellate": 45,
    "Stromal": 46, "Tcell": 47, "Telocyte": 48, "Treg": 49, "Tumor": 50,
}


# ---------------------------------------------------------------------------
# Vendored from deepcell_types.training.metrics.adjust_conf_mat_hierarchy
# (re-exported as deepcell_types.training.utils.adjust_conf_mat_hierarchy).
# Pure numpy — operates on an (N, N) confusion matrix + ct2idx dict.
# ---------------------------------------------------------------------------
def adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx):
    """Adjust confusion matrix so child predictions count as correct for parents.

    For each parent in ``hierarchy``, moves parent->child counts onto the
    parent diagonal.

    Args:
        conf_mat: (N, N) confusion matrix (rows=true, cols=predicted).
        hierarchy: dict parent-name -> list of child-names.
        ct2idx: dict cell-type-name -> index in ``conf_mat``.
    """
    adjusted = conf_mat.copy()
    for parent, children in hierarchy.items():
        if parent not in ct2idx:
            continue
        parent_idx = ct2idx[parent]
        for child in children:
            if child not in ct2idx:
                continue
            child_idx = ct2idx[child]
            adjusted[parent_idx, parent_idx] += adjusted[parent_idx, child_idx]
            adjusted[parent_idx, child_idx] = 0
    return adjusted


# ---------------------------------------------------------------------------
# Vendored from deepcell_types.abstention.compute_iqr_fence
# (re-exported as deepcell_types.training.abstention.compute_iqr_fence).
# Pure numpy — Tukey lower fence on a 1-D confidence array.
# ---------------------------------------------------------------------------
def compute_iqr_fence(max_softmax, k):
    """Tukey lower fence ``Q1 - k * IQR`` for a 1-D array.

    Returns ``None`` when fewer than 4 values are supplied (IQR undefined for
    tiny samples; mirrors the ``len(vals) < 4`` guard).
    """
    arr = np.asarray(max_softmax, dtype=np.float64)
    if arr.size < 4:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr)


# ---------------------------------------------------------------------------
# Dataset (tissue, modality) metadata — needed only for IQR-fence abstention,
# which groups cells per (tissue, modality) bucket. Read lazily from the zarr
# archive so importing this module stays dependency-light (no zarr at import).
# Reproduces _load_dataset_metadata from the DeepCell Types training code.
# ---------------------------------------------------------------------------
_DATASET_META_CACHE = None


def load_dataset_metadata(zarr_path):
    """dataset_name -> (tissue, modality) frame, read from the zarr group attrs."""
    global _DATASET_META_CACHE
    zarr_path = str(zarr_path)
    if _DATASET_META_CACHE is not None and _DATASET_META_CACHE[0] == zarr_path:
        return _DATASET_META_CACHE[1]
    import zarr

    root = zarr.open(zarr_path, mode="r")
    rows = []
    for key in root.group_keys():
        a = dict(root[key].attrs)
        rows.append(
            {
                "dataset_name": key,
                "tissue": a.get("tissue") or a.get("organ") or "unknown",
                "modality": a.get("modality") or "unknown",
            }
        )
    meta = pd.DataFrame(rows)
    _DATASET_META_CACHE = (zarr_path, meta)
    return meta


# ---------------------------------------------------------------------------
# Reproduces _kept_mask_for_abstention from the DeepCell Types training code
# ---------------------------------------------------------------------------
def kept_mask_for_abstention(df, ct_columns, zarr_path, k):
    """Boolean kept-mask: True for cells that pass the per-group k-fence.

    Per (tissue, modality) bucket, compute the IQR lower fence on the per-cell
    max-softmax and abstain (drop) cells below it. Buckets with < 4 cells (or a
    degenerate fence) abstain nobody.
    """
    probs = df[ct_columns].to_numpy(dtype=np.float32)
    max_p = probs.max(axis=1)
    meta = load_dataset_metadata(zarr_path)
    df2 = df.merge(meta, on="dataset_name", how="left")
    df2["tissue"] = df2["tissue"].fillna("unknown")
    df2["modality"] = df2["modality"].fillna("unknown")
    abstained = np.zeros(len(df), dtype=bool)
    groups = df2.groupby(["tissue", "modality"], sort=False, dropna=False).indices
    for _key, idx in groups.items():
        if len(idx) < 4:
            continue
        fence = compute_iqr_fence(max_p[idx], k)
        if fence is None:
            continue
        abstained[idx] = max_p[idx] < fence
    return ~abstained


# ---------------------------------------------------------------------------
# Reproduces _hier_conf_mat from the DeepCell Types training code
# ---------------------------------------------------------------------------
def hier_conf_mat(csv_path, ct2idx, abstention_k=None, zarr_path=None):
    """Hierarchy-adjusted confusion matrix + (n_total, n_kept).

    Loads the prediction CSV, optionally applies per-(tissue, modality) IQR
    abstention at ``abstention_k`` (dropping abstained cells), builds the raw
    confusion matrix over ``ct2idx``, then applies
    :func:`adjust_conf_mat_hierarchy`.
    """
    df = pd.read_csv(csv_path)
    ct_columns = [c for c in df.columns if c in ct2idx]
    n_total = len(df)
    n_kept = n_total
    if abstention_k is not None and abstention_k > 0:
        if zarr_path is None:
            raise ValueError(
                "abstention_k requires zarr_path (for (tissue, modality) "
                "grouping); none supplied."
            )
        kept = kept_mask_for_abstention(df, ct_columns, zarr_path, abstention_k)
        df = df.loc[kept].reset_index(drop=True)
        n_kept = int(kept.sum())
    probs = df[ct_columns].values
    pred_names = np.array(ct_columns)[probs.argmax(axis=1)]
    true_names = df["cell_type_actual"].values
    n = len(ct2idx)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true_names, pred_names):
        if t in ct2idx and p in ct2idx:
            cm[ct2idx[t], ct2idx[p]] += 1
    return adjust_conf_mat_hierarchy(cm, CELL_TYPE_HIERARCHY, ct2idx), n_total, n_kept


# ---------------------------------------------------------------------------
# Reproduces score_csv from the DeepCell Types training code
# ---------------------------------------------------------------------------
def score_csv(csv_path, ct2idx=None, abstention_k=None, zarr_path=None):
    """Hier-adjusted macro/weighted accuracy + F1 (all in %).

    Args:
        csv_path: prediction CSV (one row per cell; 51 CT-probability columns
            named after the keys of ``ct2idx`` plus a ``cell_type_actual``
            column).
        ct2idx: name->index mapping; defaults to :data:`CT2IDX`.
        abstention_k: if > 0, apply per-(tissue, modality) IQR-fence CT
            abstention at this k (requires ``zarr_path``). The paper headline
            uses k=0.2.
        zarr_path: archive providing per-dataset (tissue, modality); required
            iff ``abstention_k`` is set.

    Returns dict with macro_acc, macro_f1, weighted_acc, weighted_f1, n_cells,
    n_kept, coverage, n_classes_with_support, and per-class breakdowns.
    """
    if ct2idx is None:
        ct2idx = CT2IDX
    cm, n_cells, n_kept = hier_conf_mat(
        csv_path, ct2idx, abstention_k=abstention_k, zarr_path=zarr_path
    )
    support = cm.sum(axis=1)
    has_support = support > 0
    tp = np.diag(cm).astype(float)
    pred_sum = cm.sum(axis=0).astype(float)
    recall = np.zeros_like(tp)
    precision = np.zeros_like(tp)
    np.divide(tp, support, out=recall, where=support > 0)
    np.divide(tp, pred_sum, out=precision, where=pred_sum > 0)
    f1 = np.zeros_like(tp)
    denom = recall + precision
    np.divide(2 * recall * precision, denom, out=f1, where=denom > 0)

    macro_acc = recall[has_support].mean() * 100.0
    macro_f1 = f1[has_support].mean() * 100.0
    total = support[has_support].sum()
    if total > 0:
        weighted_acc = ((recall[has_support] * support[has_support]).sum()
                        / total) * 100.0
        weighted_f1 = ((f1[has_support] * support[has_support]).sum()
                       / total) * 100.0
    else:
        weighted_acc = weighted_f1 = float("nan")

    idx2ct = {v: k for k, v in ct2idx.items()}
    per_class_acc = {idx2ct[i]: float(recall[i] * 100.0)
                     for i in range(len(ct2idx)) if has_support[i]}
    per_class_f1 = {idx2ct[i]: float(f1[i] * 100.0)
                    for i in range(len(ct2idx)) if has_support[i]}
    per_class_support = {idx2ct[i]: int(support[i])
                         for i in range(len(ct2idx)) if has_support[i]}

    coverage = float(n_kept) / float(n_cells) if n_cells > 0 else 1.0
    return {
        "csv_path": str(csv_path),
        "n_cells": int(n_cells),
        "n_kept": int(n_kept),
        "coverage": coverage,
        "abstention_k": abstention_k,
        "n_classes_with_support": int(has_support.sum()),
        "macro_acc": float(macro_acc),
        "macro_f1": float(macro_f1),
        "weighted_acc": float(weighted_acc),
        "weighted_f1": float(weighted_f1),
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
    }


def score_many(paths_by_label, ct2idx=None):
    """paths_by_label: dict label -> csv_path. Returns dict label -> score dict.

    Reproduces score_many from the DeepCell Types training code (no abstention applied).
    """
    if ct2idx is None:
        ct2idx = CT2IDX
    out = {}
    for label, p in paths_by_label.items():
        if not Path(p).exists():
            print(f"  [score_csv] WARN: {p} not found, skipping {label}")
            continue
        out[label] = score_csv(p, ct2idx)
    return out
