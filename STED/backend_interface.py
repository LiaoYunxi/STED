"""Shared backend export utilities for STED topic-learning modules.

This module standardizes the transfer object consumed by epiDecon:
- gene_topic: genes x topics matrix
- cell_topic: topics x cells matrix
- topic_celltype: topics x celltypes matrix
- metadata: backend identity and hierarchy semantics
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _to_df(matrix, index=None, columns=None) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        df = matrix.copy()
        if index is not None:
            df.index = index
        if columns is not None:
            df.columns = columns
        return df
    arr = np.asarray(matrix)
    return pd.DataFrame(arr, index=index, columns=columns)


def export_backend_outputs(
    model_dir: str,
    gene_topic,
    cell_topic,
    topic_celltype,
    backend: str,
    hierarchy_mode: str = "posthoc",
    outname: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(model_dir, exist_ok=True)

    gene_topic_df = _to_df(gene_topic)
    cell_topic_df = _to_df(cell_topic)
    topic_celltype_df = _to_df(topic_celltype)

    suffix = "" if outname is None else f"_{outname}"
    gene_topic_df.to_csv(os.path.join(model_dir, f"gene_topic_mat{suffix}.txt"), sep="\t")
    cell_topic_df.to_csv(os.path.join(model_dir, f"topic_cell_mat{suffix}.txt"), sep="\t")
    topic_celltype_df.to_csv(os.path.join(model_dir, f"topic_celltype_mat{suffix}.txt"), sep="\t")

    metadata = {
        "backend": backend,
        "hierarchy_mode": hierarchy_mode,
        "n_genes": int(gene_topic_df.shape[0]),
        "n_topics": int(gene_topic_df.shape[1]),
        "n_cells": int(cell_topic_df.shape[1]),
        "n_celltypes": int(topic_celltype_df.shape[1]),
    }
    pd.Series(metadata).to_json(os.path.join(model_dir, f"backend_metadata{suffix}.json"), indent=2)

    return {
        "gene_topic": gene_topic_df,
        "cell_topic": cell_topic_df,
        "topic_celltype": topic_celltype_df,
        "metadata": pd.DataFrame([metadata]),
    }
