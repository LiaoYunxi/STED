"""Ablation baselines for STED.

Provides two baselines that strip components from the full STED pipeline
to quantify their individual contributions:

1. DirectErefBaseline: skips the topic layer entirely, projecting bulk GAS
   profiles directly against the cell-type reference matrix E_ref using
   non-negative least squares (NNLS).

2. NoGASBaseline: skips the GAS cross-modal bridge, using raw peak counts
   as features for deconvolution via the topic--cell-type matrix.

These baselines answer Reviewer 2 Major Comment 1:
- What does the topic layer add beyond direct reference projection?
- What does the GAS cross-modal bridge contribute?
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import nnls


class DirectErefBaseline:
    """No-topic baseline: project bulk GAS directly against E_ref via NNLS.

    This baseline bypasses the topic space entirely.  It constructs E_ref
    (cell-type x gene, library-size normalised) and solves for cell-type
    proportions by fitting each bulk GAS sample as a non-negative linear
    combination of the E_ref rows.

    Parameters
    ----------
    epi : epiDecon
        A configured epiDecon instance (after SetData, with GAS already
        computed).

    Attributes
    ----------
    celltype_frac_df : pd.DataFrame
        Estimated cell-type proportions (cell-types x samples).
    """

    def __init__(self, epi):
        self.epi = epi
        self.celltype_frac_df: Optional[pd.DataFrame] = None

    def fit(self, sc_count_file: str) -> "DirectErefBaseline":
        """Compute proportions by solving A * x = b via NNLS.

        Parameters
        ----------
        sc_count_file : str
            Path to the h5ad file with single-cell counts for building E_ref.

        Returns
        -------
        self
        """
        # Build E_ref (cell-types x genes) using the same method as STED
        E_ref = self.epi._build_E_ref(sc_count_file)

        # Get the bulk GAS matrix (genes x samples)
        # epiDecon stores GAS in self.topic_sample_mat after Decon(),
        # but for this baseline we need the raw GAS profiles.
        # We reconstruct from the stored attributes.
        if not hasattr(self.epi, "gs_genes") or self.epi.gs_genes is None:
            raise RuntimeError(
                "epiDecon instance must have gs_genes set. "
                "Call SetData and compute GAS first."
            )

        # Retrieve the bulk GAS matrix from epiDecon internals
        # After Decon(), epiDecon.topic_sample_mat contains the
        # topic-sample distribution, not the raw GAS.
        # We need to recompute GAS or access it from the stored file.
        gas_file = os.path.join(self.epi.out_dir, "gas_matrix.h5ad")
        if os.path.exists(gas_file):
            import anndata as ad
            gas_adata = ad.read_h5ad(gas_file)
            gas_df = pd.DataFrame(
                gas_adata.X.toarray() if hasattr(gas_adata.X, "toarray") else gas_adata.X,
                index=gas_adata.obs_names,
                columns=gas_adata.var_names,
            )
        else:
            raise RuntimeError(
                f"GAS matrix file not found at {gas_file}. "
                "Run the full STED pipeline first to generate it."
            )

        # Align genes between E_ref and GAS
        shared_genes = list(set(E_ref.columns) & set(gas_df.index))
        if len(shared_genes) == 0:
            raise ValueError("No shared genes between E_ref and GAS matrix")

        E_ref_aligned = E_ref[shared_genes].values  # (C x G)
        gas_aligned = gas_df.loc[shared_genes].values  # (G x S)

        # Solve NNLS for each sample: E_ref.T @ proportions = GAS
        # E_ref_aligned is (C x G), gas_aligned is (G x S)
        # We solve: E_ref_aligned.T @ x = gas_aligned[:, j] for each j
        A = E_ref_aligned.T  # (G x C)
        n_samples = gas_aligned.shape[1]
        n_celltypes = E_ref_aligned.shape[0]

        proportions = np.zeros((n_celltypes, n_samples))
        for j in range(n_samples):
            x, _ = nnls(A, gas_aligned[:, j])
            # Normalise to sum to 1
            if x.sum() > 0:
                x = x / x.sum()
            proportions[:, j] = x

        self.celltype_frac_df = pd.DataFrame(
            proportions,
            index=E_ref.index,
            columns=gas_df.columns if gas_df.columns is not None else range(n_samples),
        )
        return self

    def predict_peaks(self, sc_count_file, object_gs) -> pd.DataFrame:
        """Reconstruct cell-type-specific peak signals using direct E_ref.

        Parameters
        ----------
        sc_count_file : str
            Path to the h5ad file with single-cell counts.
        object_gs : gsPreProcessing
            Bulk genomic preprocessing object.

        Returns
        -------
        pd.DataFrame
            Cell-type x peak reconstructed signal matrix.
        """
        from .Genescore import Gene2Peaks
        import anndata as ad

        E_ref = self.epi._build_E_ref(sc_count_file)

        # Build gene-peak weight matrix
        peak_meta = self.epi.peak_meta
        gene_anno_file = object_gs.gene_anno_file
        G2P = Gene2Peaks(gene_anno_file=gene_anno_file, peaks=peak_meta, cutoff_weight=0)
        _, mat_GP = G2P.gwt_gene_peak_match()

        # Align genes
        shared_genes = list(set(E_ref.columns) & set(mat_GP.obs_names))
        E_ref_aligned = E_ref[shared_genes]
        idx = [mat_GP.obs_names.to_list().index(g) for g in shared_genes]
        mat_GP_aligned = mat_GP[idx, :]

        # S_cp = E_ref @ W_gp
        W_gp = mat_GP_aligned.X.toarray() if hasattr(mat_GP_aligned.X, "toarray") else mat_GP_aligned.X
        S_cp = E_ref_aligned.values @ W_gp

        return pd.DataFrame(S_cp, index=E_ref.index, columns=mat_GP_aligned.var_names)


class NoGASBaseline:
    """No-GAS baseline: use raw peak counts directly for deconvolution.

    IMPORTANT
    ---------
    This implementation is intentionally strict: it does NOT read
    ``gas_matrix.h5ad`` produced by the main STED pipeline. The caller must
    provide an external peak-topic matrix (peaks x topics), so the baseline
    remains methodologically independent from GAS.

    This baseline skips the GAS cross-modal bridge.  Instead of converting
    peaks to gene-level scores, it uses the raw peak counts as features
    and applies the topic--cell-type matrix for deconvolution.

    Parameters
    ----------
    epi : epiDecon
        A configured epiDecon instance (after SetData).

    Attributes
    ----------
    celltype_frac_df : pd.DataFrame
        Estimated cell-type proportions (cell-types x samples).
    """

    def __init__(self, epi):
        self.epi = epi
        self.celltype_frac_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        peak_count_file: str,
        peak_topic_matrix_file: str,
        topic_axis: str = "columns",
    ) -> "NoGASBaseline":
        """Estimate proportions using raw peak counts and an external peak-topic matrix.

        Parameters
        ----------
        peak_count_file : str
            Path to h5ad file with bulk peak counts (peaks x samples).
        peak_topic_matrix_file : str
            Path to a tab-delimited matrix with peak-topic weights.
            Accepted shapes:
            - peaks x topics (topic_axis="columns")
            - topics x peaks (topic_axis="rows")
        topic_axis : {"columns", "rows"}
            Orientation of topics in ``peak_topic_matrix_file``.

        Returns
        -------
        self
        """
        import anndata as ad

        # Read topic-celltype matrix
        topic_celltype_file = os.path.join(self.epi.model_dir, "topic_celltype_mat.txt")
        if not os.path.exists(topic_celltype_file):
            raise FileNotFoundError(f"topic_celltype_mat.txt not found in {self.epi.model_dir}")
        topic_celltype_df = pd.read_table(topic_celltype_file, sep="\t", index_col=0)

        # Read bulk peak counts
        bulk_adata = ad.read_h5ad(peak_count_file)
        bulk_peaks = pd.DataFrame(
            bulk_adata.X.toarray() if hasattr(bulk_adata.X, "toarray") else bulk_adata.X,
            index=bulk_adata.var_names,
            columns=bulk_adata.obs_names,
        )

        # Strict No-GAS: use only external peak-topic matrix
        if not os.path.exists(peak_topic_matrix_file):
            raise FileNotFoundError(
                f"peak_topic_matrix_file not found: {peak_topic_matrix_file}. "
                "NoGASBaseline requires an external peak-topic matrix and does not use gas_matrix.h5ad."
            )

        peak_topic_df = pd.read_table(peak_topic_matrix_file, sep="\t", index_col=0)

        if topic_axis not in {"columns", "rows"}:
            raise ValueError("topic_axis must be either 'columns' or 'rows'")

        # Convert to (peaks x topics)
        if topic_axis == "rows":
            peak_topic_df = peak_topic_df.transpose()

        # Align peaks deterministically using bulk peak order
        shared_peaks = [p for p in bulk_peaks.index if p in peak_topic_df.index]
        if len(shared_peaks) == 0:
            raise ValueError("No shared peaks between bulk data and external peak-topic matrix")

        bulk_sub = bulk_peaks.loc[shared_peaks].values  # (P x S)
        peak_topic_sub = peak_topic_df.loc[shared_peaks].values  # (P x K)

        # Solve for topic composition: peak_topic_sub @ theta = bulk_sub
        A = peak_topic_sub  # (P x K)
        n_samples = bulk_sub.shape[1]
        n_topics = peak_topic_sub.shape[1]

        theta = np.zeros((n_topics, n_samples), dtype=float)
        for j in range(n_samples):
            x, _ = nnls(A, bulk_sub[:, j])
            s = x.sum()
            if s > 0:
                x = x / s
            theta[:, j] = x

        # Harmonize topic-celltype orientation and topic labels
        topic_names = list(peak_topic_df.columns)

        # topic_celltype expected: celltypes x topics
        if set(topic_names).issubset(set(topic_celltype_df.columns)):
            ctm = topic_celltype_df.loc[:, topic_names]
        elif set(topic_names).issubset(set(topic_celltype_df.index)):
            ctm = topic_celltype_df.loc[topic_names, :].transpose()
        else:
            # fallback by shape if labels are unavailable
            if topic_celltype_df.shape[1] == n_topics:
                ctm = topic_celltype_df
            elif topic_celltype_df.shape[0] == n_topics:
                ctm = topic_celltype_df.transpose()
            else:
                raise ValueError(
                    "Cannot align topic_celltype matrix with inferred topic composition. "
                    f"n_topics={n_topics}, topic_celltype shape={topic_celltype_df.shape}"
                )

        if ctm.shape[1] != n_topics:
            raise ValueError(
                "After alignment, topic_celltype matrix has incompatible number of topics: "
                f"{ctm.shape[1]} vs {n_topics}"
            )

        # Convert topic composition to cell-type proportions
        # pi = C @ theta  (celltypes x topics) @ (topics x samples)
        celltype_frac = ctm.values @ theta  # (N x S)

        # Column-normalize safely
        col_sums = celltype_frac.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        celltype_frac = celltype_frac / col_sums

        self.celltype_frac_df = pd.DataFrame(
            celltype_frac,
            index=ctm.index,
            columns=bulk_peaks.columns if bulk_peaks.columns is not None else range(n_samples),
        )
        return self



def evaluate_deconvolution(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate deconvolution accuracy.

    Parameters
    ----------
    predicted : pd.DataFrame
        Predicted cell-type proportions (cell-types x samples).
    ground_truth : pd.DataFrame
        Ground-truth cell-type proportions (cell-types x samples).

    Returns
    -------
    dict
        Keys: 'pcc' (Pearson correlation), 'rmse' (root mean square error).
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    # Align cell types
    shared = list(set(predicted.index) & set(ground_truth.index))
    if len(shared) == 0:
        raise ValueError("No shared cell types between predicted and ground truth")

    pred = predicted.loc[shared].values.flatten()
    gt = ground_truth.loc[shared].values.flatten()

    pcc, _ = pearsonr(pred, gt)
    rmse = np.sqrt(mean_squared_error(gt, pred))

    return {"pcc": pcc, "rmse": rmse}


def evaluate_signal_reconstruction(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate signal reconstruction accuracy.

    Parameters
    ----------
    predicted : pd.DataFrame
        Predicted peak signals (cell-types x peaks or flattened).
    ground_truth : pd.DataFrame
        Ground-truth peak signals.

    Returns
    -------
    dict
        Keys: 'pcc', 'rmse'.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    pred = predicted.values.flatten()
    gt = ground_truth.values.flatten()

    pcc, _ = pearsonr(pred, gt)
    rmse = np.sqrt(mean_squared_error(gt, pred))

    return {"pcc": pcc, "rmse": rmse}