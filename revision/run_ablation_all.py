#!/usr/bin/env python3
"""Comprehensive ablation experiment for STED revision across multiple datasets.

Datasets: PBMC, Brain, Jejunum
Mode: Unsupervised (anchor_words = [])
Conditions: Full STED, No-Topic, No-GAS

Uncertainty estimation:
- CorEx with n_ensemble=5 produces deterministic results across seeds,
  making seed-based std meaningless (always 0).
- Instead, we use bootstrap resampling of cell types (N_BOOT=1000) to
  estimate the uncertainty of PCC and RMSE, which reflects the stability
  of the prediction across the composition of cell-type proportions.
- Each method is run once (deterministic), then bootstrap CI is computed.
"""

import os
import sys

import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import anndata as ad

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "STED_revision_code"))

from STED.Preprocessing import scPreProcessing, gsPreProcessing
from STED.scTopic import scTopic
from STED.epiDecon import epiDecon

OUT_DIR = os.path.join(PROJECT_ROOT, "ablation_output_all")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1
N_BOOT = 1000
RNG = np.random.default_rng(42)

DATASETS = {
    "PBMC": {
        "dir": "PBMC",
        "sc_count": "10k_PBMC_Multiome_RNA_counts.h5ad",
        "sc_anno": "MainCelltype.txt",
        "peak_count": "peaks_counts.h5ad",
        "gs_count": "peaks_GAS.txt",
        "gt": "MainCelltype.txt",
        "gene2peak": "gene2peak_project_mat.h5ad",
        "gene_use": "use_gene.csv",
        "ntopics": 15
    },
    "Brain": {
        "dir": "Brain",
        "sc_count": "M_Brain_count_scRNA.h5ad",
        "sc_anno": "Celltype.txt",
        "peak_count": "peaks_counts_filtered.h5ad",
        "gs_count": "peaks_geneScore_filtered.txt",
        "gt": "groud_truth.txt",
        "gene2peak": "gene2peak_project_mat.h5ad",
        "gene_use": "use_gene.csv",
        "ntopics": 20
    },
    "Jejunum": {
        "dir": "Jejunum",
        "sc_count": "scRNA_rawcounts.h5ad",
        "sc_anno": "MainCelltype.txt",
        "peak_count": "peaks_counts_filtered.h5ad",
        "gs_count": "231124_peaks_geneScore_filtered.txt",
        "gt": "Main_groud_truth.txt",
        "gene2peak": "Jejunum_filtered_peak2gene_score_mat.h5ad",
        "gene_use": "use_gene.csv",
        "ntopics": 20
    }
}


def get_ground_truth(dataset_name, config):
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])
    gt_file = os.path.join(data_dir, config["gt"])

    if dataset_name == "PBMC":
        adata = ad.read_h5ad(os.path.join(data_dir, config["sc_count"]))
        adata.obs_names_make_unique()
        ct_counts = adata.obs["main_cell_type"].value_counts()
        proportions = ct_counts / ct_counts.sum()
        return pd.DataFrame({0: proportions})
    else:
        gt = pd.read_table(gt_file, sep="\t", index_col=0)
        if gt.shape[1] > 1:
            return gt.iloc[:, [0]]
        return gt


def build_E_ref(dataset_name, config, gene_set=None, log_transform=True):
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])
    sc_file = os.path.join(data_dir, config["sc_count"])
    adata = ad.read_h5ad(sc_file)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    ct_col = "main_cell_type" if "main_cell_type" in adata.obs.columns else "Celltype"
    if ct_col not in adata.obs.columns:
        anno = pd.read_table(os.path.join(data_dir, config["sc_anno"]), index_col=0, header=None)
        anno = anno[~anno.index.duplicated(keep='first')]
        shared_cells = list(set(adata.obs_names) & set(anno.index))
        adata = adata[shared_cells].copy()
        adata.obs[ct_col] = anno.loc[adata.obs_names, 1]

    if gene_set is not None:
        use_genes = list(set(adata.var_names) & set(gene_set))
        adata = adata[:, use_genes].copy()

    cell_types = adata.obs[ct_col].unique()
    E_ref_dict = {}
    for ct in cell_types:
        ct_cells = adata.obs_names[adata.obs[ct_col] == ct]
        ct_mat = adata[ct_cells].X
        if hasattr(ct_mat, "toarray"): ct_mat = ct_mat.toarray()
        raw_sum = np.array(ct_mat.sum(axis=0)).flatten()
        total = raw_sum.sum()
        s_c = total / 1e4 if total > 0 else 1.0
        normalized = raw_sum / s_c
        E_ref_dict[ct] = np.log2(normalized + 1) if log_transform else normalized

    E_ref = pd.DataFrame.from_dict(E_ref_dict, orient="index")
    E_ref.columns = adata.var_names
    return E_ref


def load_gas(dataset_name, config, log_transform=True):
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])
    gas_file = os.path.join(data_dir, config["gs_count"])
    gas_df = pd.read_csv(gas_file, sep="\t", index_col=0)

    count_per_sample = gas_df.sum(axis=0).values.flatten()
    scale_factor = np.round(np.quantile(count_per_sample, 0.75) / 1000, 0) * 1000
    if scale_factor == 0: scale_factor = 1.0

    gas_normed = gas_df.multiply(scale_factor / count_per_sample, axis=1)
    if log_transform: gas_normed = np.log2(gas_normed + 1)
    return gas_normed


def compute_metrics(pred_vec, gt_vec):
    """Compute PCC and RMSE between two aligned vectors."""
    if np.std(pred_vec) == 0 or np.std(gt_vec) == 0 or len(pred_vec) < 3:
        return None, None
    pcc, _ = pearsonr(pred_vec, gt_vec)
    rmse = np.sqrt(mean_squared_error(gt_vec, pred_vec))
    return pcc, rmse


def evaluate_with_bootstrap(pred_df, gt_df, n_boot=N_BOOT):
    """Evaluate predictions against ground truth with bootstrap CI.

    For each bootstrap iteration, resample cell types with replacement,
    compute PCC and RMSE on the resampled subset. Returns point estimates
    and bootstrap standard deviations.
    """
    if "Bulk" in pred_df.index or "Sample_0" in pred_df.index:
        pred_df = pred_df.T

    shared = list(set(pred_df.index) & set(gt_df.index))
    if not shared:
        print(f"      Warning: No shared cell types between prediction "
              f"({list(pred_df.index)[:3]}) and GT ({list(gt_df.index)[:3]})")
        return {"pcc": np.nan, "pcc_std": np.nan, "rmse": np.nan, "rmse_std": np.nan}

    gt = gt_df.loc[shared].iloc[:, 0].values
    n_ct = len(shared)

    # Point estimate using all cell types
    if pred_df.shape[1] == 1:
        pred = pred_df.loc[shared].iloc[:, 0].values
    else:
        pred = pred_df.loc[shared].mean(axis=1).values

    pcc_point, rmse_point = compute_metrics(pred, gt)
    if pcc_point is None:
        return {"pcc": np.nan, "pcc_std": np.nan, "rmse": np.nan, "rmse_std": np.nan}

    # Bootstrap: resample cell types with replacement
    boot_pccs = []
    boot_rmses = []
    for _ in range(n_boot):
        idx = RNG.choice(n_ct, size=n_ct, replace=True)
        pred_b = pred[idx]
        gt_b = gt[idx]
        p, r = compute_metrics(pred_b, gt_b)
        if p is not None:
            boot_pccs.append(p)
            boot_rmses.append(r)

    return {
        "pcc": pcc_point,
        "pcc_std": np.std(boot_pccs) if boot_pccs else np.nan,
        "rmse": rmse_point,
        "rmse_std": np.std(boot_rmses) if boot_rmses else np.nan,
    }


def run_no_topic_baseline(dataset_name, config, gt_df, gene_set=None):
    """No-Topic baseline: direct NNLS projection of GAS onto E_ref."""
    E_ref = build_E_ref(dataset_name, config, gene_set=gene_set)
    gas_df = load_gas(dataset_name, config)
    shared_genes = list(set(E_ref.columns) & set(gas_df.index))
    E_ref_aligned = E_ref[shared_genes].values
    gas_aligned = gas_df.loc[shared_genes].values

    A = E_ref_aligned.T
    proportions = np.zeros((E_ref_aligned.shape[0], gas_aligned.shape[1]))
    for j in range(gas_aligned.shape[1]):
        x, _ = nnls(A, gas_aligned[:, j])
        if x.sum() > 0: x = x / x.sum()
        proportions[:, j] = x

    no_topic_pred = pd.DataFrame(proportions, index=E_ref.index)
    return evaluate_with_bootstrap(no_topic_pred, gt_df)


def run_no_gas_baseline(dataset_name, config, seed_dir, gt_df):
    """No-GAS baseline: NNLS in peak space using topic model outputs."""
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])

    W_gp = ad.read_h5ad(os.path.join(data_dir, config["gene2peak"]))
    W_gp.obs_names_make_unique()
    W_gp_mat = W_gp.X
    if hasattr(W_gp_mat, "toarray"): W_gp_mat = W_gp_mat.toarray()

    model_dir = os.path.join(seed_dir, "model")
    topic_celltype = pd.read_csv(os.path.join(model_dir, "topic_celltype_mat.txt"), sep="\t", index_col=0)
    gene_topic = pd.read_csv(os.path.join(model_dir, "gene_topic_mat.txt"), sep="\t", index_col=0)

    shared_genes = list(set(W_gp.obs_names) & set(gene_topic.index))
    W_gp_sub = W_gp_mat[W_gp.obs_names.get_indexer(shared_genes), :]
    gene_topic_sub = gene_topic.loc[shared_genes].values
    peak_topic = (W_gp_sub.T @ gene_topic_sub).T

    peak_adata = ad.read_h5ad(os.path.join(data_dir, config["peak_count"]))
    peak_counts = peak_adata.X
    if hasattr(peak_counts, "toarray"): peak_counts = peak_counts.toarray()
    sample_sums = np.array(peak_counts.sum(axis=1)).flatten().reshape(-1, 1)
    sample_sums[sample_sums == 0] = 1.0
    peak_norm = (peak_counts / (sample_sums / 1e4)).T
    peak_norm = np.log2(peak_norm + 1)

    ref_peak_space = peak_topic.T @ topic_celltype.values
    A = ref_peak_space
    target = peak_norm
    props = np.zeros((ref_peak_space.shape[1], target.shape[1]))
    for j in range(target.shape[1]):
        x, _ = nnls(A, target[:, j])
        if x.sum() > 0: x = x / x.sum()
        props[:, j] = x

    no_gas_pred = pd.DataFrame(props, index=topic_celltype.columns)
    return evaluate_with_bootstrap(no_gas_pred, gt_df)


def run_experiment(dataset_name):
    print(f"\n>>> Processing Dataset: {dataset_name}")
    config = DATASETS[dataset_name]
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])
    ds_out_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(ds_out_dir, exist_ok=True)

    gt_df = get_ground_truth(dataset_name, config)
    results = []

    # Full STED (single run — CorEx is deterministic with n_ensemble=5)
    seed_dir = os.path.join(ds_out_dir, f"seed_{SEED}")
    os.makedirs(seed_dir, exist_ok=True)

    print(f"  Full STED (seed={SEED})...")
    scp = scPreProcessing()
    scp.set_data(
        sc_count_file=os.path.join(data_dir, config["sc_count"]),
        sc_anno_file=os.path.join(data_dir, config["sc_anno"])
    )
    scp.cell_selection()
    gene_use_file = os.path.join(data_dir, config["gene_use"])
    gene_list = pd.read_csv(gene_use_file, index_col=0).iloc[:, 0].to_list()
    sc_anno_full_path = os.path.join(data_dir, config["sc_anno"])
    scp.geneWords(sc_anno_full_path, gene_use=gene_list)
    scp.preprocessing(linear2log=True)
    scp.preForGuide(anchored_genes=[])

    sct = scTopic()
    sct.setData(scp, out_dir=seed_dir, ntopics_list=[config["ntopics"]])
    sct.trainCorEx(seed=SEED, n_iter=200, ntopics=config["ntopics"],
                   tc_plot=False, benchmark=False, n_ensemble=5)

    epid = gsPreProcessing()
    epid.set_data(
        Epi=True,
        peak_file=os.path.join(data_dir, config["peak_count"]),
        gs_count_file=os.path.join(data_dir, config["gs_count"])
    )
    epid.gene_selection(scp.use_genes)
    epid.preprocessing(linear2log=True)
    scp.geneShared(epid.gs_genes)

    epi = epiDecon()
    epi.SetData(seed_dir, scp, epid)
    epi.Decon(ntopics_selection=config["ntopics"], seed_selection=SEED,
              model=sct, model_selection="CorEx")
    epi.Bayes()
    epi.celltype_frac_df.to_csv(os.path.join(seed_dir, "celltype_frac_df.tsv"), sep="\t")

    full_metrics = evaluate_with_bootstrap(epi.celltype_frac_df, gt_df)
    results.append({"Dataset": dataset_name, "Method": "Full_STED", **full_metrics})
    print(f"    PCC={full_metrics['pcc']:.4f} ± {full_metrics['pcc_std']:.4f}, "
          f"RMSE={full_metrics['rmse']:.4f} ± {full_metrics['rmse_std']:.4f}")

    # No-GAS Baseline
    if config["gene2peak"]:
        print(f"  No-GAS baseline...")
        no_gas_metrics = run_no_gas_baseline(dataset_name, config, seed_dir, gt_df)
        results.append({"Dataset": dataset_name, "Method": "No_GAS", **no_gas_metrics})
        print(f"    PCC={no_gas_metrics['pcc']:.4f} ± {no_gas_metrics['pcc_std']:.4f}, "
              f"RMSE={no_gas_metrics['rmse']:.4f} ± {no_gas_metrics['rmse_std']:.4f}")

    # No-Topic Baseline (deterministic)
    print(f"  No-Topic baseline...")
    no_topic_metrics = run_no_topic_baseline(dataset_name, config, gt_df, gene_set=scp.use_genes)
    results.append({"Dataset": dataset_name, "Method": "No_Topic", **no_topic_metrics})
    print(f"    PCC={no_topic_metrics['pcc']:.4f} ± {no_topic_metrics['pcc_std']:.4f}, "
          f"RMSE={no_topic_metrics['rmse']:.4f} ± {no_topic_metrics['rmse_std']:.4f}")

    return results


if __name__ == "__main__":
    all_results = []
    for ds in DATASETS:
        try:
            ds_res = run_experiment(ds)
            all_results.extend(ds_res)
        except Exception as e:
            print(f"Error processing {ds}: {e}")
            import traceback
            traceback.print_exc()

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUT_DIR, "ablation_results_all_datasets.csv"), index=False)

    print("\n" + "=" * 70)
    print("Summary Results (Point Estimate ± Bootstrap Std):")
    print("=" * 70)
    for _, row in final_df.iterrows():
        print(f"  {row['Dataset']:10s} | {row['Method']:10s} | "
              f"PCC={row['pcc']:.4f} ± {row['pcc_std']:.4f} | "
              f"RMSE={row['rmse']:.4f} ± {row['rmse_std']:.4f}")

    final_df.to_csv(os.path.join(OUT_DIR, "ablation_summary.csv"), index=False)