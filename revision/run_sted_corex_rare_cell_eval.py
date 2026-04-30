#!/usr/bin/env python3
"""STED-CorEx稀有细胞预测评估实验。

按稀有到常见排序细胞类型，用不同seed运行，计算每个细胞类型的PCC和RMSE。
"""

import os
import sys
import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "STED_revision_code"))

from STED.Preprocessing import scPreProcessing, gsPreProcessing
from STED.scTopic import scTopic
from STED.epiDecon import epiDecon

OUT_DIR = os.path.join(PROJECT_ROOT, "sted_corex_rare_cell_eval")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]
NTOPICS_LIST = [15, 20]

DATASETS = {
    "PBMC": {
        "dir": "PBMC",
        "sc_count": "10k_PBMC_Multiome_RNA_counts.h5ad",
        "sc_anno": "MainCelltype.txt",
        "peak_count": "peaks_counts.h5ad",
        "gs_count": "peaks_GAS.txt",
        "gene_use": "use_gene.csv",
        "ntopics": [15, 20],  # PBMC运行15和20
    },
    "Brain": {
        "dir": "Brain",
        "sc_count": "M_Brain_count_scRNA.h5ad",
        "sc_anno": "Celltype.txt",
        "peak_count": "peaks_counts_filtered.h5ad",
        "gs_count": "peaks_geneScore_filtered.txt",
        "gene_use": "use_gene.csv",
        "ntopics": [20],  # Brain只运行20
    },
    "Jejunum": {
        "dir": "Jejunum",
        "sc_count": "scRNA_rawcounts.h5ad",
        "sc_anno": "MainCelltype.txt",
        "peak_count": "peaks_counts_filtered.h5ad",
        "gs_count": "231124_peaks_geneScore_filtered.txt",
        "gene_use": "use_gene.csv",
        "ntopics": [20],  # Jejunum只运行20
    },
}


def get_ground_truth_proportions(dataset_name, config):
    """从单细胞数据计算真实的细胞类型比例。"""
    import anndata as ad
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])

    if dataset_name == "PBMC":
        adata = ad.read_h5ad(os.path.join(data_dir, config["sc_count"]))
        adata.obs_names_make_unique()
        ct_counts = adata.obs["main_cell_type"].value_counts()
        proportions = ct_counts / ct_counts.sum()
    else:
        anno_file = os.path.join(data_dir, config["sc_anno"])
        anno = pd.read_csv(anno_file, sep="\t", header=None, index_col=0)
        anno.columns = ["cell_type"]
        anno = anno[~anno.index.duplicated(keep='first')]
        ct_counts = anno["cell_type"].value_counts()
        proportions = ct_counts / ct_counts.sum()

    return proportions.sort_values()


def run_sted_corex(dataset_name, config, ntopics, seed):
    """运行STED-CorEx并返回预测的细胞类型比例。"""
    data_dir = os.path.join(PROJECT_ROOT, "demo", config["dir"])
    ds_out_dir = os.path.join(OUT_DIR, dataset_name, f"ntopics_{ntopics}", f"seed_{seed}")
    result_file = os.path.join(ds_out_dir, "celltype_fractions.tsv")

    # 检查是否已存在结果，跳过已完成的
    if os.path.exists(result_file):
        print(f"    [已存在，跳过] seed={seed}")
        frac_df = pd.read_csv(result_file, sep="\t", index_col=0)
        pred_frac = frac_df.loc["Bulk"]
        return pred_frac

    os.makedirs(ds_out_dir, exist_ok=True)
    os.makedirs(os.path.join(ds_out_dir, "model"), exist_ok=True)

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
    sct.setData(scp, out_dir=ds_out_dir, ntopics_list=[ntopics])
    sct.trainCorEx(seed=seed, n_iter=200, ntopics=ntopics,
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
    epi.SetData(ds_out_dir, scp, epid)
    epi.Decon(ntopics_selection=ntopics, seed_selection=seed,
              model=sct, model_selection="CorEx")
    epi.Bayes()

    # 保存结果以便后续跳过
    epi.celltype_frac_df.to_csv(result_file, sep="\t")

    pred_frac = epi.celltype_frac_df.loc["Bulk"]
    return pred_frac


def main():
    all_results = []

    for dataset_name in ["PBMC", "Brain", "Jejunum"]:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")

        config = DATASETS[dataset_name]

        # 获取真实比例（按稀有到常见排序）
        gt_sorted = get_ground_truth_proportions(dataset_name, config)
        print(f"\n真实细胞类型比例 (稀有到常见):")
        for ct, frac in gt_sorted.items():
            print(f"  {ct}: {frac:.4f}")

        # 为每个细胞类型分配稀有等级
        n_ct = len(gt_sorted)
        rare_ranks = {}
        for i, (ct, frac) in enumerate(gt_sorted.items()):
            # 稀有等级: 1=最稀有, n=最常见
            rare_ranks[ct] = i + 1

        for ntopics in config["ntopics"]:
            print(f"\n--- ntopics={ntopics} ---")

            # 收集所有seed的预测结果
            all_preds = {}

            for seed in SEEDS:
                try:
                    print(f"  Running seed={seed}...")
                    pred_frac = run_sted_corex(dataset_name, config, ntopics, seed)

                    for ct in pred_frac.index:
                        if ct not in all_preds:
                            all_preds[ct] = []
                        all_preds[ct].append(pred_frac[ct])

                except Exception as e:
                    print(f"    错误: {e}")
                    import traceback
                    traceback.print_exc()

            # 计算每个细胞类型的统计量
            print(f"\n  每个细胞类型的预测结果 (稀有到常见):")
            for ct in gt_sorted.index:
                if ct in all_preds:
                    preds = np.array(all_preds[ct])
                    gt_val = gt_sorted[ct]
                    pred_mean = np.mean(preds)
                    pred_std = np.std(preds)
                    abs_error = abs(pred_mean - gt_val)
                    rel_error = abs_error / gt_val if gt_val > 0 else np.nan

                    # 由于只有1个样本，PCC/RMSE需要跨seed计算
                    # 这里计算跨seed的变异系数
                    cv = pred_std / pred_mean if pred_mean > 0 else 0

                    all_results.append({
                        "Dataset": dataset_name,
                        "NTopics": ntopics,
                        "CellType": ct,
                        "RareRank": rare_ranks[ct],
                        "GT_Fraction": gt_val,
                        "Pred_Mean": pred_mean,
                        "Pred_Std": pred_std,
                        "Abs_Error": abs_error,
                        "Rel_Error": rel_error,
                        "CV": cv  # 变异系数
                    })

                    print(f"    {ct} (稀有级{rare_ranks[ct]}): GT={gt_val:.4f}, Pred={pred_mean:.4f}±{pred_std:.4f}, 误差={abs_error:.4f}")
                else:
                    print(f"    {ct}: 无预测结果")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(OUT_DIR, "celltype_level_evaluation.csv")
    results_df.to_csv(results_file, index=False)

    # 创建汇总表：按稀有等级分组
    summary_by_rare = results_df.groupby(["Dataset", "NTopics", "RareRank"]).agg({
        "GT_Fraction": "mean",
        "Pred_Mean": "mean",
        "Pred_Std": "mean",
        "Abs_Error": "mean",
        "Rel_Error": "mean",
        "CV": "mean"
    }).reset_index()

    summary_file = os.path.join(OUT_DIR, "celltype_level_summary_by_rare.csv")
    summary_by_rare.to_csv(summary_file, index=False)

    print(f"\n{'='*60}")
    print("结果已保存:")
    print(f"  详细结果: {results_file}")
    print(f"  按稀有等级汇总: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()