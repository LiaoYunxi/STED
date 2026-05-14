"""
epiDecon_modified.py
"""

import os
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as ss
from scipy.sparse import csc_matrix
import pickle
import gensim
import time
import copy
import anndata as ad

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger('epiDecon_modified')

from .data import *
from .Genescore import *

#%%


def SpatialDeconvolveRaw(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    topic_celltype_file = os.path.join(model_dir, "topic_celltype_mat.txt")
    celltype_topic_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0).transpose()
    celltype_spot_array = np.dot(celltype_topic_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = safe_normalize_rows(spot_celltype_array)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_df.index
    spot_celltype_array_norm_df.index = st_count_spots
    return (spot_celltype_array_df, spot_celltype_array_norm_df)


def SpatialDeconvolveNorm(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    topic_celltype_file = os.path.join(model_dir, "topic_celltype_mat.txt")
    celltype_topic_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0).transpose()
    celltype_topic_norm_df = pd.DataFrame(safe_normalize_cols(celltype_topic_df.values),
                                          index=celltype_topic_df.index,
                                          columns=celltype_topic_df.columns)
    celltype_spot_array = np.dot(celltype_topic_norm_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = safe_normalize_rows(spot_celltype_array)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_norm_df.index = st_count_spots
    return (spot_celltype_array_df, spot_celltype_array_norm_df)


def SpatialDeconvolveNormBySD(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    topic_celltype_file = os.path.join(model_dir, "topic_celltype_mat.txt")
    celltype_topic_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0).transpose()
    celltype_topic_norm_array = StandardScaler(with_mean=False).fit_transform(celltype_topic_df)
    celltype_spot_array = np.dot(celltype_topic_norm_array[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = safe_normalize_rows(spot_celltype_array)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_df.index
    spot_celltype_array_norm_df.index = st_count_spots
    return (spot_celltype_array_df, spot_celltype_array_norm_df)


def SpatialDeconvolveBayes(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
    celltype_topic_bayes_df = pd.read_csv(celltype_topic_file, sep="\t", index_col=0)
    celltype_spot_array = np.dot(celltype_topic_bayes_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_bayes_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = safe_normalize_rows(spot_celltype_array)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_bayes_df.index
    spot_celltype_array_norm_df.index = st_count_spots
    return (spot_celltype_array_df, spot_celltype_array_norm_df)


def SpatialDeconvolveBayesNorm(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
    celltype_topic_bayes_df = pd.read_csv(celltype_topic_file, sep="\t", index_col=0)
    celltype_topic_norm_df = pd.DataFrame(safe_normalize_cols(celltype_topic_bayes_df.values),
                                          index=celltype_topic_bayes_df.index,
                                          columns=celltype_topic_bayes_df.columns)
    celltype_spot_array = np.dot(celltype_topic_norm_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = safe_normalize_rows(spot_celltype_array)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_norm_df.index = st_count_spots
    return (spot_celltype_array_df, spot_celltype_array_norm_df)

#%%


class epiDecon():
    def __init__(self):
        self.out_dir = None
        self.topic_gene_mat = None
        self.topic_cell_mat = None
        self.topic_celltype_df = None
        self.sc_celltype_prediction = None
        self.norm_selection = None
        self.ntopics_selection = None
        self.seed_selection = None
        self.outname = None

    def SetData(self, out_dir, object_sc, object_gs, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(out_dir, "model")
        self.out_dir = out_dir
        self.model_dir = model_dir
        self.sc_cells = object_sc.sc_cells
        self.ann_dict = object_sc.ann_dict

        self.target_df = object_gs.target_df
        self.gs_genes = object_gs.use_genes
        self.gs_samples = object_gs.gs_samples

        if object_gs.mm_df is None:
            self.mm_df = object_gs.target_df
        else:
            self.mm_df = object_gs.mm_df

    # =================================================================
    # [NEW] NNLS 主题空间投影 — 替代原版 corex.fit() 回退
    # =================================================================
    def _project_bulk_to_topic_nnls(self, x_bulk, gene_topic_df):
        """使用 NNLS 将 bulk 基因表达投影到预训练的主题空间。

        对每个 bulk 样本独立求解非负最小二乘问题:
            min ||G @ w - x_i||_2   subject to w >= 0
        其中 G 是 gene_topic_mat (genes × topics)，
        x_i 是第 i 个样本的基因表达向量。
        求解后将权重向量 w 归一化为概率分布（和为 1）。

        原版 epiDecon 使用 corex.fit(x_bulk) 作为回退。但 CorEx 是无监督模型，
        fit() 在 n_samples=1 时跨样本互信息恒为零，梯度无法更新任何权重，
        输出的 topic_sample_mat 退化为随机初始化的定值常数，
        与输入 x_bulk 的内容完全无关。这导致极端场景实验中不同条件
        （不同稀疏度、不同衰减因子）产生完全一致的 PCC。

        NNLS 投影利用预训练的 gene_topic_mat 将新 bulk 数据映射到已知主题空间，
        即使单样本也能正确感知输入数据的变化。

        Args:
            x_bulk: np.ndarray, shape (n_samples, n_genes),
                    已按 gene_topic_df 的基因顺序对齐
            gene_topic_df: pd.DataFrame, shape (n_genes, n_topics),
                           预训练的 gene_topic_mat

        Returns:
            scipy.sparse.csc_matrix, shape (n_topics, n_samples)
        """
        from scipy.optimize import nnls

        n_samples = x_bulk.shape[0]
        n_topics = gene_topic_df.shape[1]
        G = gene_topic_df.values  # (n_genes, n_topics)

        # 防御性处理 NaN/Inf
        if not np.isfinite(G).all():
            logger.warning("gene_topic_mat contains NaN/Inf values; replacing with zeros.")
            G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(x_bulk).all():
            logger.warning("Bulk data contains NaN/Inf values; replacing with zeros.")
            x_bulk = np.nan_to_num(x_bulk, nan=0.0, posinf=0.0, neginf=0.0)

        topic_weights = np.zeros((n_topics, n_samples))
        for i in range(n_samples):
            w, residual = nnls(G, x_bulk[i, :])
            total = w.sum()
            if total > 1e-12:
                w = w / total
            else:
                logger.warning(
                    f"NNLS returned all-zero weights for sample {i}; "
                    f"falling back to uniform topic distribution."
                )
                w = np.ones(n_topics) / n_topics
            topic_weights[:, i] = w

        return ss.csc_matrix(topic_weights)

    def Decon(self, ntopics_selection, seed_selection, outname=None,
              model=None, model_selection="LDA", benchmark=False):

        if model:
            self.norm_selection = model.norm_selection

        self.ntopics_selection = ntopics_selection
        self.seed_selection = seed_selection

        gene_topic_mat = pd.read_table(os.path.join(self.model_dir, "gene_topic_mat.txt"),
                                       sep="\t", index_col=0)
        self.genes = gene_topic_mat.index.to_list()
        self.topics = gene_topic_mat.columns
        topic_cell_mat = pd.read_table(os.path.join(self.model_dir, "topic_cell_mat.txt"),
                                       sep="\t", index_col=0)
        self.cells = topic_cell_mat.columns

        if outname is None:
            self.outname = "ntopics_" + str(self.ntopics_selection) + "_seed_" + str(self.seed_selection)
        else:
            self.outname = outname

        topic_sample_file = os.path.join(self.model_dir, "topic_sample_mat_" + self.outname + ".txt")

        if model_selection == "LDA":
            try:
                lda = model.model.model
            except:
                if benchmark == False:
                    model_file = search_file_a(self.model_dir, "", "lda_model")[0]
                    lda = LdaModel.load(model_file)
                else:
                    print("error: model is None.")
            genes_dict_file = os.path.join(self.out_dir, "LDA_Gene_dict.txt")
            genes_dict = Dictionary.load_from_text(genes_dict_file)
            genes_dict_token2id = genes_dict.token2id

            count_mat_use = self.mm_df.transpose()
            gs_corpus = []
            for i in range(count_mat_use.shape[1]):
                gs_genes_nonzero_index_corpus = [genes_dict_token2id[j] for j in self.gs_genes]
                gs_genes_nonzero_count = count_mat_use[:, i].toarray().flatten().tolist()
                gs_corpus.append(list(zip(gs_genes_nonzero_index_corpus, gs_genes_nonzero_count)))

            topic_sample = lda.get_document_topics(gs_corpus, minimum_probability=0)
            topic_sample_mat = gensim.matutils.corpus2csc(topic_sample)

        elif model_selection == "ALDA":
            self.mm_int = self.mm_df.copy()
            self.mm_int.data = np.floor(self.mm_int.data).astype(int)
            try:
                alda = model.model.model
            except:
                if benchmark == False:
                    model_file = search_file_a(self.model_dir, "", "alda_model")[0]
                    f = open(model_file, 'rb')
                    alda = pickle.load(f)
                    f.close()
                else:
                    print("error: model is None.")
            print("load seed_k from scTopic or scPreProcessing object.")
            with open(os.path.join(self.model_dir, 'seed_topics.pkl'), 'rb') as f:
                seed_topics = pickle.load(f)
            f.close()

            seed_k = list(np.load(os.path.join(self.model_dir, 'seed_k.npy')))

            alda.verbose = False
            sample_topic_mat = alda.transform(self.mm_int)

            topic_sample_mat = csc_matrix(sample_topic_mat.T)

        elif model_selection == "CorEx":
            try:
                corex = model.model.model
            except:
                if benchmark == False:
                    model_file = search_file_a(self.model_dir, "", "corex_model")[0]
                    f = open(model_file, 'rb')
                    corex = pickle.load(f)
                    f.close()
                else:
                    print("error: model is None.")

            x_bulk = self.mm_df.toarray() if hasattr(self.mm_df, "toarray") else np.asarray(self.mm_df)

            # Align bulk gene columns with model gene set
            model_genes = self.genes
            bulk_genes = self.gs_genes if hasattr(self, 'gs_genes') else None
            if bulk_genes is not None:
                if x_bulk.shape[1] != len(model_genes) or bulk_genes != model_genes:
                    logger.info(f"Aligning bulk data ({x_bulk.shape[1]} genes) to "
                                f"model gene set ({len(model_genes)} genes).")
                    aligned = np.zeros((x_bulk.shape[0], len(model_genes)))
                    shared = set(bulk_genes) & set(model_genes)
                    for g in shared:
                        model_idx = model_genes.index(g)
                        bulk_idx = bulk_genes.index(g)
                        aligned[:, model_idx] = x_bulk[:, bulk_idx]
                    x_bulk = aligned

            topic_sample_mat = None
            topic_sample_logmat = None

            # 优先使用 CorEx 原生 transform（如果可用）
            if hasattr(corex, "transform"):
                try:
                    topic_probs, _ = corex.transform(x_bulk, details=True)
                    topic_sample_mat = ss.csc_matrix(np.asarray(topic_probs).transpose())
                    topic_sample_logmat = None
                    logger.info("CorEx deconvolution used transform(details=True) (inference-only path).")
                except Exception as e:
                    logger.warning(f"CorEx transform() failed ({e}); falling back to NNLS projection.")

            # =================================================================
            # [FIX] NNLS 投影替代 corex.fit() 回退
            #
            # 原版代码:
            #     corex_bulk = corex.fit(x_bulk)
            #     topic_sample_mat = ss.csc_matrix(corex_bulk.p_y_given_x.transpose())
            #
            # 致命缺陷: CorEx fit() 是无监督训练，在 n_samples=1 时跨样本
            # 互信息恒为零，模型无法学习，输出退化为随机初始化的定值。
            # 结果: topic_sample_mat 与输入 x_bulk 完全无关，
            #       反卷积退化为直接输出 scRNA-seq 先验概率。
            # =================================================================
            if topic_sample_mat is None:
                n_bulk = x_bulk.shape[0]
                if n_bulk == 1:
                    logger.info(
                        "Single bulk sample detected; using NNLS projection to "
                        "pre-trained gene_topic space. This avoids the CorEx fit() "
                        "degeneration where mutual information is zero for a single "
                        "sample, causing output to be random initialization "
                        "independent of the input bulk data."
                    )
                else:
                    logger.info(
                        f"Using NNLS projection for {n_bulk} bulk samples to "
                        f"pre-trained gene_topic space."
                    )

                # 重新读取 gene_topic_mat 作为投影基矩阵
                gene_topic_df = pd.read_table(
                    os.path.join(self.model_dir, "gene_topic_mat.txt"),
                    sep="\t", index_col=0)

                # 对齐 bulk 基因到 gene_topic_mat 的基因顺序
                proj_genes = list(gene_topic_df.index)
                x_proj = np.zeros((x_bulk.shape[0], len(proj_genes)))
                shared_g = set(model_genes) & set(proj_genes)
                n_missing = len(proj_genes) - len(shared_g)
                if n_missing > 0:
                    logger.warning(
                        f"{n_missing}/{len(proj_genes)} model genes not found "
                        f"in bulk data; filled with zero."
                    )
                for g in shared_g:
                    mi = proj_genes.index(g)
                    bi = model_genes.index(g)
                    x_proj[:, mi] = x_bulk[:, bi]

                topic_sample_mat = self._project_bulk_to_topic_nnls(x_proj, gene_topic_df)
                topic_sample_logmat = None

            if benchmark == False:
                topic_sample_npz_file = os.path.join(self.model_dir, "topic_sample_mat.npz")
                topic_sample_logfile = os.path.join(self.model_dir, "topic_sample_logmat.npz")
                ss.save_npz(topic_sample_npz_file, topic_sample_mat)
                if topic_sample_logmat is not None:
                    ss.save_npz(topic_sample_logfile, topic_sample_logmat)

        if benchmark == False:
            topic_sample_df = pd.DataFrame(topic_sample_mat.todense(), index=self.topics,
                                           columns=self.gs_samples)
            topic_sample_df.to_csv(topic_sample_file, sep="\t", index=True, header=True)
        _assert_finite("topic_sample_mat",
                       topic_sample_mat.toarray() if ss.issparse(topic_sample_mat) else topic_sample_mat)
        self.topic_sample_mat = topic_sample_mat

    def Bayes(self, norm_selection="Bayes"):
        if norm_selection is None:
            if self.norm_selection is None:
                norm_selection = "BayesNorm"
        else:
            self.norm_selection = norm_selection

        if self.norm_selection == "Raw":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveRaw(
                self.gs_samples, self.ntopics_selection,
                self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "Norm":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveNorm(
                self.gs_samples, self.ntopics_selection,
                self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "NormBySD":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveNormBySD(
                self.gs_samples, self.ntopics_selection,
                self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "Bayes":
            if hasattr(self, "external_topic_celltype_df"):
                logger.info("Using external topic-celltype matrix for Bayes deconvolution")
                tmp = os.path.join(self.model_dir, "topic_celltype_mat.txt")
                self.external_topic_celltype_df.to_csv(tmp, sep="\t")
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveBayes(
                self.gs_samples, self.ntopics_selection,
                self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "BayesNorm":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveBayesNorm(
                self.gs_samples, self.ntopics_selection,
                self.topic_sample_mat, self.out_dir, self.model_dir)
        self.celltype_frac_df = bulk_celltype_array_norm_df

    def PeakPredict(self, sc_count_file, object_gs, GS_file=None, gene_anno_file=None,
                    scATAC_count_file=None, groud_truth_file=None,
                    ifscale=False, pseudo=False, benchmark=False):
        self.peak_file = object_gs.peak_file
        celltype_exp_libnorm, celltype_peak_weight_t = self._bulk_prepare(
            sc_count_file, object_gs, GS_file, gene_anno_file)
        if pseudo:
            adata_CP = ad.read_h5ad(self.peak_file)
            self.peak_meta = adata_CP.var
            celltype_index, celltype_peak_exp, celltype_peak_exp_scaled = self._pseudo_prepare(
                scATAC_count_file, celltype_exp_libnorm)
            if ifscale:
                celltype_peak_exp = None
                celltype_peak_exp_scaled = celltype_peak_exp_scaled.transpose()
            else:
                celltype_peak_exp_scaled = None
        else:
            celltype_index = celltype_exp_libnorm.index
            celltype_peak_exp = None
            celltype_peak_exp_scaled = None

        if self.celltype_frac_df.shape[0] > 1:
            print("put only one sample for predicting peak signals.just predict first sample")
            celltype_frac_df = pd.DataFrame(self.celltype_frac_df.transpose().iloc[:, 0])
        else:
            celltype_frac_df = self.celltype_frac_df.transpose()
        celltype_frac_df.columns = ['predict']

        return self._predict_peak(celltype_peak_weight_t, celltype_index,
                                  celltype_frac_df=celltype_frac_df,
                                  celltype_peak_exp=celltype_peak_exp,
                                  celltype_peak_exp_scaled=celltype_peak_exp_scaled,
                                  groud_truth_file=groud_truth_file,
                                  pseudo=pseudo, benchmark=benchmark)

    def load_external_topic_matrices(self, topic_sample_df, topic_celltype_df):
        if not isinstance(topic_sample_df, pd.DataFrame):
            topic_sample_df = pd.DataFrame(topic_sample_df)
        if not isinstance(topic_celltype_df, pd.DataFrame):
            topic_celltype_df = pd.DataFrame(topic_celltype_df)

        if topic_sample_df.index.tolist() != topic_celltype_df.index.tolist():
            original_n = len(topic_sample_df.index)
            shared = [t for t in topic_sample_df.index if t in topic_celltype_df.index]
            if len(shared) == 0:
                raise ValueError("No shared topics between topic_sample_df and topic_celltype_df")
            logger.warning(
                f"Re-aligned topics: kept {len(shared)} of {original_n} topics "
                f"from topic_sample_df after alignment with topic_celltype_df"
            )
            topic_sample_df = topic_sample_df.loc[shared]
            topic_celltype_df = topic_celltype_df.loc[shared]

        self.topics = topic_sample_df.index.tolist()
        self.gs_samples = topic_sample_df.columns.tolist()
        self.topic_sample_mat = ss.csc_matrix(topic_sample_df.values)
        self.external_topic_celltype_df = topic_celltype_df.copy()
        logger.info(
            f"Loaded external topic matrices: {len(self.topics)} topics, "
            f"{len(self.gs_samples)} samples, "
            f"{topic_celltype_df.shape[1]} cell types"
        )
        return self

    def _build_E_ref(self, sc_count_file):
        from collections import defaultdict

        adata = ad.read_h5ad(sc_count_file)
        use_genes = list(set(adata.var_names) & set(self.gs_genes))
        adata = adata[:, use_genes].copy()

        cell_dict = defaultdict(list)
        for k, v in self.ann_dict.items():
            if k in adata.obs_names:
                cell_dict[v].append(k)

        row_name = adata.obs_names.to_list()
        mat = adata.X
        dic_exp = {}
        for type_, names in cell_dict.items():
            if len(names) == 0:
                continue
            idx = [row_name.index(i) for i in names]
            tmp_mat = mat[idx, :]
            tmp_exp = np.asarray(tmp_mat.sum(axis=0)).flatten()
            dic_exp[type_] = tmp_exp

        celltype_exp = pd.DataFrame.from_dict(dic_exp, orient='index')
        celltype_exp.columns = adata.var_names.to_list()

        scale = celltype_exp.sum(axis=1) / 10000.0
        scale[scale == 0] = 1.0
        E_ref = celltype_exp.div(scale, axis=0)

        if hasattr(self, "model_dir") and self.model_dir:
            E_ref.to_csv(os.path.join(self.model_dir, "E_ref.tsv"), sep="\t")
            pd.Series(scale, index=E_ref.index).to_csv(
                os.path.join(self.model_dir, "E_ref_scale_factors.tsv"),
                sep="\t", header=["scale_factor"],
            )
            logger.info(
                f"Saved E_ref ({E_ref.shape[0]} cell types x {E_ref.shape[1]} genes) "
                f"and scale factors to {self.model_dir}"
            )

        return E_ref

    def _bulk_prepare(self, sc_count_file, object_gs, GS_file=None, gene_anno_file=None):

        if object_gs.GS_file:
            self.GS_file = object_gs.GS_file
            gene_anno_file = object_gs.gene_anno_file
            self.peak_file = object_gs.peak_file
            adata_GS = ad.read_h5ad(GS_file)

        if GS_file:
            try:
                adata_GS = ad.read_h5ad(GS_file)
                print("read gene peak weight matrix")
            except:
                gene_anno = pd.read_csv(gene_anno_file,
                                        encoding='utf8', sep='\t', header=None,
                                        names=['chr', 'start', 'end', 'symbol', 'strand'])
                G2P = Gene2Peaks(gene_anno_file=gene_anno_file, peaks=self.peak_meta, cutoff_weight=0)
                df, mat_GP = G2P.gwt_gene_peak_match()
                adata_GS = ad.AnnData(mat_GP)
                row_names = self.peak_meta.index.to_list()
                adata_GS.obs_names = pd.Index(gene_anno.symbol)
                adata_GS.var_names = self.peak_meta.index
                print("compute gene peak weight matrix")

        celltype_exp_libnorm = self._build_E_ref(sc_count_file)

        gene_select = list(set(celltype_exp_libnorm.columns) & set(adata_GS.obs_names))
        celltype_exp_libnorm = celltype_exp_libnorm[gene_select]
        ex = np.array(celltype_exp_libnorm)
        idx = [adata_GS.obs_names.to_list().index(i) for i in gene_select]
        adata_GS = adata_GS[idx, :]
        gs = adata_GS.X.toarray()

        Ep2 = np.dot(ex, gs)
        Ep2_df = pd.DataFrame(Ep2)
        Ep2_df.columns = adata_GS.var_names
        Ep2_df.index = celltype_exp_libnorm.index

        gs_scale_factor = Ep2_df.sum(axis=0) / len(celltype_exp_libnorm.index)
        dic_exp = {}
        for idx, row in Ep2_df.transpose().iterrows():
            if gs_scale_factor[idx] != 0:
                dic_exp[idx] = row / gs_scale_factor[idx]
            else:
                dic_exp[idx] = row
        celltype_peak_weight = pd.DataFrame.from_dict(dic_exp, orient='index')

        try:
            peak_name = adata_GS.var["chr"].astype(str) + "_" + \
                adata_GS.var["start"].astype(str) + "_" + \
                adata_GS.var["end"].astype(str)
            adata_GS.var_names = peak_name
        except:
            print("default peak name")
        self.var_names = adata_GS.var_names

        celltype_peak_weight_t = np.array(celltype_peak_weight.transpose())
        return celltype_exp_libnorm, celltype_peak_weight_t

    def _pseudo_prepare(self, scATAC_count_file, celltype_exp_libnorm):

        adata_sc_CP = ad.read_h5ad(scATAC_count_file)
        celltype_col = 'cell_type'
        celltype_col = 'main_cell_type'
        try:
            adata_sc_CP.var_names = [i.replace(':', '_').replace('-', '_') for i in adata_sc_CP.var_names]
        except:
            pass
        try:
            adata_sc_CP = adata_sc_CP[self.sc_cells, self.peak_meta.index]
            adata_sc_CP.obs.loc[:, celltype_col] = [v for k, v in self.ann_dict.items()]
        except:
            adata_sc_CP = adata_sc_CP[:, self.peak_meta.index]
        row_name = adata_sc_CP.obs_names.to_list()
        col_name = adata_sc_CP.var_names.to_list()

        dic_exp = {}
        for type_ in celltype_exp_libnorm.index:
            names = adata_sc_CP.obs_names[adata_sc_CP.obs[celltype_col] == type_].to_list()
            tmp_mat = adata_sc_CP.X[[row_name.index(i) for i in names], :]
            tmp_exp = tmp_mat.mean(axis=0).flatten().tolist()[-1]
            dic_exp[type_] = tmp_exp

        celltype_peak_exp = pd.DataFrame.from_dict(dic_exp, orient='index')
        celltype_peak_exp.columns = col_name
        celltype_peak_exp.index = celltype_exp_libnorm.index

        gs_scale_factor = np.round(celltype_peak_exp.sum(axis=0) / 10000)
        dic_exp = {}
        for idx, row in celltype_peak_exp.transpose().iterrows():
            if gs_scale_factor[idx] != 0:
                dic_exp[idx] = row / gs_scale_factor[idx]
            else:
                dic_exp[idx] = row
        celltype_peak_exp_scaled = pd.DataFrame.from_dict(dic_exp, orient='index')

        celltype_index = celltype_exp_libnorm.index

        return celltype_index, celltype_peak_exp, celltype_peak_exp_scaled

    def _predict_peak(self, celltype_peak_weight_t, celltype_index, celltype_frac_df,
                      celltype_peak_exp=None, celltype_peak_exp_scaled=None,
                      groud_truth_file=None, pseudo=False, benchmark=False):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from math import sqrt
        import scipy.sparse

        if self.peak_file.endswith("h5ad"):
            adata_CP = ad.read_h5ad(self.peak_file)
        elif self.peak_file.endswith("bed"):
            peak_tmp = pd.read_table(self.peak_file, sep="\t", header=None)
            peak_tmp.columns = ["chr", "start", "end", "score"]
            peak_tmp.index = ["peak" + str(i) for i in range(peak_tmp.shape[0])]
            adata_CP = ad.AnnData(X=peak_tmp['score'].values.reshape(-1, 1).transpose(),
                                  obs=pd.DataFrame(index=["Bulk"]),
                                  var=peak_tmp[["chr", "start", "end"]])

        expected_peaks = len(self.var_names)

        if adata_CP.shape[1] != expected_peaks:
            print(f"检测到维度不匹配: adata_CP {adata_CP.shape}, 预期列数 {expected_peaks}")
            if adata_CP.shape[0] == expected_peaks:
                print("正在转置 adata_CP 以匹配 (Samples x Peaks) ...")
                adata_CP = adata_CP.T
            else:
                raise ValueError(f"Peak文件的维度 {adata_CP.shape} 与预期的 Peak 数量 {expected_peaks} 不匹配。")

        try:
            peak_name = adata_CP.var["chr"].astype(str) + "_" + \
                adata_CP.var["start"].astype(str) + "_" + \
                adata_CP.var["end"].astype(str)
            adata_CP.var_names = peak_name
        except:
            print("default peak name")

        self.peak_meta = adata_CP.var
        peak = adata_CP.X

        if peak.shape[0] >= 1:
            if scipy.sparse.issparse(peak):
                peak_vec = peak[0, :].toarray()
            else:
                peak_vec = peak[0, :].reshape(1, -1)
        else:
            raise ValueError("Peak 数据为空")

        peak = peak_vec

        if peak.shape[1] != expected_peaks:
            raise ValueError(f"处理后的 Peak 向量形状 {peak.shape} 仍不匹配预期列数 {expected_peaks}")

        res_df = celltype_frac_df

        if groud_truth_file is not None:
            gt = pd.read_table(groud_truth_file, sep='\t', index_col=0)
            res_df = res_df.loc[gt.index, :]
            res_df["gt"] = gt['Truth']
            self.gt = gt

        per = np.array(res_df['predict'].loc[celltype_index].tolist())
        per = np.expand_dims(per, axis=1)

        Ep1 = np.dot(per, peak)

        Ep1_df = pd.DataFrame(Ep1, columns=self.var_names, index=celltype_index)

        Ep = np.multiply(celltype_peak_weight_t, Ep1_df)
        Ep_df = pd.DataFrame(Ep, columns=self.var_names, index=celltype_index)
        nonzero_cols = Ep_df.sum(0) > 0
        if nonzero_cols.sum() == 0:
            print(f"警告：模型 {self.norm_selection} (ntop={self.ntopics_selection}, seed={self.seed_selection}) 无有效列")
        Ep_df_filter = Ep_df.loc[:, nonzero_cols]

        if pseudo:
            if celltype_peak_exp_scaled is None:
                celltype_peak_exp_filter = celltype_peak_exp.loc[:, nonzero_cols]
            else:
                celltype_peak_exp_filter = celltype_peak_exp_scaled.loc[:, nonzero_cols]

            assert not Ep_df_filter.isnull().values.any(), "NaN 存在于 Ep_df_filter"

            value_cor_dic = {
                idx: Ep_df_filter.loc[idx].corr(value)
                for idx, value in celltype_peak_exp_filter.iterrows()
            }
            final_p = pd.DataFrame.from_dict(value_cor_dic, orient='index').mean(0)[0]
            self.predicted_peak = Ep_df_filter

            if benchmark:
                cor_res = res_df.corr().iloc[0, 1]
                rmse_frac = sqrt(mean_squared_error(res_df["gt"], res_df['predict']))

                y_predict = Ep_df_filter.sum(0).to_numpy()
                y_test = celltype_peak_exp_filter.sum(0).to_numpy()

                if scipy.sparse.issparse(adata_CP.X):
                    y_bulk_test = adata_CP.X[:, nonzero_cols].toarray().flatten()
                else:
                    y_bulk_test = adata_CP.X[:, nonzero_cols].flatten()

                assert not np.isnan(y_predict).any(), "y_predict 含 NaN"
                assert not np.isnan(y_test).any(), "y_test 含 NaN"
                assert not np.isnan(y_bulk_test).any(), "y_bulk_test 含 NaN"

                mae = mean_absolute_error(y_test, y_predict)
                mse = mean_squared_error(y_test, y_predict)
                rmse = sqrt(mse)
                r2 = r2_score(y_test, y_predict)

                mae_b = mean_absolute_error(y_bulk_test, y_predict)
                mse_b = mean_squared_error(y_bulk_test, y_predict)
                rmse_b = sqrt(mse_b)
                r2_b = r2_score(y_bulk_test, y_predict)
                results = {
                    'ntop': self.ntopics_selection,
                    'model': self.norm_selection,
                    'seed': self.seed_selection,
                    'percent_cor': cor_res,
                    'RMSE_frac': rmse_frac,
                    'RMSE_bulk': rmse_b,
                    'MSE_bulk': mse_b,
                    'R2_bulk': r2_b,
                    'MAE_bulk': mae_b,
                    'RMSE': rmse,
                    'MSE': mse,
                    'R2': r2,
                    'MAE': mae,
                    'peak_count_cor': final_p
                }
                return results
            else:
                return final_p, celltype_peak_exp_filter
        else:
            return Ep_df_filter


def _to_ndarray(x):
    if ss.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_normalize_rows(arr, eps=1e-12):
    a = _to_ndarray(arr).astype(float, copy=True)
    row_sums = a.sum(axis=1, keepdims=True)
    zero_mask = row_sums <= eps
    if np.any(zero_mask):
        logger.warning(f"Row normalization encountered {int(zero_mask.sum())} zero-sum rows; using denominator=1.")
        row_sums[zero_mask] = 1.0
    out = a / row_sums
    return out


def safe_normalize_cols(arr, eps=1e-12):
    a = _to_ndarray(arr).astype(float, copy=True)
    col_sums = a.sum(axis=0, keepdims=True)
    zero_mask = col_sums <= eps
    if np.any(zero_mask):
        logger.warning(f"Column normalization encountered {int(zero_mask.sum())} zero-sum columns; using denominator=1.")
        col_sums[zero_mask] = 1.0
    out = a / col_sums
    return out


def _assert_finite(name, arr):
    a = _to_ndarray(arr)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN/Inf values")
