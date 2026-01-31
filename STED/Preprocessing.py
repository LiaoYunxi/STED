# -*- coding: utf-8 -*-
#%%
import gc
import copy
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler #MinMaxScaler
from sklearn.preprocessing import StandardScaler


from logging import getLogger
logger = getLogger('Preprocessing')

from .utils import *
from .data import *
from .MarkerFind import *
from .Genescore import *

def Gene_scores(adata,
                genome,
                gene_anno=None,
                tss_upstream=1e5,
                tss_downsteam=1e5,
                gb_upstream=5000,
                cutoff_weight=1,
                use_top_pcs=True,
                use_precomputed=True,
                use_gene_weigt=True,
                min_w=1,
                max_w=5):
    """Calculate gene scores

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genome : `str`
        Reference genome. Choose from {'hg19', 'hg38', 'mm9', 'mm10'}
    gene_anno : `pandas.DataFrame`, optional (default: None)
        Dataframe of gene annotation.
        If None, built-in gene annotation will be used depending on `genome`;
        If provided, custom gene annotation will be used instead.
    tss_upstream : `int`, optional (default: 1e5)
        The number of base pairs upstream of TSS
    tss_downsteam : `int`, optional (default: 1e5)
        The number of base pairs downstream of TSS
    gb_upstream : `int`, optional (default: 5000)
        The number of base pairs upstream by which gene body is extended.
        Peaks within the extended gene body are given the weight of 1.
    cutoff_weight : `float`, optional (default: 1)
        Weight cutoff for peaks
    use_top_pcs : `bool`, optional (default: True)
        If True, only peaks associated with top PCs will be used
    use_precomputed : `bool`, optional (default: True)
        If True, overlap bewteen peaks and genes
        (stored in `adata.uns['gene_scores']['overlap']`) will be imported
    use_gene_weigt : `bool`, optional (default: True)
        If True, for each gene, the number of peaks assigned to it
        will be rescaled based on gene size
    min_w : `int`, optional (default: 1)
        The minimum weight for each gene.
        Only valid if `use_gene_weigt` is True
    max_w : `int`, optional (default: 5)
        The maximum weight for each gene.
        Only valid if `use_gene_weigt` is True

    Returns
    -------
    adata_new: AnnData
        Annotated data matrix.
        Stores #cells x #genes gene score matrix

    updates `adata` with the following fields.
    overlap: `pandas.DataFrame`, (`adata.uns['gene_scores']['overlap']`)
        Dataframe of overlap between peaks and genes
    """
    GS = GeneScores(adata,
                    genome,
                    gene_anno=gene_anno,
                    tss_upstream=tss_upstream,
                    tss_downsteam=tss_downsteam,
                    gb_upstream=gb_upstream,
                    cutoff_weight=cutoff_weight,
                    use_top_pcs=use_top_pcs,
                    use_precomputed=use_precomputed,
                    use_gene_weigt=use_gene_weigt,
                    min_w=min_w,
                    max_w=max_w)
    mat_GP,df_overlap_updated = GS.cal_gene_scores()
    return mat_GP,df_overlap_updated

#%%
class SetData():
    def __init__(self):
        self.raw_df = None
        self.genes = None
        self.anchor_dict = {}
        self.final_int = None
        self.final_genes = None
        #self.input_mat = None
    
    def set_expression(self,df,genes,other_genes):
        self.raw_df = df
        self.genes = genes
        self.other_genes = other_genes
        # seed-topic preparation
        self.gene2id = dict((v, idx) for idx, v in enumerate(self.genes))

    def set_anchor(self,anchored_genes=[],do_plot=True):
        """
        Set anchored genes
        ----------
        anchored_genes : list or dict
            e.g. {'Cell A': ['gene A1', 'gene A2'],'Cell B': ['gene B1', 'gene B2'],...}
            
        """
        if isinstance(anchored_genes,list):
            anchor_dict = {i: words for i, words in enumerate(anchored_genes)}
        elif isinstance(anchored_genes,dict):
            anchor_dict = anchored_genes
        else:
            print("Error: please input anchored genes in right format.")

        genes = self.genes
        new_v = []
        new_k = []
        for i,k in enumerate(anchor_dict):
            marker = anchor_dict.get(k)
            tmp_common = sorted(list(set(marker) & set(genes)))
            if len(tmp_common) > 0:
                tmp_v = [t for t in tmp_common]
                new_v.append(tmp_v)
                new_k.append(k)
            else:
                pass
        anchor_dict2 = dict(zip(new_k,new_v))
        self.anchor_dict = anchor_dict2
        anchors = set(list(itertools.chain.from_iterable(list(self.anchor_dict.values()))))
        print('--- reflect genes in expression ---')
        print(len(self.anchor_dict),'anchored topics')
        print(len(anchors),'genes were registered')
        
        # plot the original registered anchors size
        if do_plot:
            y = [len(t) for t in self.anchor_dict.values()]
            x = [i for i in range(len(y))]
            plt.bar(x,y)
            plt.xticks(x,self.anchor_dict.keys(),rotation=75)
            plt.title('Original anchored gene size')
            plt.show()
        
        # detect cell-type specific anchors
        count_dic = dict(collections.Counter(list(itertools.chain.from_iterable(list(self.anchor_dict.values())))))
        sort_count = sorted(count_dic.items(),key=lambda x : x[1])
        unique_anchor = [] # no overlap

        for t in sort_count:
            if t[1] == 1:
                unique_anchor.append(t[0])
            else:
                pass
        new_v = []
        new_k = []
        for i,k in enumerate(self.anchor_dict):
            tmp_v = sorted(list(set(self.anchor_dict.get(k)) & set(unique_anchor)))
            if len(tmp_v) > 0:
                new_v.append(tmp_v)
                new_k.append(k)
            else:
                pass
        self.spe_anchor_dict = dict(zip(new_k,new_v))
        spe_anchors = set(list(itertools.chain.from_iterable(list(self.spe_anchor_dict.values()))))

        print('--- extract cell type specific anchors ---')
        print(len(self.spe_anchor_dict),'cells')
        print(set(self.anchor_dict.keys())-set(self.spe_anchor_dict.keys()),'type of anchored genes were removed (no specific after removing overlap)')
        print(len(spe_anchors),'genes were registered')
        
        # plot the cell-type specific anchor size
        if do_plot:
            y = [len(t) for t in self.spe_anchor_dict.values()]
            x = [i for i in range(len(y))]
            plt.bar(x,y)
            plt.xticks(x,self.spe_anchor_dict.keys(),rotation=75)
            plt.title('Specific Anchor Size')
            plt.show()
    
    def expression_processing(self,do_scale=True):
        """
        1. Determine if the anchors are cell specific.
        2. Add non-seleted gene at random.
        3. Process expression data into a format for analysis
        ----------
        random_n : int
            Number of genes to be added randomly. The default is 0.
        specific : bool
            Whether or not to select only cell-specific markers. The default is True.
        """

        # normalization
        if do_scale:
            final_df = StandardScaler(with_mean=False).fit_transform(self.raw_df)
            self.final_linear = final_df
            self.final_int = final_df.copy()
            #final_df = self.raw_df.multiply(1 / norm_scale)
        
        self.final_linear = self.raw_df
        self.final_int = self.raw_df.copy()
        self.final_int.data = np.floor(self.final_int.data).astype(int)
        #self.input_mat = np.array(self.final_int.T.todense(), dtype='int64')

    def seed_processing(self,specific=True):
        """
        Prepare seed information for guiding each topic. These are used as prior information.
        
        input_mat : np.array
            samples are in rows and genes (markers) are in columns.
        seed_topics : dict
        seed_k : list
        use_anchor_dict: dcit
        """
        if specific:
            self.use_anchor_dict = self.spe_anchor_dict
        else:
            self.use_anchor_dict = self.anchor_dict
        
        if self.use_anchor_dict is None:
            print('Anchors were not defined, use unsupervised model.')
        else:
            # seed_topic preparation
            self.use_anchor_list = [self.use_anchor_dict[i] for i in sorted(self.use_anchor_dict.keys())]
            genes = list(itertools.chain.from_iterable(self.use_anchor_list))
            seed_topic_list = self.use_anchor_list
            seed_topics = {}
            finish_genes = []
            for t_id, st in enumerate(seed_topic_list):
                for gene in st:
                    try:
                        if gene in finish_genes:
                            tmp = seed_topics[self.gene2id[gene]]
                            seed_topics[self.gene2id[gene]] = tmp + [t_id]
                        else:
                            seed_topics[self.gene2id[gene]] = [t_id]
                            finish_genes.append(gene)
                    except:
                        # not included in target expression table
                        print(gene)
                        pass
                    
            # reliable anchored genes to guide
            genes = list(itertools.chain.from_iterable(self.use_anchor_list))
            seed_k = []
            for g in genes:
                if self.gene2id.get(g) is None:
                    pass
                else:
                    seed_k.append(self.gene2id.get(g))

            self.seed_topics = seed_topics
            seed_k = sorted(list(set(seed_k)))
            self.seed_k = seed_k

            print('seed number:',len(self.seed_topics))
            print("seed_k:",len(self.seed_k))

#%%
class scPreProcessing():
    def __init__(self):
        self.mix_raw = None
        self.mm_df = None
        self.target_df = None
        self.use_genes = None
        self.anchor_dict = None
        self.anchor_list = None
        self.seed_topics = None
        self.seed_k = None
        self.use_anchor_dict =None
    
    def set_data(self,sc_count_file,sc_anno_file,batch_info=None):
        # read scRNA-seq data
        if sc_count_file.endswith(".h5"):
            sc_count = read_10X_h5(sc_count_file)
            sc_count_mat = sc_count.matrix
            sc_count_genes = sc_count.names.tolist()
            sc_count_cells = sc_count.barcodes.tolist()
            if type(sc_count_genes[0]) == bytes:
                sc_count_genes = [i.decode() for i in sc_count_genes]
            if type(sc_count_cells[0]) == bytes:
                sc_count_cells = [i.decode() for i in sc_count_cells]
        elif sc_count_file.endswith(".h5ad"):
            sc_count = read_10X_h5ad(sc_count_file)
            sc_count_mat = sc_count.matrix
            sc_count_genes = sc_count.names.tolist()
            sc_count_cells = sc_count.barcodes.tolist()
            if type(sc_count_genes[0]) == bytes:
                sc_count_genes = [i.decode() for i in sc_count_genes]
            if type(sc_count_cells[0]) == bytes:
                sc_count_cells = [i.decode() for i in sc_count_cells]
        else:
            sc_count = read_count(sc_count_file)
            sc_count_mat = sc_count["matrix"]
            sc_count_mat = csc_matrix(sc_count_mat, dtype=np.float32)
            sc_count_genes = sc_count["features"]
            sc_count_cells = sc_count["barcodes"]

        # filter the count matrix
        # remove negative values (can be directly removed in csr_matrix without using applymap)
        if len(sc_count_genes) != sc_count_mat.shape[0]:
            sc_count_mat =sc_count_mat.transpose()

        sc_count_mat.data[sc_count_mat.data < 0] = 0
        # remove 0 values by filtering out columns where the sum of each column is zero
        count_per_cell = np.array(sc_count_mat.sum(axis=0)).flatten()
        non_zero_columns_idx = np.where(count_per_cell != 0)[0]
        sc_count_mat = sc_count_mat[:, non_zero_columns_idx]
        # sc_count_cells = [sc_count_cells[i] for i in non_zero_columns_idx.tolist()]
        count_per_cell = count_per_cell[non_zero_columns_idx]
        # TODO:sc_count_mat shape: gene X cell 注释掉了下面一句
        #sc_count_genes = [sc_count_genes[i] for i in range(len(sc_count_genes)) if i in non_zero_columns_idx.tolist()]

        # read cell-type meta file
        cell_celltype_dict = {}
        for line in open(sc_anno_file, "r"):
            items = line.strip().split("\t")
            cell_celltype_dict[items[0]] = items[1]
            self.ann_dict = cell_celltype_dict

        self.mix_raw = sc_count_mat
        self.batch_info = batch_info
        self.sc_genes = sc_count_genes
        self.sc_cells = sc_count_cells
        logger.info('original: {}'.format(self.mix_raw.shape))

    def cell_selection(self, target_samples=[]):
        # Select samples containing the specified prefixes for analysis
        if len(target_samples) > 0:
            samples = self.sc_cells
            use_samples = []
            for t in samples:
                if t.split('_')[0] in target_samples:
                    use_samples.append(t)
            use_samples_idx = [i for i, sample in enumerate(samples) if sample in use_samples]
            # TODO:self.mix_raw[use_samples_idx,:]-->self.mix_raw[use_samples_idx,:]
            self.mix_raw = self.mix_raw[:,use_samples_idx]
            self.sc_cells = use_samples
    
    def preprocessing(self,linear2log=False,log2linear=False,do_drop=True,do_batch_norm=False,do_norm=True,do_quantile=False):
        """
        数据预处理函数，包含多种预处理步骤
        
        参数:
        linear2log: 是否进行线性到对数转换
        log2linear: 是否进行对数到线性转换
        do_drop: 是否去除全零行
        do_batch_norm: 是否进行批次归一化
        do_norm: 是否进行标准化处理
        do_quantile: 是否进行分位数归一化
        """

        if self.target_df is None:
            data_ = copy.deepcopy(self.mix_raw)
        else:
            data_ = copy.deepcopy(self.target_df)
        
        if not np.issubdtype(data_.data.dtype, np.floating):
            data_.data = data_.data.astype(np.float32)  # 或 float64，根据需求
        
        # trimming
        if do_drop:
            df_c = copy.deepcopy(data_)
            df_c.data[df_c.data == 0] = np.nan  # 将 0 替换为 NaN
            data_ = csr_matrix(df_c)  # 去除全为 NaN 的行
            logger.info('trimming: {}'.format(data_.shape))

        # batch normalization
        if self.batch_info is None:
            do_batch_norm = False
        else:
            if do_batch_norm:
                df_c = copy.deepcopy(data_)
                info = self.batch_info.loc[self.sc_cells] # sample selection

                replace_list = info["replace"].tolist()
                prep_list = info["prep_batch"].tolist()
                lane_list = info["lane_batch"].tolist()
                lst_batch = [replace_list,prep_list,lane_list]

                comb_df = multi_batch_norm(df_c,lst_batch,do_plots=False)
                comb_df.data = np.maximum(comb_df.data, 0)  # 将负值替换为 0
                data_ = csr_matrix(comb_df)
                logger.info('batch normalization: {}'.format(data_.shape))
        
        if do_norm:
            df_c = copy.deepcopy(data_)
            count_per_cell = np.array(df_c.sum(axis=0)).flatten()
            # print("count_per_cell shape:",count_per_cell.shape)
            sc_scale_factor = np.round(np.quantile(count_per_cell, 0.75)/1000, 0)*1000
            r,c = df_c.nonzero()
            count_per_cell_sp = csr_matrix(((1.0/count_per_cell)[c], (r,c)), shape=(df_c.shape))
            data_ = df_c.multiply(count_per_cell_sp)*sc_scale_factor

        # linear --> log2
        if linear2log:
            def log2_sparse(sparse_matrix):
                sparse_matrix.data = np.log2(sparse_matrix.data + 1)
                return sparse_matrix
                
            df_c = copy.deepcopy(data_)
            data_ = log2_sparse(df_c)
            logger.info('linear2log: {}'.format(data_.shape))

        # log2 --> linear
        if log2linear:
            df_c = copy.deepcopy(data_)
            df_c = np.where(df_c.data < 30, 2 ** df_c.data, 1073741824) # avoid overflow
            data_= csr_matrix(df_c)
            logger.info('log2linear: {}'.format(data_.shape))

        # quantile normalization
        if do_quantile:
            # gene x cells
            df_c = copy.deepcopy(data_)
            qn_df = quantile_sparse(df_c)
            data_ = csr_matrix(qn_df)
            logger.info('quantile normalization: {}'.format(data_.shape))

        self.target_df = copy.deepcopy(data_)

    def geneWords(self, sc_anno_file, gene_use=None, ntopMarker=200, ntopHvg=500,save_path=None):
        """
        Select genes based on the specified method (MARKER, HVG).

        Parameters
        ----------
        sc_anno_file : str
            The annotation file used for MARKER gene selection.
        gene_use : str, default 'HVG'
            The method to use for gene selection. Options are 'All', 'MARKER', 'HVG'.
        ntopMarker : int, default 200
            Number of top marker genes to select when using 'MARKER'.
        ntopHvg : int, default 500
            Number of highly variable genes (HVG) to select when using 'HVG'.
        
        Returns
        -------
        genes_shared : list
            List of shared genes based on the specified selection method.
        """
        if isinstance(gene_use, list):
            genes_shared = list(set(self.sc_genes) & set(gene_use))
        else:
            if gene_use == 'MARKER':
                marker_genes = MarkerFind(self.mix_raw, self.sc_genes, self.sc_cells, sc_anno_file, ntopMarker)
                genes_shared = list(set(self.sc_genes) & set(marker_genes))
            elif gene_use == 'HVG':
                # Use highly variable genes (HVG)
                # Calculate variance and mean on the sparse matrix to avoid dense conversion
                if self.target_df is None:
                    target_df = self.mix_raw
                else:
                    target_df = self.target_df
                X = self.mix_raw.toarray()
                mean_df = X.mean(axis=1)
                var_df = X.var(axis=1, ddof=1)
                cv_df = var_df / mean_df
                valid = mean_df > 0
                cv_df[~valid] = -np.inf  # 无效基因排最后
                top_indices = np.argsort(cv_df)[::-1][:ntopHvg]  # 降序前500
                high_genes = [self.sc_genes[i] for i in top_indices]

                # var_df = np.var(target_df.toarray(), axis=1, ddof=1)
                # mean_df = target_df.toarray().mean(axis=1)
                # cv_df = var_df / mean_df
                # cv_df = pd.DataFrame(cv_df).sort_values(by=0, ascending=False)
                # high_genes = [self.sc_genes[i] for i in cv_df.index[0:ntopHvg]]
                marker_genes = MarkerFind(self.mix_raw.transpose(), self.sc_genes, self.sc_cells, sc_anno_file, ntop=ntopMarker)

                # Combine HVG and marker genes
                gene_tmp = set(marker_genes) | set(high_genes)
                genes_shared = list(set(self.sc_genes) & gene_tmp)
            else:
                genes_shared = list(set(self.sc_genes))

        use_gene_idx = [i for i, gene in enumerate(self.sc_genes) if gene in genes_shared]
        try:
            # TODO: self.target_df[:,use_gene_idx]-->self.target_df[use_gene_idx,:]
            self.target_df = self.target_df[use_gene_idx,:]
        except:
            # TODO: self.mix_raw[:,use_gene_idx]-->self.mix_raw[use_gene_idx,:]
            self.mix_raw = self.mix_raw[use_gene_idx,:]
        self.use_genes = [self.sc_genes[i] for i in use_gene_idx]
        self.other_genes = list(set(self.sc_genes)-set(self.use_genes))
        
        if save_path is not None:
            pd.DataFrame(self.use_genes).to_csv(os.path.join(save_path,"use_gene.csv"))

    def geneShared(self,epi_genes):
        if self.use_genes is None:
            self.use_genes = self.vector_genes
        if self.target_df is None:
            try:
                self.target_df = self.input_df
            except:
                self.target_df = self.mix_raw
        use_gene_idx = [i for i, gene in enumerate(self.use_genes) if gene in epi_genes]
        # TODO:self.target_df[:,use_gene_idx]-->self.target_df[use_gene_idx,:]
        self.target_df = self.target_df[use_gene_idx,:]
        self.use_genes = [self.use_genes[i] for i in use_gene_idx]

    def preForGuide(self,anchored_genes=[],specific=True,scale=False,minmax =True,mm_scale=10,do_plot=False):
        SD = SetData()
        # TODO:self.target_df:n_genes * n_cells--> n_cells * n_genes
        SD.set_expression(self.target_df.transpose(),self.use_genes,self.other_genes)
        SD.set_anchor(anchored_genes,do_plot)
        SD.expression_processing(scale)
        SD.seed_processing(specific=specific)

        print("done.")
        self.target_df = SD.final_linear
        self.target_int = SD.final_int

        self.gene2id = SD.gene2id

        self.use_anchor_dict = SD.use_anchor_dict
        self.use_anchor_list = SD.use_anchor_list

        self.seed_topics = SD.seed_topics
        self.seed_k = SD.seed_k

        # correlation between samples
        if do_plot:
            print("plot heatmap.")
            dense_matrix = pd.DataFrame.sparse.from_spmatrix(self.target_int.transpose(), 
                                                     index=self.use_genes,
                                                     columns=self.sc_cells)
            cor = dense_matrix.corr()
            sns.heatmap(cor)
            plt.show()

        if minmax:
            # Sample-wide normalization
            mm_scaler = MaxAbsScaler()
            self.mm_df = (mm_scaler.fit_transform(self.target_int.T)*mm_scale).T
            # correlation between samples
            if do_plot:
                dense_matrix = pd.DataFrame.sparse.from_spmatrix(self.mm_df, 
                                            index=self.use_genes,
                                            columns=self.sc_cells)
                cor = dense_matrix.corr()
                sns.heatmap(cor)
                plt.show()
            logger.info('minmax_scaling: {}'.format(mm_scale))
        else:
            pass
        gc.collect()

    def preprocessing_forBERT(self,sc_count_file,sc_anno_file,gene_names,pre_normalized=False,
                              anchored_genes=[],anchored_strength=2,specific=True):
        import scanpy as sc
        # gene_embeddings = np.load(gene2vec_file)
        data = sc.read_h5ad(sc_count_file)
        data = data[:,data.var_names.isin(gene_names)].copy()

        indices = [index for index, element in enumerate(gene_names) if element in data.var_names] 
        # indices = [gene_names.index(element) for i, element in enumerate(data.var_names)] 
        data_csr = np.zeros((data.shape[0], len(gene_names)), dtype=np.float32) 

        if pre_normalized == False:
            raw_csr = np.zeros((data.shape[0], len(gene_names)), dtype=np.float32) 
            raw_csr[:, indices]=np.array(data.X.todense())
            self.mix_raw = copy.deepcopy(csr_matrix(raw_csr))
            sc.pp.normalize_total(data, target_sum=1e4)
            sc.pp.log1p(data)

            data_csr[:, indices]=np.array(data.X.todense())
        else:
            data_csr[:, indices]=np.array(data.X)
            
        self.input_df = copy.deepcopy(csr_matrix(data_csr))

        self.sc_genes = gene_names
        self.sc_cells = data.obs_names
        # self.sc_genes = self.vector_genes
        self.input_int = copy.deepcopy(data_csr)
        self.input_int.data = np.floor(self.input_int.data).astype(int)

        # read cell-type meta file
        cell_celltype_dict = {}
        for line in open(sc_anno_file, "r"):
            items = line.strip().split("\t")
            cell_celltype_dict[items[0]] = items[1]
            self.ann_dict = cell_celltype_dict

        if len(anchored_genes)>0:
            if isinstance(anchored_genes,list):
                anchor_dict = {i: words for i, words in enumerate(anchored_genes)}
            elif isinstance(anchored_genes,dict):
                anchor_dict = anchored_genes
            else:
                print("Error: please input anchored genes in right format.")
            
            row = []
            col = []
            data = []
            gene_name_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
            for i, word_list in enumerate(anchored_genes):
                for word in word_list:
                    if word in gene_name_to_idx:
                        row.append(i)
                        col.append(gene_name_to_idx[word])
                        data.append(anchored_strength) 
            anchor_matrix = csr_matrix((data, (row, col)), shape=(len(anchored_genes), len(gene_names)))

            SD = SetData()
            SD.set_expression(self.input_int,self.sc_genes,[])
            SD.set_anchor(anchored_genes,do_plot=False)
            SD.seed_processing(specific=specific)

            self.gene2id = SD.gene2id
            self.use_anchor_dict = SD.use_anchor_dict
            self.use_anchor_list = SD.use_anchor_list
            self.seed_topics = SD.seed_topics
            self.seed_k = SD.seed_k
            self.seed_mat = anchor_matrix

class gsPreProcessing():
    def __init__(self):
        self.mix_raw = None
        self.mm_df = None
        self.target_df = None
        self.batch_info = None
        self.gene_anno_file=None
        self.GS_file = None
        self.peak_file = None
    
    def set_data(self,benchmark=False,Epi=False,gs_count_file=None,
                 GS_file=None,batch_info=None,
                 gene_anno_file=None,peak_file=None):
        if Epi:
            if gs_count_file is None:
                if benchmark:
                    use_top_pcs=True
                else:
                    use_top_pcs=False
                    
                mat_GP,df_overlap_updated = Gene_scores(ad.read_h5ad(gene_anno_file),#peaks=None,
                    #adatatype=True,
                    #return_dataframe=True,
                    #n_batch=100,
                    tss_upstream=1e5,
                    tss_downsteam=1e5,
                    gb_upstream=5000,
                    cutoff_weight=1,
                    use_top_pcs=use_top_pcs,
                    use_precomputed=False,
                    use_gene_weigt=True,
                    min_w=1,
                    max_w=5)
                
                adata_peak = ad.read_h5ad(peak_file)
                gene_anno = pd.read_csv(gene_anno_file,
                                    encoding='utf8',
                                    sep='\t',
                                    header=None,
                                    names=['chr', 'start', 'end',
                                        'symbol', 'strand'])

                adata_mt = ad.AnnData(mat_GP,var=adata_peak.var.copy())
                adata_mt.obs_names = gene_anno.symbol.to_list()
                adata_mt.write_h5ad("Peak2Gene_matrix.h5ad")
                
                gene_size = gene_anno['end'] - gene_anno['start']
                w = 1/gene_size
                w_scaled = (5-1) * (w-min(w)) / (max(w)-min(w)) + 1
                gene_scores = adata_peak.X * mat_GP.T.multiply(w_scaled)
                adata_CG_atac = ad.AnnData(gene_scores,obs=adata_peak.obs.copy())
                adata_CG_atac.var_names = gene_anno.symbol.to_list()
                gs_count_mat=adata_CG_atac.X
                adata_CG_atac.write_h5ad("GeneScores.h5ad")

                self.GS_file = "Peak2Gene_matrix.h5ad"
                self.gs_count_file = "GeneScores.h5ad"
                self.gene_anno_file = gene_anno_file
                self.peak_file =peak_file

        # read data
        if gs_count_file.endswith(".h5"):
            gs_count = read_10X_h5(gs_count_file)
            gs_count_mat = gs_count.matrix
            gs_count_genes = gs_count.names.tolist()
            gs_count_samples = gs_count.barcodes.tolist()
            if type(gs_count_genes[0]) == bytes:
                gs_count_genes = [i.decode() for i in gs_count_genes]
            if type(gs_count_samples[0]) == bytes:
                gs_count_samples = [i.decode() for i in gs_count_samples]
        elif gs_count_file.endswith(".h5ad"):
            gs_count = read_10X_h5ad(gs_count_file)
            gs_count_mat = gs_count.matrix
            gs_count_genes = gs_count.names.tolist()
            gs_count_samples = gs_count.barcodes.tolist()
            if type(gs_count_genes[0]) == bytes:
                gs_count_genes = [i.decode() for i in gs_count_genes]
            if type(gs_count_samples[0]) == bytes:
                gs_count_samples = [i.decode() for i in gs_count_samples]
        else:
            gs_count = pd.read_table(gs_count_file, sep="\t", index_col=0)
            gs_count_mat = gs_count.values
            gs_count_mat = csc_matrix(gs_count_mat, dtype=np.float32)
            gs_count_genes = gs_count.index.to_list()
            gs_count_samples = gs_count.columns.to_list()


        if len(gs_count_genes) != gs_count_mat.shape[0]:
            gs_count_mat = gs_count_mat.transpose()
        # filter the count matrix
        # remove negative values (can be directly removed in csr_matrix without using applymap)
        gs_count_mat.data[gs_count_mat.data < 0] = 0

        # remove 0 values by filtering out columns where the sum of each column is zero
        count_per_sample = np.array(gs_count_mat.sum(axis=0)).flatten()
        non_zero_columns_idx = np.where(count_per_sample != 0)[0]
        gs_count_mat = gs_count_mat[:, non_zero_columns_idx]
        count_per_sample = count_per_sample[non_zero_columns_idx]

        self.mix_raw = gs_count_mat
        self.gs_genes = gs_count_genes
        self.gs_samples = gs_count_samples
        self.batch_info = batch_info
        self.GS_file = GS_file
        self.peak_file =peak_file

        if self.mix_raw.shape[0] != len(set(self.gs_genes)):
            tmp_df = copy.deepcopy(self.mix_raw.todense())
            idx_name = self.gs_genes
            tmp_df = pd.DataFrame(tmp_df, index=idx_name)
            tmp_df['symbol'] = idx_name
            tmp_df = tmp_df.dropna(subset=["symbol"])
            self.mix_raw = csr_matrix(tmp_df.groupby("symbol").median().values) # Take median value for duplication rows
            self.gs_genes = tmp_df.groupby("symbol").median().index

        logger.info('original: {}'.format(self.mix_raw.shape))

    def preprocessing(self,linear2log=True,log2linear=False,
                      do_drop=True,do_batch_norm=False,do_norm=True,do_quantile=False):
        if self.target_df is None:
            data_ = copy.deepcopy(self.mix_raw)
        else:
            data_ = copy.deepcopy(self.target_df)
        
        if not np.issubdtype(data_.data.dtype, np.floating):
            data_.data = data_.data.astype(np.float32)  # 或 float64，根据需求
        
        # trimming
        if do_drop:
            df_c = copy.deepcopy(data_)
            df_c.data[df_c.data == 0] = np.nan  # 将 0 替换为 NaN
            data_ = csr_matrix(df_c)  # 去除全为 NaN 的行
            logger.info('trimming: {}'.format(data_.shape))

        # batch normalization
        if self.batch_info is None:
            do_batch_norm = False
        else:
            if do_batch_norm:
                # TODO: 11.03 改了utils，可能需要修改
                df_c = copy.deepcopy(data_)
                info = self.batch_info.loc[self.sc_cells] # sample selection

                replace_list = info["replace"].tolist()
                prep_list = info["prep_batch"].tolist()
                lane_list = info["lane_batch"].tolist()
                lst_batch = [replace_list,prep_list,lane_list]

                comb_df = multi_batch_norm(df_c,lst_batch,do_plots=False)
                comb_df.data = np.maximum(comb_df.data, 0)  # 将负值替换为 0
                data_ = csr_matrix(comb_df)
                logger.info('batch normalization: {}'.format(data_.shape))
        
        if do_norm:
            # TODO: 去掉transpose()
            df_c = copy.deepcopy(data_)
            count_per_cell = np.array(df_c.sum(axis=0)).flatten()
            print("count_per_cell shape:",count_per_cell.shape)
            sc_scale_factor = np.round(np.quantile(count_per_cell, 0.75)/1000, 0)*1000
            r,c = df_c.nonzero()
            count_per_cell_sp = csr_matrix(((1.0/count_per_cell)[c], (r,c)), shape=(df_c.shape))
            data_ = df_c.multiply(count_per_cell_sp)*sc_scale_factor

        # linear --> log2
        if linear2log:
            def log2_sparse(sparse_matrix):
                sparse_matrix.data = np.log2(sparse_matrix.data + 1)
                return sparse_matrix
                
            df_c = copy.deepcopy(data_)
            data_ = log2_sparse(df_c)
            logger.info('linear2log: {}'.format(data_.shape))

        # log2 --> linear
        if log2linear:
            df_c = copy.deepcopy(data_)
            df_c = np.where(df_c.data < 30, 2 ** df_c.data, 1073741824) # avoid overflow
            data_= csr_matrix(df_c)
            logger.info('log2linear: {}'.format(data_.shape))

        # quantile normalization
        if do_quantile:
            # gene x cells
            df_c = copy.deepcopy(data_)
            qn_df = quantile_sparse(df_c)
            data_ = csr_matrix(qn_df)
            logger.info('quantile normalization: {}'.format(data_.shape))

        # TODO: 加上了transpose()
        self.target_df = copy.deepcopy(data_.transpose())

    def gene_selection(self, gene_shared=None,gene2id=None):
        # Select genes based on the provided gene_shared list
        if gene_shared is not None:
            # Convert to list if necessary using more Pythonic syntax
            if not isinstance(self.gs_genes, list):
                self.gs_genes = self.gs_genes.to_list()
            
            # Create a set for faster lookups
            gs_genes_set = set(self.gs_genes)
            # Filter out genes not present in gs_genes to prevent ValueError
            valid_genes = [gene for gene in gene_shared if gene in gs_genes_set]
            
            # Create a dictionary for O(1) lookups instead of O(n) index calls
            gene_to_index = {gene: idx for idx, gene in enumerate(self.gs_genes)}
            use_gene_idx = [gene_to_index[gene] for gene in valid_genes]
            
            # TODO:self.target_df = self.mix_raw[:, use_gene_idx]-->self.target_df = self.mix_raw[use_gene_idx,:]
            self.target_df = self.mix_raw[use_gene_idx,:]
            self.use_genes = valid_genes
            self.other_genes = list(gs_genes_set - set(valid_genes))
            
            # Log if any genes were filtered out
            if len(valid_genes) < len(gene_shared):
                filtered_count = len(gene_shared) - len(valid_genes)
                logger.info(f'Filtered out {filtered_count} genes not found in gs_genes')
        else:
            self.target_df = self.mix_raw
            self.use_genes = self.gs_genes
            self.other_genes = None
        #self.seed_topics = seed_topics
        self.gene2id = gene2id
        logger.info('sample selection: {}'.format(self.target_df.shape))
    def preprocessing_forBERT(self,gene_names):
        import scanpy as sc
        import anndata as ad
        # TODO:20260128
        self.mix_raw = self.mix_raw.transpose()
        # gene_embeddings = np.load(gene2vec_file)
        data = ad.AnnData(self.mix_raw,
                          var=pd.DataFrame(index=self.gs_genes),
                          obs=pd.DataFrame(index=self.gs_samples))
        data = data[:,data.var_names.isin(gene_names)].copy()
        indices = [index for index, element in enumerate(gene_names) \
                   if element in data.var_names]
        
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
        
        data_csr = csr_matrix((data.shape[0], len(gene_names)), dtype=np.float32) 
        data_csr[:, indices]  = np.array(data.X.todense())

        self.use_genes = self.gs_genes = gene_names
        self.target_df = data_csr
        self.target_int = copy.deepcopy(data_csr)
        self.target_int.data = np.floor(self.target_int.data).astype(int)

