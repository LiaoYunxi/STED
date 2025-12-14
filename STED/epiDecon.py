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

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger('epiDecon')

from .data import *
from .Genescore import *

#%%

def SpatialDeconvolveRaw(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    # read the celltype-topic file
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
    celltype_topic_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0).transpose()
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = np.divide(spot_celltype_array, np.array([spot_celltype_array.sum(axis=1)]).T)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return (spot_celltype_array_df, spot_celltype_array_norm_df)

def SpatialDeconvolveNorm(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    # read the celltype-topic file
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
    celltype_topic_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0).transpose()
    celltype_topic_norm_df = np.divide(celltype_topic_df, np.array([celltype_topic_df.sum(axis=0)]))
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_norm_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = np.divide(spot_celltype_array, np.array([spot_celltype_array.sum(axis=1)]).T)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return (spot_celltype_array_df, spot_celltype_array_norm_df)

def SpatialDeconvolveNormBySD(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    # read the celltype-topic file
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
    celltype_topic_df = pd.read_csv(topic_celltype_file, sep="\t", index_col=0).transpose()
    celltype_topic_norm_array = StandardScaler(with_mean=False).fit_transform(celltype_topic_df)
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_norm_array[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = np.divide(spot_celltype_array, np.array([spot_celltype_array.sum(axis=1)]).T)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return (spot_celltype_array_df, spot_celltype_array_norm_df)

def SpatialDeconvolveBayes(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    # read the celltype-topic file
    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
    celltype_topic_bayes_df = pd.read_csv(celltype_topic_file, sep="\t", index_col=0)
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_bayes_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_bayes_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = np.divide(spot_celltype_array, np.array([spot_celltype_array.sum(axis=1)]).T)
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_bayes_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return (spot_celltype_array_df, spot_celltype_array_norm_df)

def SpatialDeconvolveBayesNorm(st_count_spots, ntopics, topic_spot_mat, out_dir, model_dir):
    # read the celltype-topic file
    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
    celltype_topic_bayes_df = pd.read_csv(celltype_topic_file, sep="\t", index_col=0)
    celltype_topic_norm_df = np.divide(celltype_topic_bayes_df, np.array([celltype_topic_bayes_df.sum(axis=0)]))
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_norm_df.iloc[:, 0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_df = pd.DataFrame(spot_celltype_array)
    spot_celltype_array_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_df.index = st_count_spots
    spot_celltype_array_norm = np.divide(spot_celltype_array, np.array([spot_celltype_array.sum(axis=1)]).T)
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
        self.outname=None
    
    def SetData(self,out_dir,object_sc,object_gs,model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(out_dir,"model")
        self.out_dir = out_dir
        self.model_dir = model_dir
        self.sc_cells = object_sc.sc_cells
        self.ann_dict = object_sc.ann_dict
        
        self.target_df = object_gs.target_df
        self.gs_genes = object_gs.use_genes 
        self.gs_samples = object_gs.gs_samples
        
        if object_gs.mm_df is None:
            self.mm_df = object_gs.target_df
            # self.mm_int =object_gs.target_int
        else:
            self.mm_df = object_gs.mm_df
            # self.mm_int = object_gs.mm_int
    
    def Decon(self,ntopics_selection,seed_selection,outname=None,
              model=None,model_selection="LDA",benchmark=False):
        
        if model:
            self.norm_selection = model.norm_selection

        self.ntopics_selection = ntopics_selection
        self.seed_selection = seed_selection

        gene_topic_mat = pd.read_table(os.path.join(self.model_dir,"gene_topic_mat.txt"),sep="\t",index_col=0)
        self.genes = gene_topic_mat.index.to_list()
        self.topics = gene_topic_mat.columns
        topic_cell_mat = pd.read_table(os.path.join(self.model_dir,"topic_cell_mat.txt"),sep="\t",index_col=0)
        self.cells = topic_cell_mat.columns


        if outname is None:
            self.outname = "ntopics_"+str(self.ntopics_selection)+"_seed_"+str(self.seed_selection)
        else:
            self.outname = outname

        topic_sample_file = os.path.join(self.model_dir, "topic_sample_mat_"+self.outname+".txt")

        # if os.path.exists(topic_sample_file):
            # topic_sample_mat = scipy.sparse.load_npz(topic_sample_file)
            # topic_sample_mat = pd.read_table(topic_sample_file,sep="\t",index_col=0).values
        if model_selection=="LDA":
            try:
                lda = model.model.model
            except:
                if benchmark == False:
                    model_file =search_file_a(self.model_dir,"","lda_model")[0]
                    lda = LdaModel.load(model_file)
                else:
                    print("error: model is None.")
            #genes_dict_token2id = np.load(genes_dict_file,allow_pickle=True).item()
            genes_dict_file = os.path.join(self.out_dir,"LDA_Gene_dict.txt")
            genes_dict = Dictionary.load_from_text(genes_dict_file)
            genes_dict_token2id = genes_dict.token2id

            count_mat_use = self.mm_df.transpose()
            gs_corpus = []
            for i in range(count_mat_use.shape[1]):
                gs_genes_nonzero_index_corpus = [genes_dict_token2id[j] for j in self.gs_genes]
                gs_genes_nonzero_count = count_mat_use[:, i].toarray().flatten().tolist()
                gs_corpus.append(list(zip(gs_genes_nonzero_index_corpus, gs_genes_nonzero_count)))

            topic_sample = lda.get_document_topics(gs_corpus,minimum_probability=0)
            topic_sample_mat = gensim.matutils.corpus2csc(topic_sample)

        elif model_selection =="ALDA":
            self.mm_int = self.mm_df.copy()
            self.mm_int.data = np.floor(self.mm_int.data).astype(int)
            try:
                alda = model.model.model
            except:
                if benchmark == False:
                    model_file =search_file_a(self.model_dir,"","alda_model")[0]
                    f = open(model_file,'rb')
                    alda = pickle.load(f)
                    f.close()
                else:
                    print("error: model is None.")
            print("load seed_k from scTopic or scPreprocessing object.")
            with open(os.path.join(self.model_dir,'seed_topics.pkl'), 'rb') as f:
                seed_topics = pickle.load(f)
            f.close()

            seed_k =list(np.load(os.path.join(self.model_dir,'seed_k.npy')))

            alda.verbose = False
            sample_topic_mat = alda.transform(self.mm_int) #,seed_topics=seed_topics,initial_conf=1.0,seed_conf=1.0,other_conf=0.0,fix_seed_k=True,seed_k=seed_k

            topic_sample_mat = csc_matrix(sample_topic_mat.T)

        elif model_selection=="CorEx":
            try:
                corex = model.model.model
            except:
                if benchmark == False:
                    model_file =search_file_a(self.model_dir,"","corex_model")[0]
                    f = open(model_file,'rb')
                    corex = pickle.load(f)
                    f.close()
                else:
                    print("error: model is None.")

            corex_bulk = corex.fit(self.mm_df.A)
            topic_sample_mat = ss.csc_matrix(corex_bulk.p_y_given_x.transpose()) #the probability distribution over all topics
            topic_sample_logmat = ss.csc_matrix(corex_bulk.log_p_y_given_x.transpose()) #the log-probability distribution over all topics
            if benchmark ==False:
                # save the topic-bulk matrix
                topic_sample_file = os.path.join(self.model_dir, "topic_sample_mat.npz")
                topic_sample_logfile = os.path.join(self.model_dir, "topic_sample_logmat.npz")
                ss.save_npz(topic_sample_file, topic_sample_mat)
                ss.save_npz(topic_sample_logfile, topic_sample_logmat)

        
        if benchmark == False:
            topic_sample_df = pd.DataFrame(topic_sample_mat.todense(),index=self.topics,
                        columns=self.gs_samples)
            topic_sample_df.to_csv(topic_sample_file, sep="\t", index=True, header=True)
        self.topic_sample_mat =topic_sample_mat 

    def Bayes(self,norm_selection="Bayes"):
        if norm_selection is None:
            if self.norm_selection is None:
                norm_selection = "BayesNorm"
        else:
            self.norm_selection = norm_selection
            
        if self.norm_selection == "Raw":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveRaw(self.gs_samples, self.ntopics_selection,
                                                                                       self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "Norm":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveNorm(self.gs_samples, self.ntopics_selection,
                                                                                       self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "NormBySD":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveNormBySD(self.gs_samples, self.ntopics_selection,
                                                                                       self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "Bayes":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveBayes(self.gs_samples, self.ntopics_selection,
                                                                                       self.topic_sample_mat, self.out_dir, self.model_dir)
        if self.norm_selection == "BayesNorm":
            bulk_celltype_array_df, bulk_celltype_array_norm_df = SpatialDeconvolveBayesNorm(self.gs_samples, self.ntopics_selection,
                                                                                       self.topic_sample_mat, self.out_dir, self.model_dir)
        self.celltype_frac_df = bulk_celltype_array_norm_df

    def PeakPredict(self,sc_count_file,object_gs,GS_file=None,gene_anno_file=None,
                scATAC_count_file=None,groud_truth_file=None,
                ifscale = False,pseudo=False,benchmark=False):
        self.peak_file = object_gs.peak_file
        celltype_exp_libnorm,celltype_peak_weight_t = self._bulk_prepare(sc_count_file,object_gs,GS_file,gene_anno_file)
        if pseudo:
            adata_CP = ad.read_h5ad(self.peak_file)
            self.peak_meta = adata_CP.var
            celltype_index,celltype_peak_exp,celltype_peak_exp_scaled =self._pseudo_prepare(scATAC_count_file,celltype_exp_libnorm)
            if ifscale:
                celltype_peak_exp = None
                celltype_peak_exp_scaled =celltype_peak_exp_scaled.transpose()
            else:
                celltype_peak_exp_scaled = None
        else:
            celltype_index = celltype_exp_libnorm.index
            celltype_peak_exp = None
            celltype_peak_exp_scaled = None

        if self.celltype_frac_df.shape[0]>1:
            print("put only one sample for predicting peak signals.just predict first sample")
            celltype_frac_df = pd.DataFrame(self.celltype_frac_df.transpose().iloc[:,0])
        else:
            celltype_frac_df = self.celltype_frac_df.transpose()
        celltype_frac_df.columns = ['predict']  

        return self._predict_peak(celltype_peak_weight_t,celltype_index,celltype_frac_df = celltype_frac_df,
                           celltype_peak_exp = celltype_peak_exp,celltype_peak_exp_scaled = celltype_peak_exp_scaled,
                           groud_truth_file=groud_truth_file,pseudo=pseudo,benchmark=benchmark)

    def _bulk_prepare(self,sc_count_file,object_gs,GS_file=None,gene_anno_file=None):
        
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
                        encoding='utf8',sep='\t',header=None,
                        names=['chr', 'start', 'end','symbol', 'strand'])
                G2P = Gene2Peaks(gene_anno_file=gene_anno_file,peaks=self.peak_meta,cutoff_weight=0)
                df, mat_GP = G2P.gwt_gene_peak_match()
                adata_GS = ad.AnnData(mat_GP)
                row_names = self.peak_meta.index.to_list()
                adata_GS.obs_names = pd.Index(gene_anno.symbol)
                adata_GS.var_names = self.peak_meta.index
                print("compute gene peak weight matrix")

        from collections import defaultdict
        cell_dict = defaultdict(list)
        for k,v in self.ann_dict.items():
            cell_dict[v].append(k)

        adata = ad.read_h5ad(sc_count_file)
        use_genes = list(set(adata.var_names)&set(self.gs_genes))
        epid_data = copy.deepcopy(object_gs)
        epid_data.gene_selection(gene_shared=use_genes)
        adata = adata[:,use_genes].copy()
        mat = adata.X
        row_name = adata.obs_names.to_list()
        col_name = adata.var_names.to_list()

        dic_exp ={}
        for type_,names in cell_dict.items():
            tmp_mat = mat[[row_name.index(i) for i in names],:]
            tmp_exp = tmp_mat.sum(axis=0).flatten().tolist()[-1]
            dic_exp[type_] = tmp_exp
        celltype_exp = pd.DataFrame.from_dict(dic_exp, orient='index')
        celltype_exp.columns = col_name

        gs_scale_factor = celltype_exp.sum(axis=1)/10000
        dic_exp ={}
        for idx,row in celltype_exp.iterrows():
            dic_exp[idx] = row/gs_scale_factor[idx]
        celltype_exp_libnorm = pd.DataFrame.from_dict(dic_exp, orient='index')
        celltype_exp_libnorm.columns = col_name

       
        # TODO: check if the gene in adata_GS is the same as in celltype_exp_libnorm
        gene_select = list(set(celltype_exp_libnorm.columns)& set(adata_GS.obs_names))
        celltype_exp_libnorm = celltype_exp_libnorm[gene_select]
        ex = np.array(celltype_exp_libnorm)
        idx = [adata_GS.obs_names.to_list().index(i) for i in gene_select]
        adata_GS = adata_GS[idx,:]
        gs= adata_GS.X.toarray()

        # get gene score·scRNA-seq lib-normalized count
        Ep2 = np.dot(ex,gs)
        Ep2_df = pd.DataFrame(Ep2) 
        Ep2_df.columns = adata_GS.var_names
        Ep2_df.index = celltype_exp_libnorm.index

        gs_scale_factor = Ep2_df.sum(axis=0)/len(cell_dict.keys())
        dic_exp ={}
        for idx,row in Ep2_df.transpose().iterrows():
            if gs_scale_factor[idx] !=0:
                dic_exp[idx] = row/gs_scale_factor[idx]
            else:
                dic_exp[idx] = row
        # gene score·scRNA-seq lib-normalized count to celltype_peak_weight
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
        return celltype_exp_libnorm,celltype_peak_weight_t

    def _pseudo_prepare(self,scATAC_count_file,celltype_exp_libnorm):

        adata_sc_CP =ad.read_h5ad(scATAC_count_file)
        celltype_col = 'cell_type'
        celltype_col = 'main_cell_type'
        try:
            adata_sc_CP.var_names =[i.replace(':','_').replace('-','_') for i in adata_sc_CP.var_names]
        except:
            pass
        try:
            adata_sc_CP =adata_sc_CP[self.sc_cells,self.peak_meta.index]
            adata_sc_CP.obs.loc[:,celltype_col] = [v for k,v in self.ann_dict.items()]
        except:
            adata_sc_CP =adata_sc_CP[:,self.peak_meta.index]
        row_name = adata_sc_CP.obs_names.to_list()
        col_name = adata_sc_CP.var_names.to_list()

        dic_exp ={}
        for type_ in celltype_exp_libnorm.index:
            names = adata_sc_CP.obs_names[adata_sc_CP.obs[celltype_col]==type_].to_list()
            tmp_mat = adata_sc_CP.X[[row_name.index(i) for i in names],:]
            tmp_exp = tmp_mat.mean(axis=0).flatten().tolist()[-1] ### sum ###
            dic_exp[type_] = tmp_exp
    
        celltype_peak_exp = pd.DataFrame.from_dict(dic_exp, orient='index')
        celltype_peak_exp.columns =col_name
        celltype_peak_exp.index =celltype_exp_libnorm.index
            
        gs_scale_factor = np.round(celltype_peak_exp.sum(axis=0)/10000)
        dic_exp ={}
        for idx,row in celltype_peak_exp.transpose().iterrows():
            if gs_scale_factor[idx] !=0:
                dic_exp[idx] = row/gs_scale_factor[idx]
            else:
                dic_exp[idx] = row
        celltype_peak_exp_scaled = pd.DataFrame.from_dict(dic_exp, orient='index')

        celltype_index = celltype_exp_libnorm.index

        return celltype_index,celltype_peak_exp,celltype_peak_exp_scaled

    def _predict_peak(self, celltype_peak_weight_t, celltype_index, celltype_frac_df,
                      celltype_peak_exp=None, celltype_peak_exp_scaled=None,
                      groud_truth_file=None, pseudo=False, benchmark=False):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from math import sqrt
        import scipy.sparse

        # --- 加载 Peak 文件 ---
        if self.peak_file.endswith("h5ad"):
            adata_CP = ad.read_h5ad(self.peak_file)
        elif self.peak_file.endswith("bed"):
            peak_tmp = pd.read_table(self.peak_file, sep="\t", header=None)
            peak_tmp.columns = ["chr", "start", "end", "score"]
            peak_tmp.index = ["peak" + str(i) for i in range(peak_tmp.shape[0])]
            # 注意：这里构造时如果不转置，通常 X 是 (Samples, Features)
            # 这里的原始代码逻辑假定 peak_tmp['score'] 是一列，reshape后转置变成 (1, N_peaks)
            adata_CP = ad.AnnData(X=peak_tmp['score'].values.reshape(-1, 1).transpose(),
                                  obs=pd.DataFrame(index=["Bulk"]),
                                  var=peak_tmp[["chr", "start", "end"]])

        # --- 关键修复 1: 维度检查与转置 ---
        # self.var_names 来自 _bulk_prepare (基因-Peak映射)，是预期的 Peak 列表
        expected_peaks = len(self.var_names)
        
        # 检查 adata_CP 的列数 (var) 是否匹配
        if adata_CP.shape[1] != expected_peaks:
            print(f"检测到维度不匹配: adata_CP {adata_CP.shape}, 预期列数 {expected_peaks}")
            # 如果行数匹配，说明矩阵被转置了 (Peaks x Samples)
            if adata_CP.shape[0] == expected_peaks:
                print("正在转置 adata_CP 以匹配 (Samples x Peaks) ...")
                adata_CP = adata_CP.T
            else:
                raise ValueError(f"Peak文件的维度 {adata_CP.shape} 与预期的 Peak 数量 {expected_peaks} 不匹配。")

        # 尝试设置 var_names (如果原来的 var 不含 chr/start/end 会报错，这里保留原有 try-except)
        try:
            peak_name = adata_CP.var["chr"].astype(str) + "_" + \
                        adata_CP.var["start"].astype(str) + "_" + \
                        adata_CP.var["end"].astype(str)
            adata_CP.var_names = peak_name
        except:
            print("default peak name")

        self.peak_meta = adata_CP.var
        peak = adata_CP.X

        # --- 关键修复 2: 提取 Peak 向量并确保形状正确 ---
        # 我们只需要第一个样本(Bulk)的 Peak 信号，形状必须是 (1, N_peaks)
        
        # 提取第一行
        if peak.shape[0] >= 1:
            # 如果是稀疏矩阵，先切片再转 dense，确保 reshape 行为可控
            if scipy.sparse.issparse(peak):
                peak_vec = peak[0, :].toarray()  # 结果是 (1, N_peaks)
            else:
                peak_vec = peak[0, :].reshape(1, -1) # 结果是 (1, N_peaks)
        else:
             raise ValueError("Peak 数据为空")
             
        # 强制 peak 为 dense 矩阵用于点积 (避免 Pandas 构造时的稀疏兼容性问题)
        peak = peak_vec 

        # 再次检查维度
        if peak.shape[1] != expected_peaks:
             raise ValueError(f"处理后的 Peak 向量形状 {peak.shape} 仍不匹配预期列数 {expected_peaks}")

        res_df = celltype_frac_df
        
        # 读取真值文件
        if groud_truth_file is not None:
            gt = pd.read_table(groud_truth_file, sep='\t', index_col=0)
            res_df = res_df.loc[gt.index, :]
            res_df["gt"] = gt['Truth']
            self.gt = gt

        # 计算预测矩阵
        # per shape: (N_celltypes, 1)
        per = np.array(res_df['predict'].loc[celltype_index].tolist())
        per = np.expand_dims(per, axis=1)
        
        # Ep1 shape: (N_celltypes, 1) dot (1, N_peaks) -> (N_celltypes, N_peaks)
        Ep1 = np.dot(per, peak)
        
        # 这里不会再报错了，因为 Ep1 的列数现在保证是 expected_peaks
        Ep1_df = pd.DataFrame(Ep1, columns=self.var_names, index=celltype_index)

        # 过滤非零列
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

            # 计算相关性
            value_cor_dic = {
                idx: Ep_df_filter.loc[idx].corr(value)
                for idx, value in celltype_peak_exp_filter.iterrows()
            }
            final_p = pd.DataFrame.from_dict(value_cor_dic, orient='index').mean(0)[0]
            self.predicted_peak = Ep_df_filter

            if benchmark:
                cor_res = res_df.corr().iloc[0, 1]
                rmse_frac = sqrt(mean_squared_error(res_df["gt"], res_df['predict']))

                # 计算性能指标
                y_predict = Ep_df_filter.sum(0).to_numpy()
                y_test = celltype_peak_exp_filter.sum(0).to_numpy()
                
                # 注意：adata_CP 可能已经被转置，这里取值要小心
                # 此时 adata_CP.X 是 (Samples, Peaks)，直接切片即可
                if scipy.sparse.issparse(adata_CP.X):
                    y_bulk_test = adata_CP.X[:, nonzero_cols].toarray().flatten()
                else:
                    y_bulk_test = adata_CP.X[:, nonzero_cols].flatten()

                # 再次检查 NaN
                assert not np.isnan(y_predict).any(), "y_predict 含 NaN"
                assert not np.isnan(y_test).any(), "y_test 含 NaN"
                assert not np.isnan(y_bulk_test).any(), "y_bulk_test 含 NaN"

                # 计算指标
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