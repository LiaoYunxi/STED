""" Topic Model
References
----------
Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
doi:10.1073/pnas.0307752101.

Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
Priors Matter." In Advances in Neural Information Processing Systems 22,
edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
Culotta, 1973–1981, 2009.

Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
doi:10.1007/978-3-642-05224-8_6.
"""
import os
import gc
import random
from tqdm.auto import tqdm
import copy
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as ss
from collections import defaultdict,Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from scipy.cluster import hierarchy as sch
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from typing import Optional, Union, Tuple,List,Any, Callable
from itertools import combinations

import gensim
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

import corextopic.corextopic as ct
import corextopic.vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice
import textwrap
from networkx.readwrite import json_graph
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx  
from networkx.drawing.nx_agraph import graphviz_layout
import plotly.graph_objects as go
import plotly.figure_factory as ff

#matplotlib.use('TkAgg')  # 'Qt5Agg', 'GTK3Agg'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# from .gldadec import _lda_basic
from .guidedlda.guidedlda import GuidedLDA
from .Normalize import *
from .plot_utils import *

import logging
logger = logging.getLogger('Model')

#%%
def get_top_words(topic_word, top_n=30):
    if isinstance(topic_word, ss.spmatrix):
        sparse_mat = topic_word.T
        num_topics = sparse_mat.shape[1]
        top_indices = np.full((num_topics, top_n), -1, dtype=int)  # 初始化为 -1

        # 遍历每列
        for col_idx in range(num_topics):
            # 提取每列非零元素的索引和值
            col_data = sparse_mat.getcol(col_idx)
            nonzero_indices = col_data.nonzero()[0]
            nonzero_values = col_data.data

            # 检查非零元素的数量
            if len(nonzero_indices) > 0:
                # 获取最大 top_n 的非零索引
                sorted_nonzero_indices = nonzero_indices[np.argsort(nonzero_values)[-top_n:]]
                # 填充到 top_indices
                top_indices[col_idx, :len(sorted_nonzero_indices)] = sorted_nonzero_indices

        return top_indices  # 返回二维数组

    else:  # 处理稠密矩阵
        return np.argsort(topic_word, axis=1)[:, -top_n:]

def compute_coherence(articles,topics_top_genes, genes, epsilon=1e-12):
    coherence_scores = []
    for words in topics_top_genes:
        score = 0
        count = 0

        for w1, w2 in combinations(words, 2):
            co_occurrence_count = sum(1 for doc in genes if w1 in doc and w2 in doc)
            w2_count = sum(1 for doc in genes if w2 in doc)
            score += np.log((co_occurrence_count + epsilon) / (w2_count + epsilon))
            count += 1
        coherence_scores.append(score / count if count > 0 else 0)
    return coherence_scores

def Modelselection(res_dict):
    max_value = float('-inf')  # 初始化为负无穷
    max_key = None
    max_index = None

    for key, values in res_dict.items():
        for idx, value in enumerate(values):
            if value > max_value:
                max_value = value
                max_key = key
                max_index = idx
    return max_index,max_key,max_value

def RunBayesEvaluate(topic_cell_mat, topic_celltype_df,ann_list, celltype_num_dict, model_dir):
    bayes_res = ModelEvaluateBayes(topic_cell_mat, topic_celltype_df, ann_list, celltype_num_dict, model_dir)
    celltype_topic_bayes_df = bayes_res["celltype_topic_bayes_df"]
    bayesnorm_res = ModelEvaluateBayesNorm(topic_cell_mat, celltype_topic_bayes_df, ann_list)
    if bayesnorm_res["accuracy"] >= bayes_res["accuracy"]:
        v = bayesnorm_res["accuracy"]
        model_norm = "BayesNorm"
        celltype_prediction = bayesnorm_res["celltype_prediction"]
    else:
        v = bayes_res["accuracy"]
        model_norm = "Bayes"
        celltype_prediction = bayes_res["celltype_prediction"]
    return bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction

def RunCoherenceEvaluate(indicator_selection,lda,sc_corpus,genes,topic_cell_mat, topic_celltype_df,ann_list):
    if indicator_selection == "nmi":
                            v_dict = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, ann_list)
                            v = v_dict[["nmi"]]
    elif indicator_selection == "umass_coherence":
        cm = CoherenceModel(model = lda, corpus = sc_corpus, coherence='u_mass')
        v = cm.get_coherence()
    elif indicator_selection == "cv_coherence":
        cm = CoherenceModel(model = lda, corpus = sc_corpus, texts = genes, coherence='c_v')
        v = cm.get_coherence()
    return v

#%%
class AnchoredLDA():
    def __init__(self):
        self.sc_corpus = None
        self.genes = None
        self.cells = None
        self.celltypes = None
        self.write = False
    def setData(self,sc_corpus, genes, sc_count_cells, ann_list, anchor_dict,out_dir,
                alpha=None,eta=None, refresh=20,seed_confidence=0.15):
        self.sc_corpus = sc_corpus.tocsc()
        self.genes = copy.deepcopy(genes)
        self.cells = sc_count_cells
        self.ann_list = ann_list
        self.anchor_dict = anchor_dict
        self.out_dir = out_dir
        self.alpha = alpha
        self.eta = eta
        self.refresh =refresh
        self.seed_conf = seed_confidence
    
    def benchmark(self,model_dir,ntopics_list,seed_topics,gene2id,n_ensemble=1,n_iter=100):
        accuracy_bayes_dict = defaultdict(list)
        accuracy_bayesnorm_dict = defaultdict(list)
        accuracy_lg_dict = defaultdict(list)
        umass_coherence_dict = defaultdict(list)
        cv_coherence_dict = defaultdict(list)
        nmi_dict = defaultdict(list)
        # final_ll_dict = defaultdict(list)
        perplexity_dict = defaultdict(list)

        deconv_df = self.sc_corpus.toarray().astype('int64')

        for ntopics in ntopics_list:
            for i in tqdm(range(n_ensemble)):
                random.seed(i+1)
                if self.alpha is None:
                    alpha = 1/ntopics
                if self.eta is None:
                    eta = 1/ntopics

                alda = GuidedLDA(n_topics=ntopics, n_iter=n_iter, alpha=alpha,eta=eta,random_state=i+1, refresh=self.refresh)
                alda.fit(deconv_df, seed_topics=seed_topics, seed_confidence=self.seed_conf)

                target_res = alda.doc_topic_
                res = pd.DataFrame(target_res,index = self.cells)
                gc_df = pd.DataFrame(alda.word_topic_,index = self.genes) # (gene, topic)
                gc_df.columns = res.columns = [f"Topic{i}" for i in range(ntopics)]

                df = res.transpose()
                # z_res = standardz_sample(res) 
                row_sums = df.sum(axis=1)
                df_normalized = df.div(row_sums, axis=0)

                topic_cell_mat = df_normalized.values
                gene_topic_mat = gc_df.values
                gene_topic_mat_list = gene_topic_mat.tolist()

                # convert topic_cell_mat to topic_celltype_mat
                celltype_topic_dict = {}
                celltype_num_dict = {}
                celltypes = sorted(list(set(self.ann_list)))
                for celltype in celltypes:
                    celltype_topic_dict[celltype] = [0]*ntopics
                    celltype_num_dict[celltype] = 0
                for i in range(topic_cell_mat.shape[1]):
                    cell_celltype = self.ann_list[i]
                    celltype_topic_dict[cell_celltype] = [celltype_topic_dict[cell_celltype][j] + topic_cell_mat[j,i] for j in range(topic_cell_mat.shape[0])]
                    celltype_num_dict[cell_celltype] = celltype_num_dict[cell_celltype] + 1
                celltype_topic_mean_dict = {}

                for celltype in celltypes:
                    celltype_topic_mean_dict[celltype] = [i/celltype_num_dict[celltype] for i in celltype_topic_dict[celltype]]
                topic_celltype_df = pd.DataFrame(data = celltype_topic_mean_dict)

                raw_res = ModelEvaluateRaw(csc_matrix(topic_cell_mat), topic_celltype_df, self.ann_list)
                nmi_dict[ntopics].append(raw_res["nmi"])

                cvc = self.compute_cv_coherence(alda,gene2id)
                umc = alda.compute_umass_coherence(X=csc_matrix(deconv_df))
                pp = self.perplexity(alda,csc_matrix(deconv_df))

                umass_coherence_dict[ntopics].append(umc)
                cv_coherence_dict[ntopics].append(cvc)

                bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                    RunBayesEvaluate(csc_matrix(topic_cell_mat), topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])

                y=self.ann_list
                cell_topic_mat = topic_cell_mat.transpose()

                model_l = LogisticRegression(random_state=i+1)
                model_l.fit(cell_topic_mat,y=y)
                y_p = model_l.predict(cell_topic_mat)
                accuracy = accuracy_score(y, y_p)
                accuracy_lg_dict[ntopics].append(accuracy)
                perplexity_dict[ntopics].append(pp)

        metrics_dict = {"Topic": ntopics_list,
                        "lg_accuracy":accuracy_lg_dict,
                        "Bayes_accuracy": accuracy_bayes_dict,
                        "BayesNorm_accuracy": accuracy_bayesnorm_dict,
                        "umass_coherence": umass_coherence_dict,
                        "cv_coherence": cv_coherence_dict,
                        "nmi": nmi_dict,
                        "perplexity":perplexity_dict}
        self.benchmark_metrics_dict = metrics_dict
        self.model = alda
        gc.collect()

    def ll_plot(self, model):
        # 1. 获取数据
        if not hasattr(model, 'loglikelihoods_'):
            print("错误: 模型中没有 'loglikelihoods_' 属性，无法绘图。")
            return

        ll = model.loglikelihoods_
        # 确保 self.refresh 存在，如果不存在默认为 1
        refresh_rate = getattr(self, 'refresh', 1) 
        x = [i * refresh_rate for i in range(len(ll))]
        
        # 保存数据到实例变量 (保留原有逻辑)
        self.ll_plot_data = {'x': x, "y": ll}
                    
    def compute_cv_coherence(self,model,gene2id, top_n=30):
        dictionary = gene2id  
        topic_word_matrix = model.topic_word_ 
        
        top_words = get_top_words(topic_word_matrix, top_n)

        coherence_scores = []

        for words in top_words:
            word_vecs = []
            for w in words:
                if w in dictionary:  # 检查单词是否在字典中
                    word_index = dictionary[w]
                    word_vecs.append(topic_word_matrix[:, word_index])  # 提取单词向量

            if len(word_vecs) < 2:
                continue

            word_vecs = np.array(word_vecs)  # 转换为矩阵格式
            sim_matrix = cosine_similarity(word_vecs)

            upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            coherence_scores.append(np.mean(upper_triangle))

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def perplexity(self,model, X=None):
        """
        Perplexity is defined as exp(-1. * log-likelihood per word)
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.
        Returns
        -------
        score : float
            Perplexity score.
        """
        ll = model.loglikelihood()
        N = int(X.sum())
        return np.exp(-1* ll / N)
    
    def save(self,file):
        with open(file, 'wb') as file:
            pickle.dump(self.model, file)

    def conduct_train(self,model_dir,ntopics,seed_topics,n_iter=100,ll_plot=True,random_state =9):
        self.model_dir =model_dir
        print("Number of topics: %s" %(ntopics))
        if self.alpha is None:
            alpha = 1/ntopics
        if self.eta is None:
            eta = 1/ntopics
        # deconv_df = pd.DataFrame(self.sc_corpus.transpose().toarray(),index=self.genes,columns=self.cells)
        deconv_df = self.sc_corpus.toarray().astype('int64')

        random.seed(random_state)
        # re_order = random.sample(self.genes,len(self.genes)) # randomly sort the gene order
        # mm_target = deconv_df.loc[re_order]
        # conduct deconvolution
        print(f"n_topics={ntopics}, n_iter={n_iter}, alpha={alpha},eta={eta},random_state={random_state}, refresh={self.refresh}")
        alda = GuidedLDA(n_topics=ntopics, n_iter=n_iter, alpha=alpha,eta=eta,random_state=random_state, refresh=self.refresh)
        print(f"seed_topics={seed_topics}, seed_confidence={self.seed_conf}")
        print(f"deconv_df.shape={deconv_df.shape}")
        alda.fit(deconv_df, seed_topics=seed_topics, seed_confidence=self.seed_conf)

        target_res = alda.doc_topic_
        res = pd.DataFrame(target_res,index = self.cells)
        gc_df = pd.DataFrame(alda.word_topic_,index = self.genes) # (gene, topic)
        gc_df.columns = res.columns = [f"Topic{i}" for i in range(ntopics)]

        df = res.transpose()
        row_sums = df.sum(axis=1)
        df_normalized = df.div(row_sums, axis=0)

        topic_cell_mat = df_normalized.values
        gene_topic_mat = gc_df.values
        gene_topic_mat_list = gene_topic_mat.tolist()

        # convert topic_cell_mat to topic_celltype_mat
        celltype_topic_dict = {}
        celltype_num_dict = {}
        celltypes = sorted(list(set(self.ann_list)))
        for celltype in celltypes:
            celltype_topic_dict[celltype] = [0]*ntopics
            celltype_num_dict[celltype] = 0
        for i in range(topic_cell_mat.shape[1]):
            cell_celltype = self.ann_list[i]
            celltype_topic_dict[cell_celltype] = [celltype_topic_dict[cell_celltype][j] + topic_cell_mat[j,i] for j in range(topic_cell_mat.shape[0])]
            celltype_num_dict[cell_celltype] = celltype_num_dict[cell_celltype] + 1
        celltype_topic_mean_dict = {}

        for celltype in celltypes:
            celltype_topic_mean_dict[celltype] = [i/celltype_num_dict[celltype] for i in celltype_topic_dict[celltype]]
        topic_celltype_df = pd.DataFrame(data = celltype_topic_mean_dict)

        bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
            RunBayesEvaluate(csc_matrix(topic_cell_mat), topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)

        
        if ll_plot:
            self.ll_plot(alda)
            ll_dict = self.ll_plot_data
            fig,ax = plt.subplots(figsize=(5,4),dpi=100)
            plt.plot(ll_dict["x"],ll_dict["y"],label="log-likelihood of selected model")
            plt.xlabel('iterations')
            plt.ylabel('log-likelihood')
            plt.savefig(os.path.join(model_dir,"LDA_log_likelihood_plot.pdf"))

        # save model
        model_file = os.path.join(model_dir, "alda_model")
        with open(model_file, 'wb') as file:
            pickle.dump(alda, file)

        # save the topic-cell matrix
        topic_cell_file = os.path.join(model_dir, "topic_cell_mat.npz")
        ss.save_npz(topic_cell_file, csr_matrix(topic_cell_mat))

        topic_cell_df = pd.DataFrame(topic_cell_mat,index = [f"Topic{i}" for i in range(ntopics)],columns = self.cells)
        # save the topic file
        topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat.txt" )
        topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)

        # save the gene-topic matrix
        gene_topic_file = os.path.join(model_dir, "gene_topic_mat.txt")
        gene_topic_out = open(gene_topic_file, "w")
        gene_topic_out.write("\t".join(["Topic%s" %i for i in range(1, ntopics + 1)]) + "\n")
        for i in range(len(gene_topic_mat_list)):
            gene_topic_out.write(self.genes[i] + "\t" + "\t".join([str(j) for j in gene_topic_mat_list[i]]) + "\n")
        gene_topic_out.close()

        # save topic_celltype file !!
        topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
        topic_celltype_df.to_csv(topic_celltype_file, sep="\t")
        
        celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
        celltype_topic_bayes_df.to_csv(celltype_topic_file, sep="\t")

        self.topic_cell_mat = topic_cell_mat
        self.topic_celltype_df = topic_celltype_df
        self.sc_celltype_prediction = celltype_prediction
        self.celltype_num_dict = celltype_num_dict
        self.norm_selection = model_norm
        self.model = alda
            
        gc.collect()

    def ensemble_train(self,model_dir,ntopics_list,seed_k,seed_topics,indicator_selection = "accuracy",n_ensemble=1,n_iter=100,ll_plot=True,benchmark=False):

        if indicator_selection!="accuracy":
            print("Anchored LDA only support accuracy as indicator")
            indicator_selection = "accuracy"

        accuracy_bayes_dict = defaultdict(list)
        accuracy_bayesnorm_dict = defaultdict(list)
        accuracy_lg_dict = defaultdict(list)
        umass_coherence_dict = defaultdict(list)
        cv_coherence_dict = defaultdict(list)
        nmi_dict = defaultdict(list)
        final_ll_dict = defaultdict(list)

        deconv_df = pd.DataFrame(self.sc_corpus.transpose().toarray(),index=self.genes,columns=self.cells)
        max_value = -float('inf')
        max_index = 1
        max_key = ntopics_list[0]
        for ntopics in ntopics_list:
            print("Number of topics: %s" %(ntopics))
            for i in tqdm(range(n_ensemble)):
                random.seed(i+1)
                re_order = random.sample(self.genes,len(self.genes)) # randomly sort the gene order
                mm_target = deconv_df.loc[re_order]
                # conduct deconvolution
                alda = aLDA(verbose=False)
                # All anchored genes are considered reliable
                alda.set_marker(marker_final_dic=self.anchor_dict,anchor_dic=self.anchor_dict)
                alda.marker_redefine()
                alda.set_random(random_sets=[i+1])
                alda.set_final_int(final_int=mm_target) # sample-wide norm and re-order

                alda.seed_processing(re_set=False,seed_k=seed_k,seed_topics=seed_topics)

                

                if self.alpha is None:
                    alpha = 1/ntopics
                if self.eta is None:
                    eta = 1/ntopics

                alda.train(ntopics,self.ann_list, n_iter=n_iter,alpha=alpha,eta=eta,random_state=i+1,refresh=self.refresh,ll_plot=ll_plot)
                
                topic_cell_mat,topic_celltype_df,celltype_num_dict,\
                    gene_topic_mat_list,genes,topic_names,\
                        model,cvc,umc=\
                      (csc_matrix(alda.topic_cell_mat),alda.topic_celltype_df,alda.celltype_num_dict,\
                       alda.gene_topic_mat_list,alda.genes,alda.topic_names,\
                        alda.model,alda.cv_coherence,alda.umass_coherence)
                
                y=self.ann_list
                cell_topic_mat = topic_cell_mat.T

                model_l = LogisticRegression(random_state=i+1)
                model_l.fit(cell_topic_mat,y=y)
                y_p = model_l.predict(cell_topic_mat)
                accuracy = accuracy_score(y, y_p)

                if ll_plot:
                    self.ll_plot(alda)
                    final_ll_dict[ntopics].append(alda.ll_plot_data["y"][-1])
                
                bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                            RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                
                raw_res = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, self.ann_list)
                nmi_dict[ntopics].append(raw_res["nmi"])
                accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])
                accuracy_lg_dict[ntopics].append(accuracy)
                umass_coherence_dict[ntopics].append(umc)
                cv_coherence_dict[ntopics].append(cvc)

                if max_value < umc:
                    self.write = True
                    max_index = i+1
                    max_key = ntopics
                    max_value= umc
                else:
                    self.write = False
                max_model_norm = "accuracy"

                if self.write:
                    print(f"write")
                    # save model
                    model_file = os.path.join(model_dir, "alda_model")
                    with open(model_file, 'wb') as file:
                        pickle.dump(model, file)
                        #model.save(model_file)

                    # save the topic-cell matrix
                    topic_cell_file = os.path.join(model_dir, "topic_cell_mat.npz")
                    ss.save_npz(topic_cell_file, topic_cell_mat)

                    topic_cell_df = pd.DataFrame(topic_cell_mat.todense(),index = ["Topic %s" %i for i in topic_names],columns = self.cells)
                    # save the topic file
                    topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat.txt" )
                    topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)

                    # save the gene-topic matrix
                    gene_topic_file = os.path.join(model_dir, "gene_topic_mat.txt")
                    gene_topic_out = open(gene_topic_file, "w")
                    gene_topic_out.write("\t".join(["Topic%s" %i for i in range(1, ntopics + 1)]) + "\n")
                    for i in range(len(gene_topic_mat_list)):
                        gene_topic_out.write(genes[i] + "\t" + "\t".join([str(j) for j in gene_topic_mat_list[i]]) + "\n")
                    gene_topic_out.close()

                    # save topic_celltype file !!
                    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
                    topic_celltype_df.to_csv(topic_celltype_file, sep="\t")
                    
                    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
                    celltype_topic_bayes_df.to_csv(celltype_topic_file, sep="\t")

                    # save single cell celltype prediction
                    # celltype_prediction_file = os.path.join(model_dir,"celltype_prediction.txt")
                    # celltype_prediction.to_csv(celltype_prediction_file, sep="\t")

                    self.topic_cell_mat = topic_cell_mat
                    self.topic_celltype_df = topic_celltype_df
                    self.sc_celltype_prediction = celltype_prediction
                    self.celltype_num_dict = celltype_num_dict
                    self.model = model
                    self.write = False

        # return results
        # print(f"!!### best model contain '{max_key}' topics, when seed is {max_index}, use {max_model_norm} Normlize method, and the max value is {max_value} ###!!")
        print(f"!!### best model contain '{max_key}' topics, when seed is {max_index}, and the max value is {max_value} ###!!")
        if benchmark:
            self.accuracy_bayes_dict = accuracy_bayes_dict
            self.accuracy_bayesnorm_dict = accuracy_bayesnorm_dict
            self.accuracy_lg_dict = accuracy_lg_dict
            self.nmi_dict =nmi_dict
            self.umass_coherence_dict = umass_coherence_dict
            self.cv_coherence_dict = cv_coherence_dict
            self.model = alda

        self.norm_selection = max_model_norm
        self.ntopics_selection = max_key
        self.seed_selection = max_index
        self.indicator_selection = indicator_selection
        self.model_dir = model_dir
        self.final_ll_dict = final_ll_dict

        if ll_plot:
            ll_dict = final_ll_dict[max_key][max_index-1]
            fig,ax = plt.subplots(figsize=(5,4),dpi=100)
            plt.plot(ll_dict["x"],ll_dict["y"],label="log-likelihood of selected model")
            plt.xlabel('iterations')
            plt.ylabel('log-likelihood')
            plt.savefig(os.path.join(model_dir,"LDA_log_likelihood_plot.png"))
        gc.collect()

    def Plot(self,ll_plot=True,var_plot=True):
        # log-likelihood
        if ll_plot:
            fig,ax = plt.subplots(figsize=(5,4),dpi=100)
            for k_,v_ in self.final_ll_dict.items():
                if k_ ==min(self.final_ll_dict.keys()):
                    x  = v_[0]["x"]
                for id1,d1 in enumerate(v_):
                    y = d1["y"]
                    plt.plot(x,y,label=f"ntopic:{k_}, {id1}") 

            plt.xlabel('iterations')
            plt.ylabel('log-likelihood')
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.savefig(os.path.join(self.model_dir,"all_log_likelihood_plot.png"))
            
            # total_elements = sum(len(v) for v in self.final_ll_dict.values())

            # var plot
        if var_plot:
            all_elements = [item for sublist in self.total_res_dict.values() for item in sublist]
            for cell in self.cells:
                try:
                    estimation_var(total_res=all_elements,cell=str(cell))
                except:
                    pass
        gc.collect()

    def hierarchy(self,bow_matrix):
        distance_function = lambda x: 1 - cosine_similarity(x)
        linkage_function = lambda x: sch.linkage(x, method="ward", optimal_ordering=True)
        gene_topic_file = os.path.join(self.model_dir, "gene_topic_mat.txt")
        gene_topic_mat = pd.read_table(gene_topic_file).values
        gene_topic_mat = np.nan_to_num(gene_topic_mat, nan=0.0)
        topic_gene_mat = gene_topic_mat.transpose()

        X = distance_function(topic_gene_mat)
        
        # 进行层次聚类
        Z = linkage_function(X)

        # 修正 Z 中的距离问题
        if len(Z[:, 2]) != len(np.unique(Z[:, 2])):
            Z[:, 2] = get_unique_distances(Z[:, 2])

        # 获取目标词汇和基因
        words = copy.deepcopy(self.genes)    # 假设这包含了词汇的名称/ID
        phi = self.model.get_topics()  # 形状: (num_topics, num_words)
        hier_topics = pd.DataFrame(columns=["Parent_ID", "Parent_Name","Topics",
                                            "Child_Left_ID","Child_Left_Name",
                                            "Child_Right_ID","Child_Right_Name"])
        
        top_10_indices_per_column = np.argpartition(self.model.word_topic_, -3, axis=0)[-3:]
        sorted_indices = np.argsort(-self.model.word_topic_[top_10_indices_per_column, np.arange(self.model.word_topic_.shape[1])], axis=0)
        top_10_indices_per_column = top_10_indices_per_column[sorted_indices, np.arange(self.model.word_topic_.shape[1])]
        str_array = np.empty(top_10_indices_per_column.shape, dtype=object)
        for idx, value in np.ndenumerate(top_10_indices_per_column):
            str_array[idx] = self.genes[value]

        for index in range(len(Z)):
            # Find clustered documents
            clusters = sch.fcluster(Z, t=Z[index][2], criterion="distance")
            nr_clusters = len(clusters)

            # Extract first topic we find to get the set of topics in a merged topic
            topic = None
            val = Z[index][0]
            while topic is None:
                if val - len(clusters) < 0:
                    topic = int(val)
                else:
                    val = Z[int(val - len(clusters))][0]
            clustered_topics = [i for i, x in enumerate(clusters) if x == clusters[topic]]

            # 计算文档数量
            num_documents = bow_matrix[clustered_topics].shape[0]
            # 计算每个词在多少文档中出现
            doc_freq = np.array(np.sum(bow_matrix[clustered_topics] > 0, axis=0)).flatten()
            idf = np.log((num_documents + 1) / (doc_freq + 1)) + 1  # 加1以避免除零错误

            c_tf_idf_list = []
            # 对每个主题
            for topic_idx in range(phi.shape[0]):  # 遍历每个主题
                # 计算该主题下每个词的 `c-tf-idf`
                topic_phi = phi[topic_idx, :]  # 该主题下每个词的概率
                topic_c_tf_idf = topic_phi * idf  # `c-tf-idf` = 主题-词概率 * IDF

                c_tf_idf_list.append(topic_c_tf_idf)

            # 将所有主题的 `c-tf-idf` 合并成一个矩阵
            c_tf_idf = csr_matrix(np.vstack(c_tf_idf_list))
            words_per_topic = extract_words_per_topic(words,list(range(X.shape[0])),c_tf_idf)# calculate_aspects=False

            # Extract parent's name and ID
            parent_id = index + len(clusters)
            parent_name = "_".join([x[0] for x in words_per_topic[0]][:5])

            # Extract child's name and ID
            Z_id = int(Z[index][0])

            child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_left_name = "_".join(str_array[int(Z_id),:3].tolist())
            else:
                child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

            # Extract child's name and ID
            Z_id = Z[index][1]
            child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_left_name = "_".join(str_array[int(Z_id),:3].tolist())
            else:
                child_right_name = hier_topics.iloc[int(child_right_id)].Parent_Name

            # Save results
            hier_topics.loc[len(hier_topics), :] = [
                parent_id,
                parent_name,
                clustered_topics,
                int(Z[index][0]),
                child_left_name,
                int(Z[index][1]),
                child_right_name,
            ]
        hier_topics["Distance"] = Z[:, 2]
        hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
        hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[
            ["Parent_ID", "Child_Left_ID", "Child_Right_ID"]
        ].astype(str)
        return hier_topics

    def plot_hierarchy(
        self,
        orientation="left",
        topics = None,
        title = "<b>Hierarchical Clustering</b>",
        width = 1000,
        height = 600,
        color_threshold= 1
    ) -> go.Figure:
        distance_function = lambda x: 1 - cosine_similarity(x)
        linkage_function = lambda x: sch.linkage(x, method="ward", optimal_ordering=True)

        gene_topic_file = os.path.join(self.model_dir, "gene_topic_mat.txt")
        gene_topic_mat = pd.read_table(gene_topic_file).values
        gene_topic_mat = np.nan_to_num(gene_topic_mat, nan=0.0)
        topic_gene_mat = gene_topic_mat.transpose()
        
        X = distance_function(topic_gene_mat)

        all_topics = topics= list(range(X.shape[0]))
        indices = np.array([all_topics.index(topic) for topic in topics])

        # wrap distance function to validate input and return a condensed distance matrix
        distance_function_viz = lambda x: validate_distance_matrix(distance_function(x), topic_gene_mat.shape[0])
        # Create dendogram
        fig = ff.create_dendrogram(
            topic_gene_mat,
            orientation=orientation,
            distfun=distance_function_viz,
            linkagefun=linkage_function,
            color_threshold=color_threshold,
        )

        axis = "yaxis" if orientation == "left" else "xaxis"

        top_10_indices_per_column = np.argpartition(self.model.word_topic_, -3, axis=0)[-3:]
        sorted_indices = np.argsort(-self.model.word_topic_[top_10_indices_per_column, np.arange(self.model.word_topic_.shape[1])], axis=0)
        top_10_indices_per_column = top_10_indices_per_column[sorted_indices, np.arange(self.model.word_topic_.shape[1])]
        str_array = np.empty(top_10_indices_per_column.shape, dtype=object)
        for idx, value in np.ndenumerate(top_10_indices_per_column):
            str_array[idx] = self.genes[value]

        new_labels = [
            str(topics[int(x)])+":"+" ".join(str_array[:,topics[int(x)]].tolist()) for x in fig.layout[axis]["ticktext"]
        ]

        # Stylize layout
        fig.update_layout(
            plot_bgcolor="#ECEFF1",
            template="plotly_white",
            title={
                "text": f"{title}",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=22, color="Black"),
            },
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        )

        # Stylize orientation
        if orientation == "left":
            fig.update_layout(
                height=200 + (15 * len(topics)),
                width=width,
                yaxis=dict(tickmode="array", ticktext=new_labels,side="left"),
            )

            # Fix empty space on the bottom of the graph
            y_max = max([trace["y"].max() + 5 for trace in fig["data"]])
            y_min = min([trace["y"].min() - 5 for trace in fig["data"]])
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))

        else:
            fig.update_layout(
                width=200 + (15 * len(topics)),
                height=height,
                xaxis=dict(tickmode="array", ticktext=new_labels),
            )


        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    marker_color="black",
                    hovertext=hovertext,
                    hoverinfo="text",
                    mode="markers",
                    showlegend=False,
                )
            )
        return fig

#%%
def LDA(i,gene2id,ntopics,genes,cells,cell_celltype_list,sc_corpus,n_iter,alpha,eta):
        random.seed(i+1)
        re_order = random.sample(range(len(genes)), k=len(genes))# randomly sort the gene order
        fake_docs = [[word] for word in gene2id.keys()]
        genes_dict = Dictionary(fake_docs)
        sc_corpus = gensim.matutils.Sparse2Corpus(sc_corpus[:,re_order].transpose())

        lda = LdaModel(corpus = sc_corpus, num_topics = ntopics, id2word = genes_dict,
                       alpha = alpha,eta = eta,
                       iterations=n_iter,random_state=i+1)

        topic_cell = lda.get_document_topics(sc_corpus)
        topic_cell_mat = gensim.matutils.corpus2csc(topic_cell)

        topic_gene_mat_list = lda.get_topics()
        topic_gene_mat = np.array(topic_gene_mat_list)
        gene_topic_mat = topic_gene_mat.transpose()
        gene_topic_mat_list = gene_topic_mat.tolist()

        # convert topic_cell_mat to topic_celltype_mat
        celltype_topic_dict = {}
        celltype_num_dict = {}
        celltypes = sorted(list(set(cell_celltype_list)))
        for celltype in celltypes:
            celltype_topic_dict[celltype] = [0]*ntopics
            celltype_num_dict[celltype] = 0
        for i in range(topic_cell_mat.shape[1]):
            cell_celltype = cell_celltype_list[i]
            celltype_topic_dict[cell_celltype] = [celltype_topic_dict[cell_celltype][j] + topic_cell_mat[j,i] for j in range(topic_cell_mat.shape[0])]
            celltype_num_dict[cell_celltype] = celltype_num_dict[cell_celltype] + 1
        celltype_topic_mean_dict = {}

        for celltype in celltypes:
            celltype_topic_mean_dict[celltype] = [i/celltype_num_dict[celltype] for i in celltype_topic_dict[celltype]]
        topic_celltype_df = pd.DataFrame(data = celltype_topic_mean_dict)
        return topic_cell_mat,topic_celltype_df,celltype_num_dict,gene_topic_mat_list,genes,sc_corpus,genes_dict,lda

class nonAnchoredLDA():
    def __init__(self):
        self.sc_corpus = None
        self.genes = None
        self.cells = None
        self.celltypes = None
        self.write = False
    def setData(self,sc_corpus, genes, sc_count_cells, ann_list, alpha,eta,out_dir):
        self.sc_corpus = sc_corpus.tocsc()
        self.genes = copy.deepcopy(genes)
        self.cells = sc_count_cells
        self.ann_list = ann_list
        self.alpha = alpha
        self.eta = eta
        self.out_dir = out_dir
        #original_order = copy.deepcopy(self.genes)

    def conduct_train(self,gene2id,model_dir,ntopics,n_iter=100,random_state =9):
        print("Number of topics: %s" %(ntopics))
            
        if self.alpha is None:
            alpha = 1/ntopics
        if self.eta is None:
            eta = 1/ntopics

        topic_cell_mat,topic_celltype_df,celltype_num_dict,\
        gene_topic_mat_list,genes,sc_corpus,genes_dict,lda = LDA(random_state-1,gene2id,ntopics,self.genes,self.cells,self.ann_list,self.sc_corpus,n_iter,alpha,eta)

        # save model
        model_file = os.path.join(model_dir, "lda_model")
        lda.save(model_file)

        # save the topic-cell matrix
        topic_cell_file = os.path.join(model_dir, "topic_cell_mat.npz")
        ss.save_npz(topic_cell_file, topic_cell_mat)

        topic_cell_df = pd.DataFrame(topic_cell_mat.todense(),index = ["Topic %s" %i for i in range(1, 1 + topic_cell_mat.shape[0])],columns = self.cells)

        # save the topic file
        topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat.txt" )
        topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)

        # save the gene-topic matrix
        gene_topic_file = os.path.join(model_dir, "gene_topic_mat.txt")
        gene_topic_out = open(gene_topic_file, "w")
        gene_topic_out.write("\t".join(["Topic%s" %i for i in range(1, ntopics + 1)]) + "\n")
        for i in range(len(gene_topic_mat_list)):
            gene_topic_out.write(genes[i] + "\t" + "\t".join([str(j) for j in gene_topic_mat_list[i]]) + "\n")
        gene_topic_out.close()

        # save topic_celltype file !!
        topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
        topic_celltype_df.to_csv(topic_celltype_file, sep="\t")

        genes_dict_file = os.path.join(self.out_dir, "LDA_Gene_dict.txt")
        genes_dict.save_as_text(genes_dict_file)

        bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                    RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
        self.norm_selection = model_norm

        celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
        celltype_topic_bayes_df.to_csv(celltype_topic_file, sep="\t")

        self.genes_dict = genes_dict
        self.topic_cell_mat = topic_cell_mat
        self.topic_celltype_df = topic_celltype_df
        self.celltype_num_dict = celltype_num_dict
        self.model = lda


    
    def ensemble_train(self,gene2id,model_dir,ntopics_list,indicator_selection = "accuracy",n_ensemble=10,n_iter=100,benchmark=False):
        if indicator_selection in ["accuracy","nmi","umass_coherence", "cv_coherence"]:
            accuracy_bayes_dict = defaultdict(list)
            accuracy_bayesnorm_dict = defaultdict(list)
            compare_dict = defaultdict(list)
        else:
            raise ValueError("indicator_selection only support accuracy, nmi, umass_coherence or cv_coherence.")
        model_norm = None
        celltype_prediction = None
        for ntopics in ntopics_list:
            print("Number of topics: %s" %(ntopics))
            if self.alpha is None:
                alpha = 1/ntopics
            if self.eta is None:
                eta = 1/ntopics
            for i in tqdm(range(n_ensemble)):
                       
                topic_cell_mat,topic_celltype_df,celltype_num_dict,\
                gene_topic_mat_list,genes,sc_corpus,genes_dict,lda = LDA(i,gene2id,ntopics,self.genes,self.cells,self.ann_list,self.sc_corpus,n_iter,alpha,eta)


                if i+1 ==1 and ntopics == min(ntopics_list):
                    self.write = True
                    max_key = ntopics
                    max_index = 1
                    bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,max_value,celltype_prediction = \
                            RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                    max_model_norm = model_norm

                    if indicator_selection!="accuracy":
                        v = RunCoherenceEvaluate(indicator_selection,lda,sc_corpus,genes,topic_cell_mat, topic_celltype_df,self.ann_list)
                        compare_dict[ntopics].append(v)
                        max_value =v
                    else:
                        accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                        accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])
                    print(f"init value {max_value}")

                else:
                    if indicator_selection!="accuracy":
                        v = RunCoherenceEvaluate(indicator_selection,lda,sc_corpus,genes,topic_cell_mat, topic_celltype_df,self.ann_list)
                        compare_dict[ntopics]=v
                    else:
                        bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                                RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                        accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                        accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])
                        
                    if max_value<v:
                        self.write = True
                        max_index = i+1
                        max_key = ntopics

                        if indicator_selection!="accuracy":
                            bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,_,celltype_prediction = \
                                RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                        
                        max_model_norm = model_norm
                        max_value=v
                    else:
                        self.write = False

                if self.write:
                    print(f"write")
                    # save model
                    model_file = os.path.join(model_dir, "lda_model")
                    lda.save(model_file)

                    # save the topic-cell matrix
                    topic_cell_file = os.path.join(model_dir, "topic_cell_mat.npz")
                    ss.save_npz(topic_cell_file, topic_cell_mat)

                    topic_cell_df = pd.DataFrame(topic_cell_mat.todense(),index = ["Topic %s" %i for i in range(1, 1 + topic_cell_mat.shape[0])],columns = self.cells)

                    # save the topic file
                    topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat.txt" )
                    topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)

                    # save the gene-topic matrix
                    gene_topic_file = os.path.join(model_dir, "gene_topic_mat.txt")
                    gene_topic_out = open(gene_topic_file, "w")
                    gene_topic_out.write("\t".join(["Topic%s" %i for i in range(1, ntopics + 1)]) + "\n")
                    for i in range(len(gene_topic_mat_list)):
                        gene_topic_out.write(genes[i] + "\t" + "\t".join([str(j) for j in gene_topic_mat_list[i]]) + "\n")
                    gene_topic_out.close()

                    # save topic_celltype file !!
                    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
                    topic_celltype_df.to_csv(topic_celltype_file, sep="\t")

                    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
                    celltype_topic_bayes_df.to_csv(celltype_topic_file, sep="\t")

                    # save single cell celltype prediction
                    # celltype_prediction_file = os.path.join(model_dir,"celltype_prediction.txt")
                    # celltype_prediction.to_csv(celltype_prediction_file, sep="\t")

                    genes_dict_file = os.path.join(self.out_dir, "LDA_Gene_dict.txt")
                    genes_dict.save_as_text(genes_dict_file)

                    self.genes_dict = genes_dict
                    self.topic_cell_mat = topic_cell_mat
                    self.topic_celltype_df = topic_celltype_df
                    self.sc_celltype_prediction = celltype_prediction
                    self.celltype_num_dict = celltype_num_dict
                    self.model = lda
                    self.write = False

        # return results
        print(f"!!### best model contain '{max_key}' topics, when seed is {max_index}, use {max_model_norm} Normlize method, and the max value is {max_value} ###!!")

        self.norm_selection = max_model_norm
        self.ntopics_selection = max_key
        self.seed_selection = max_index
        self.indicator_selection = indicator_selection
        if benchmark:
            self.accuracy_bayes_dict = accuracy_bayes_dict
            self.accuracy_bayesnorm_dict = accuracy_bayesnorm_dict
   
    def benchmark(self,gene2id,model_dir,ntopics_list,n_ensemble=10,n_iter=100):
        print("benchmark:will save all results!")
        accuracy_bayes_dict = defaultdict(list)
        accuracy_bayesnorm_dict = defaultdict(list)
        accuracy_lg_dict = defaultdict(list)
        umass_coherence_dict = defaultdict(list)
        cv_coherence_dict = defaultdict(list)
        nmi_dict = defaultdict(list)
        final_ll_dict = defaultdict(list)

        for ntopics in ntopics_list:
            print("Number of topics: %s" %(ntopics))
            if self.alpha is None:
                alpha = 1/ntopics
            if self.eta is None:
                eta = 1/ntopics
            for i in tqdm(range(n_ensemble)):
                topic_cell_mat,topic_celltype_df,celltype_num_dict,\
                gene_topic_mat_list,genes,sc_corpus,genes_dict,lda = LDA(i,gene2id,ntopics,self.genes,self.cells,self.ann_list,self.sc_corpus,n_iter,alpha,eta)

                raw_res = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, self.ann_list)
                nmi_dict[ntopics].append(raw_res["nmi"])

                cm = CoherenceModel(model = lda, corpus = sc_corpus, coherence='u_mass')
                umass_coherence_dict[ntopics].append(cm.get_coherence())
                cm = CoherenceModel(model = lda, corpus = sc_corpus, texts = genes, coherence='c_v')
                cv_coherence_dict[ntopics].append(cm.get_coherence())

                bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                    RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])

                y=self.ann_list
                cell_topic_mat = topic_cell_mat.T

                model_l = LogisticRegression(random_state=i+1)
                model_l.fit(cell_topic_mat,y=y)
                y_p = model_l.predict(cell_topic_mat)
                accuracy = accuracy_score(y, y_p)
                accuracy_lg_dict[ntopics].append(accuracy)

        metrics_dict = {"Topic": ntopics_list,
                        "lg_accuracy":accuracy_lg_dict,
                                        "Bayes_accuracy": accuracy_bayes_dict,
                                        "BayesNorm_accuracy": accuracy_bayesnorm_dict,
                                        "umass_coherence": umass_coherence_dict,
                                        "cv_coherence": cv_coherence_dict,
                                        "nmi": nmi_dict}
        self.benchmark_metrics_dict = metrics_dict
    def hierarchy(self,bow_matrix):
        distance_function = lambda x: 1 - cosine_similarity(x)
        linkage_function = lambda x: sch.linkage(x, method="ward", optimal_ordering=True)
        topic_gene_mat_list = self.model.get_topics()
        topic_gene_mat = np.array(topic_gene_mat_list)
        topic_gene_mat = np.nan_to_num(topic_gene_mat, nan=0.0)
        X = distance_function(topic_gene_mat)
        
        # 进行层次聚类
        Z = linkage_function(X)

        # 修正 Z 中的距离问题
        if len(Z[:, 2]) != len(np.unique(Z[:, 2])):
            Z[:, 2] = get_unique_distances(Z[:, 2])

        # 获取目标词汇和基因
        words = copy.deepcopy(self.genes)    # 假设这包含了词汇的名称/ID
        phi = self.model.get_topics()  # 形状: (num_topics, num_words)
        hier_topics = pd.DataFrame(columns=["Parent_ID", "Parent_Name","Topics",
                                            "Child_Left_ID","Child_Left_Name",
                                            "Child_Right_ID","Child_Right_Name"])

        for index in range(len(Z)):
            # Find clustered documents
            clusters = sch.fcluster(Z, t=Z[index][2], criterion="distance")
            nr_clusters = len(clusters)

            # Extract first topic we find to get the set of topics in a merged topic
            topic = None
            val = Z[index][0]
            while topic is None:
                if val - len(clusters) < 0:
                    topic = int(val)
                else:
                    val = Z[int(val - len(clusters))][0]
            clustered_topics = [i for i, x in enumerate(clusters) if x == clusters[topic]]

            # 计算文档数量
            num_documents = bow_matrix[clustered_topics].shape[0]
            # 计算每个词在多少文档中出现
            doc_freq = np.array(np.sum(bow_matrix[clustered_topics] > 0, axis=0)).flatten()
            idf = np.log((num_documents + 1) / (doc_freq + 1)) + 1  # 加1以避免除零错误

            c_tf_idf_list = []
            # 对每个主题
            for topic_idx in range(phi.shape[0]):  # 遍历每个主题
                # 计算该主题下每个词的 `c-tf-idf`
                topic_phi = phi[topic_idx, :]  # 该主题下每个词的概率
                topic_c_tf_idf = topic_phi * idf  # `c-tf-idf` = 主题-词概率 * IDF

                c_tf_idf_list.append(topic_c_tf_idf)

            # 将所有主题的 `c-tf-idf` 合并成一个矩阵
            c_tf_idf = csr_matrix(np.vstack(c_tf_idf_list))
            words_per_topic = extract_words_per_topic(words,list(range(X.shape[0])),c_tf_idf)# calculate_aspects=False

            # Extract parent's name and ID
            parent_id = index + len(clusters)
            parent_name = "_".join([x[0] for x in words_per_topic[0]][:5])

            # Extract child's name and ID
            Z_id = int(Z[index][0])

            child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_left_name = "_".join([self.genes_dict.id2token[i[0]] for i in self.model.get_topic_terms(int(Z_id),5)])
            else:
                child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

            # Extract child's name and ID
            Z_id = Z[index][1]
            child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_right_name = "_".join([self.genes_dict.id2token[i[0]] for i in self.model.get_topic_terms(int(Z_id),5)])
            else:
                child_right_name = hier_topics.iloc[int(child_right_id)].Parent_Name

            # Save results
            hier_topics.loc[len(hier_topics), :] = [
                parent_id,
                parent_name,
                clustered_topics,
                int(Z[index][0]),
                child_left_name,
                int(Z[index][1]),
                child_right_name,
            ]
        hier_topics["Distance"] = Z[:, 2]
        hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
        hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[
            ["Parent_ID", "Child_Left_ID", "Child_Right_ID"]
        ].astype(str)
        return hier_topics

    def plot_hierarchy(
        self,
        orientation="left",
        topics = None,
        top_n_topics = None,
        title = "<b>Hierarchical Clustering</b>",
        width = 1000,
        height = 600,
        color_threshold= 1
    ) -> go.Figure:
        distance_function = lambda x: 1 - cosine_similarity(x)
        linkage_function = lambda x: sch.linkage(x, method="ward", optimal_ordering=True)
        topic_gene_mat_list = self.model.get_topics()
        topic_gene_mat = np.array(topic_gene_mat_list)
        topic_gene_mat = np.nan_to_num(topic_gene_mat, nan=0.0)
        gene_topic_mat = topic_gene_mat.transpose()
        
        X = distance_function(topic_gene_mat)

        all_topics = topics= list(range(X.shape[0]))
        indices = np.array([all_topics.index(topic) for topic in topics])

        # wrap distance function to validate input and return a condensed distance matrix
        distance_function_viz = lambda x: validate_distance_matrix(distance_function(x), topic_gene_mat.shape[0])
        # Create dendogram
        fig = ff.create_dendrogram(
            topic_gene_mat,
            orientation=orientation,
            distfun=distance_function_viz,
            linkagefun=linkage_function,
            color_threshold=color_threshold,
        )

        axis = "yaxis" if orientation == "left" else "xaxis"
        new_labels = [
            str(topics[int(x)])+":"+" ".join([self.genes_dict.id2token[i[0]] for i in self.model.get_topic_terms(topics[int(x)],3)]) for x in fig.layout[axis]["ticktext"]
        ]

        # Stylize layout
        fig.update_layout(
            plot_bgcolor="#ECEFF1",
            template="plotly_white",
            title={
                "text": f"{title}",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=22, color="Black"),
            },
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        )

        # Stylize orientation
        if orientation == "left":
            fig.update_layout(
                height=200 + (15 * len(topics)),
                width=width,
                yaxis=dict(tickmode="array", ticktext=new_labels,side="left"),
            )

            # Fix empty space on the bottom of the graph
            y_max = max([trace["y"].max() + 5 for trace in fig["data"]])
            y_min = min([trace["y"].min() - 5 for trace in fig["data"]])
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))

        else:
            fig.update_layout(
                width=200 + (15 * len(topics)),
                height=height,
                xaxis=dict(tickmode="array", ticktext=new_labels),
            )


        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    marker_color="black",
                    hovertext=hovertext,
                    hoverinfo="text",
                    mode="markers",
                    showlegend=False,
                )
            )
        return fig

#%%
def corex(i, ntopics, genes,cell_celltype_list, sc_corpus,n_iter,anchored_words,anchor_strength=2,log=False):
    random.seed(i+1)
    re_order = random.sample(range(len(genes)), k=len(genes))# randomly sort the gene order
    genes = [genes[i] for i in re_order]
    sc_corpus = sc_corpus[:,re_order]

    corex = ct.Corex(n_hidden=ntopics, words=genes, max_iter=n_iter, verbose=False, seed=i)

    if anchored_words is not None and len(anchored_words)<=ntopics:
        corex.fit(sc_corpus, words=genes, anchors=anchored_words, anchor_strength=anchor_strength)
    else:
        if anchored_words is not None:
            print("Warning: Too many anchor types. anchor_words will be none.")
        corex.fit(sc_corpus, words=genes)

    corex_cell_mat = ss.csc_matrix(corex.p_y_given_x.transpose()) #the probability distribution over all topics

    if log:
        corex_cell_mat = ss.csc_matrix(corex.log_p_y_given_x.transpose()) #the log-probability distribution over all topics

    corex_topic_gene_mat_list = corex.get_topics(n_words=len(genes), print_words=False)
    corex_topic_gene_mat = np.zeros((len(genes), ntopics))
    for j, tuples in enumerate(corex_topic_gene_mat_list):
        for i, value, _ in tuples:
            corex_topic_gene_mat[i, j] = value
    index2word = {v: k for k, v in corex.word2col_index.items()}
    genes = [index2word[i] for i in range(len(genes))]
    gene_topic_df = pd.DataFrame(corex_topic_gene_mat,index=genes,columns=["Topic%s" %i for i in range(1, ntopics + 1)])

    # convert topic_cell_mat to topic_celltype_mat
    corex_celltype_topic_dict = {}
    corex_celltype_num_dict = {}
    celltypes = sorted(list(set(cell_celltype_list)))

    for celltype in celltypes:
        corex_celltype_topic_dict[celltype] = [0]*ntopics
        corex_celltype_num_dict[celltype] = 0
    for i in range(corex_cell_mat.shape[1]):
        cell_celltype = cell_celltype_list[i]
        corex_celltype_topic_dict[cell_celltype] = [corex_celltype_topic_dict[cell_celltype][j] + corex_cell_mat[j,i] for j in range(corex_cell_mat.shape[0])]
        corex_celltype_num_dict[cell_celltype] = corex_celltype_num_dict[cell_celltype] + 1
        
    corex_celltype_topic_mean_dict = {}
    for celltype in celltypes:
        corex_celltype_topic_mean_dict[celltype] = [i/corex_celltype_num_dict[celltype] for i in corex_celltype_topic_dict[celltype]]
    corex_celltype_df = pd.DataFrame(data = corex_celltype_topic_mean_dict)
    return corex_cell_mat,corex_celltype_df,corex_celltype_num_dict,gene_topic_df,genes,sc_corpus,corex     

class CorEx():
    def __init__(self):
        self.sc_corpus = None
        self.genes = None
        self.cells = None
        self.celltypes = None
        self.write = False
    def setData(self,sc_corpus, genes, sc_count_cells, ann_list, anchor_list,out_dir):
        self.sc_corpus = sc_corpus.tocsc()
        self.genes = copy.deepcopy(genes)
        self.cells = sc_count_cells
        self.ann_list = ann_list
        self.anchor_list = anchor_list
        self.out_dir = out_dir
        #original_order = copy.deepcopy(self.genes)

    def ensemble_train(self,model_dir,ntopics_list,anchor_strength=2,indicator_selection = "accuracy",n_ensemble=10,n_iter=100,log=False):
        if indicator_selection!="accuracy":
            print("CorEx only support accuracy as indicator")
            indicator_selection = "accuracy"
        accuracy_bayes_dict = defaultdict(list)
        accuracy_bayesnorm_dict = defaultdict(list)
        model_norm = None
        celltype_prediction = None
        max_value = 0

        for ntopics in ntopics_list:
            print("Number of topics: %s" %(ntopics))
            for i in tqdm(range(n_ensemble)):
                
                topic_cell_mat,topic_celltype_df,celltype_num_dict,gene_topic_mat_list,genes,sc_corpus,model\
                      = corex(i,ntopics,self.genes,self.ann_list,self.sc_corpus,n_iter,self.anchor_list,anchor_strength,log)

                # top_words = get_top_words(topic_gene_mat, len(genes), genes)
                # coherence_scores = compute_coherence(genes, top_words)
                
                bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                            RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                
                accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])

                if max_value<v:
                    self.write = True
                    max_index = i+1
                    max_key = ntopics
                    
                    max_model_norm = model_norm
                    max_value=v
                else:
                    self.write = False

                if self.write:
                    print(f"write")
                    # save model
                    model_file = os.path.join(model_dir, "corex_model")
                    model.save(model_file, ensure_compatibility = True)

                    # save the topic-cell matrix
                    topic_cell_file = os.path.join(model_dir, "topic_cell_mat.npz")
                    ss.save_npz(topic_cell_file, topic_cell_mat)

                    topic_cell_df = pd.DataFrame(topic_cell_mat.todense(),index = ["Topic %s" %i for i in range(1, 1 + topic_cell_mat.shape[0])],columns = self.cells)
                    # save the topic file
                    topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat.txt" )
                    topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)

                    # save the gene-topic matrix
                    gene_topic_file = os.path.join(model_dir, "gene_topic_mat.txt")
                    gene_topic_out = open(gene_topic_file, "w")
                    gene_topic_out.write("\t".join(["Topic%s" %i for i in range(1, ntopics + 1)]) + "\n")
                    for i in range(len(gene_topic_mat_list)):
                        gene_topic_out.write(genes[i] + "\t" + "\t".join([str(j) for j in gene_topic_mat_list[i]]) + "\n")
                    gene_topic_out.close()

                    # save topic_celltype file !!
                    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
                    topic_celltype_df.to_csv(topic_celltype_file, sep="\t")

                    celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
                    celltype_topic_bayes_df.to_csv(celltype_topic_file, sep="\t")

                    # save single cell celltype prediction
                    # celltype_prediction_file = os.path.join(model_dir,"celltype_prediction.txt")
                    # celltype_prediction.to_csv(celltype_prediction_file, sep="\t")

                    self.topic_cell_mat = topic_cell_mat
                    self.topic_celltype_df = topic_celltype_df
                    self.sc_celltype_prediction = celltype_prediction
                    self.celltype_num_dict = celltype_num_dict
                    self.model = model
                    self.genes = genes
                    self.write = False

        # return results
        print(f"!!### best model contain '{max_key}' topics, when seed is {max_index}, use {max_model_norm} Normlize method, and the max value is {max_value} ###!!")

        self.norm_selection = max_model_norm
        self.ntopics_selection = max_key
        self.seed_selection = max_index
        self.indicator_selection = indicator_selection
        self.model_dir = model_dir
        

    def conduct_train(self,model_dir,ntopics,anchor_strength=2,n_iter=100,random_state =9,log=False):

        topic_cell_mat,topic_celltype_df,celltype_num_dict,gene_topic_df,genes,sc_corpus,model\
            = corex(random_state-1,ntopics,self.genes,self.ann_list,
                    self.sc_corpus,n_iter,self.anchor_list,anchor_strength,log)

        bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
            RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
        
        model_file = os.path.join(model_dir, "corex_model")
        model.save(model_file, ensure_compatibility = True)

        # save the topic-cell matrix
        topic_cell_file = os.path.join(model_dir, "topic_cell_mat.npz")
        ss.save_npz(topic_cell_file, topic_cell_mat)

        topic_cell_df = pd.DataFrame(topic_cell_mat.todense(),index = ["Topic %s" %i for i in range(1, 1 + topic_cell_mat.shape[0])],columns = self.cells)
        # save the topic file
        topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat.txt" )
        topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)

        # save the gene-topic matrix
        gene_topic_file = os.path.join(model_dir, "gene_topic_mat.txt")
        gene_topic_df.to_csv(gene_topic_file, sep = "\t", index = True, header = True)

        # save topic_celltype file !!
        topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat.txt")
        topic_celltype_df.to_csv(topic_celltype_file, sep="\t")

        celltype_topic_file = os.path.join(model_dir, "celltype_topic_mat_bayes.txt")
        celltype_topic_bayes_df.to_csv(celltype_topic_file, sep="\t")
    
        self.topic_cell_mat = topic_cell_mat
        self.topic_celltype_df = topic_celltype_df
        self.sc_celltype_prediction = celltype_prediction
        self.celltype_num_dict = celltype_num_dict
        self.model = model
        self.genes = genes
        self.model_dir = model_dir
        self.norm_selection = model_norm

    def benchmark(self,model_dir,ntopics_list,anchor_strength=2,n_ensemble=10,n_iter=100,log=False):
        accuracy_bayes_dict = defaultdict(list)
        accuracy_bayesnorm_dict = defaultdict(list)
        accuracy_lg_dict = defaultdict(list)
        umass_coherence_dict = defaultdict(list)
        cv_coherence_dict = defaultdict(list)
        nmi_dict = defaultdict(list)
        final_tcs_dict = defaultdict(list)
        for ntopics in ntopics_list:
            print("Number of topics: %s" %(ntopics))
            for i in tqdm(range(n_ensemble)):
                
                topic_cell_mat,topic_celltype_df,celltype_num_dict,topic_gene_mat,genes,sc_corpus,model\
                      = corex(i,ntopics,self.genes,self.ann_list,self.sc_corpus,n_iter,self.anchor_list,anchor_strength,log)
                final_tcs_dict[ntopics].append(model.tcs)

                # cvc = self.compute_cv_coherence(topic_gene_mat)
                # cv_coherence_dict[ntopics].append(cvc)
                # umc = self.compute_umass_coherence(topic_gene_mat)
                # umass_coherence_dict[ntopics].append(umc)

                raw_res = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, self.ann_list)
                nmi_dict[ntopics].append(raw_res["nmi"])

                bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                            RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                
                accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])

                y=self.ann_list
                cell_topic_mat = topic_cell_mat.T

                model_l = LogisticRegression(random_state=i+1)
                model_l.fit(cell_topic_mat,y=y)
                y_p = model_l.predict(cell_topic_mat)
                accuracy = accuracy_score(y, y_p)
                accuracy_lg_dict[ntopics].append(accuracy)

        self.accuracy_bayes_dict = accuracy_bayes_dict
        self.accuracy_bayesnorm_dict = accuracy_bayesnorm_dict
        self.accuracy_lg_dict = accuracy_lg_dict
        self.nmi_dict =nmi_dict
        self.umass_coherence_dict = umass_coherence_dict
        self.cv_coherence_dict = cv_coherence_dict
        self.final_tcs_dict = final_tcs_dict
        self.model = model
        self.model_dir = model_dir

    def hierarchy(self,n_hidden_list=[6,2],max_edges=200,plot = True,figfile=None):
        # Train sub layers to the topic model
        layer_list = [self.model]
        for n_hidden in n_hidden_list:
            tm_layer = ct.Corex(n_hidden=n_hidden)
            tm_layer.fit(layer_list[-1].labels);
            layer_list.append(tm_layer)
        
        self.topic_layers = layer_list

        if figfile is None:
            figfile = "CorEx_Topic_Hierarchy"
        G = vt.vis_hierarchy(layer_list, column_label=self.genes, max_edges=max_edges,prefix=figfile)

        edges = []
        for u, v, data in G.edges(data=True):
            parent = u
            child = v
            weight = data['weight']
            
            parent_id = parent[0]
            parent_name = f"Layer {parent_id}, Node {parent[1]}"
            child_id = child[0]
            child_name = f"Layer {child_id}, Node {child[1]}"
            
            edges.append([parent_id, parent_name, weight, child_id, child_name])
        
        if plot:
            fig = plot_tree_with_plotly(layer_list,G)
            save_figure(fig, os.path.join(self.model_dir,figfile+".pdf"))

            # plt.close('all')  
            # pos = graphviz_layout(G, prog="dot")  # 使用graphviz布局，适合树结构
            # nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='b',
            #         node_size=700, font_weight='bold', font_size=12,arrows=True)  
            # edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}  
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  
            # plt.title('CorEx Topic Hierarchy')  
            # plt.savefig(os.path.join(self.model_dir,figfile+".png"))
        return layer_list,G #pd.DataFrame(edges, columns=["Parent_ID", "Parent_Name", "Weight", "Child_ID", "Child_Name"])

    def tc_plot(self,figfile=None):
        if figfile is None:
            figfile = os.path.join(self.model_dir,"CorEx_Topic_Correlation.pdf")
        plt.figure(figsize=(10,5))
        plt.bar(range(self.model.tcs.shape[0]), self.model.tcs, color='#4e79a7', width=0.5)
        plt.xlabel('Topic', fontsize=16)
        plt.ylabel('Total Correlation (nats)', fontsize=16);
        plt.savefig(figfile)
    
    def compute_cv_coherence(self,topic_word_matrix, top_n=30):
        top_words = get_top_words(topic_word_matrix, top_n)

        coherence_scores = []

        for words in top_words:
            word_vecs = []
            for w in words:
                word_vecs.append(topic_word_matrix[:, w])  # 提取单词向量

            if len(word_vecs) < 2:
                continue

            word_vecs = np.array(word_vecs)  # 转换为矩阵格式
            sim_matrix = cosine_similarity(word_vecs)

            upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            coherence_scores.append(np.mean(upper_triangle))

        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def compute_umass_coherence(self, topic_word_matrix, top_n=30, epsilon=1e-12):
        # Get the top N words for each topic
        top_words = get_top_words(topic_word_matrix, top_n)

        # Convert document-term matrix to a binary matrix (word appears or not in each document)
        binary_matrix = (self.sc_corpus > 0).astype(int)

        # Calculate word document frequency (number of documents each word appears in)
        word_doc_freq = binary_matrix.sum(axis=0)

        # Calculate pairwise document frequency (number of documents each word pair co-occurs in)
        pair_doc_freq = Counter()
        for doc in binary_matrix:
            if isinstance(doc, ss.spmatrix):  # 如果是稀疏矩阵
                word_indices = doc.nonzero()[1]  # 获取非零列的索引
            else:
                word_indices = np.where(doc > 0)[0]  # 如果是普通数组
            for i in range(len(word_indices)):
                for j in range(i + 1, len(word_indices)):
                    pair_doc_freq[(word_indices[i], word_indices[j])] += 1
                    pair_doc_freq[(word_indices[j], word_indices[i])] += 1

        # Compute UMass coherence for each topic
        coherence_scores = []
        for words in top_words:
            score = 0
            if len(words) > 1:
                for i in range(1, len(words)):
                    for j in range(0, i):
                        # 检查索引范围
                        if words[j] >= len(word_doc_freq):
                            print(f"Warning: words[j]={words[j]} exceeds word_doc_freq size.")
                            continue
                        if words[i] >= len(word_doc_freq):
                            print(f"Warning: words[i]={words[i]} exceeds word_doc_freq size.")
                            continue

                        freq_wj = word_doc_freq[words[j]]
                        freq_wi_wj = pair_doc_freq.get((words[i], words[j]), 0)

                        # 防止频率为0的情况
                        if freq_wj == 0:
                            print(f"Warning: freq_wj is zero for words[j]={words[j]}")
                            continue
                        score += np.log((freq_wi_wj + epsilon) / (freq_wj + epsilon))
            coherence_scores.append(score)

        # Return the average coherence score
        return np.mean(coherence_scores) if coherence_scores else 0


        # top_n_words = max(self.top_n_words, 30)

    def plot_hierarchy(
        self,
        hier_topics,
        orientation="left",
        topics = None,
        top_n_topics = None,
        title = "<b>Hierarchical Clustering</b>",
        width = 1000,
        height = 600,
        color_threshold= 1
    ) -> go.Figure:
        distance_function = lambda x: squareform(pdist(x, metric='euclidean'))
        linkage_function = lambda x: sch.linkage(x, method="ward", optimal_ordering=True)
        gene_topic_file = os.path.join(self.model_dir, "gene_topic_mat.txt")
        gene_topic_mat = pd.read_table(gene_topic_file,index_col=0).values
        gene_topic_mat = np.nan_to_num(gene_topic_mat, nan=0.0)
        topic_gene_mat = gene_topic_mat.transpose()
        
        X = distance_function(topic_gene_mat)

        all_topics = topics= list(range(X.shape[0]))
        indices = np.array([all_topics.index(topic) for topic in topics])

        # wrap distance function to validate input and return a condensed distance matrix
        distance_function_viz = lambda x: validate_distance_matrix(distance_function(x), topic_gene_mat.shape[0])
        # Create dendogram
        # fig = ff.create_dendrogram(hier_topics, labels=hier_topics['Parent_Name'].values)
        fig = ff.create_dendrogram(
            topic_gene_mat,
            orientation=orientation,
            distfun=distance_function_viz,
            linkagefun=linkage_function,
            color_threshold=color_threshold,
        )

        axis = "yaxis" if orientation == "left" else "xaxis"
        new_labels = [
            str(topics[int(x)]+1)+":"+' '.join([n[0] for n in self.model.get_topics(topic=topics[int(x)],n_words=3)]) for x in fig.layout[axis]["ticktext"]
        ]
        # new_labels = [
        #     str(x)+":"+' '.join([n[0] for n in self.model.get_topics(topic=x,n_words=3)]) for x in fig.layout[axis]["ticktext"]
        # ]

        # Stylize layout
        fig.update_layout(
            plot_bgcolor="#ECEFF1",
            template="plotly_white",
            title={
                "text": f"{title}",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=22, color="Black"),
            },
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        )

        # Stylize orientation
        if orientation == "left":
            fig.update_layout(
                height=200 + (15 * len(topics)),
                width=width,
                yaxis=dict(tickmode="array", ticktext=new_labels,side="left"),
            )

            # Fix empty space on the bottom of the graph
            y_max = max([trace["y"].max() + 5 for trace in fig["data"]])
            y_min = min([trace["y"].min() - 5 for trace in fig["data"]])
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))

        else:
            fig.update_layout(
                width=200 + (15 * len(topics)),
                height=height,
                xaxis=dict(tickmode="array", ticktext=new_labels),
            )

        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    marker_color="black",
                    hovertext=hovertext,
                    hoverinfo="text",
                    mode="markers",
                    showlegend=False,
                )
            )
        return fig

def plot_tree_with_plotly(topic_layers, g, node_color='skyblue', edge_color='gray',
                          node_text_color='black', weight_range=(1, 5), show_label=True):
    """
    使用 Plotly 绘制旋转 90 度的树状图，并根据边的权重调整边的粗细。
    """
    # 使用 Graphviz 布局
    pos = graphviz_layout(g, prog="dot")
    
    # 提取边的坐标并动态调整线宽
    edge_traces = []
    raw_weights = [g.edges[edge].get('weight', 1) for edge in g.edges()]
    min_raw_weight, max_raw_weight = min(raw_weights), max(raw_weights)
    min_width, max_width = weight_range
    norm_weights = np.interp(raw_weights, (min_raw_weight, max_raw_weight), (min_width, max_width))
    
    for i, edge in enumerate(g.edges()):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[-y0, -y1, None],  # 交换 x 和 y
            y=[-x0, -x1, None],
            line=dict(width=norm_weights[i], color=edge_color),  # 动态线宽
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # 提取节点的坐标和标签
    node_x = []
    node_y = []
    node_text = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(-y)  # 交换 x 和 y
        node_y.append(-x)
        if node[0] == 0:
            label = f"{node[1]+1}: {' '.join([str(n[0]) for n in topic_layers[node[0]].get_topics(topic=node[1], n_words=3)])}"
        else:
            if show_label:
                label = g.nodes[node].get('label', str(node))  # 节点标签
            else:
                label = " "  # 空标签
        node_text.append(label)
    
    # 绘制节点
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_color,
            line=dict(width=1, color=node_color)
        ),
        text=node_text,
        textposition="middle left",  # 标签放在节点右侧
        textfont=dict(color=node_text_color)
    )
    
    # 创建 Plotly 图形
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title="Topic Hierarchical",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=10, l=10, r=10, t=50),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'  # 背景颜色
                    ))
    fig.update_layout(
        autosize=True,  # 自动调整画布大小
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[min(node_x) - 100, max(node_x) + 100]  # 根据节点位置动态调整范围
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[min(node_y) - 100, max(node_y) + 100]
        )
    )

    return fig
