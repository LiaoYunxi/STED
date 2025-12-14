import os
import numpy as np

from .utils import *
from .data import *
from .TopicModel import *
from .epiDecon import *
from .plot_utils import *

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

class scTopic():
    def __init__(self):
        self.genes = None
        self.cells = None
        self.ann_dict = None
        self.anchor_dict = None
        self.anchor_list = None
        self.target_df = None
        self.target_int = None
        self.mm_df = None
        self.ntopics_list = None
    def setData(self,object,out_dir,ntopics_list=None):
        # object is a scPreProcessing instance
        self.genes = object.use_genes
        self.cells = object.sc_cells
        self.ann_dict = object.ann_dict
        self.seed_k = object.seed_k
        self.seed_topics =object.seed_topics
        self.gene2id = object.gene2id

        if object.use_anchor_dict is not None:
            self.anchor_dict = object.use_anchor_dict
            self.anchor_list = object.use_anchor_list

        if object.mm_df is None:
            self.mm_df = object.target_df
        else:
            self.mm_df = object.mm_df

        self.target_int = object.target_int

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_dir = os.path.join(out_dir, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.out_dir = out_dir
        self.model_dir = model_dir

        cell_celltype_list = []
        for i in range(len(self.cells)):
            cell_celltype = self.ann_dict[self.cells[i]]
            cell_celltype_list.append(cell_celltype)
        celltypes = sorted(list(set(cell_celltype_list)))

        if ntopics_list is None:
            ntopics_list = list(range(len(celltypes), 3*len(celltypes)+1))
        
        self.ntopics_list = ntopics_list
        self.celltypes = celltypes
        self.ann_list = cell_celltype_list

    def trainLDA(self,indicator_selection="accuracy",seed=9,
                 alpha=None,eta=None,
                 refresh=30,seed_confidence=0.15,
                 n_iter=100,n_ensemble=0,ntopics=None,
                 ll_plot=False,benchmark =False,ifint=False):
        """_summary_
        Args:
            n_ensemble (int, optional): The number of ensembles. Defaults to 100.
            n_iter (int, optional): The number of iterations for each run. Defaults to 100.
            alpha (float, optional): A hyperparameter for Dirichlet distribution. Defaults to 1/ntopics.
            eta (float, optional): A hyperparameter for Dirichlet distribution.. Defaults to 1/ntopics.
            refresh (int, optional): _description_. Defaults to 10.
            ll_plot (bool, optional): _description_. Defaults to True.
            var_plot (bool, optional): _description_. Defaults to True.
        """
        # original_order = sorted(self.genes)
        if ifint:
            deconv_df = self.target_int
        else:
            deconv_df = self.mm_df

        if len(self.anchor_dict) < 1:
            print("Selecting the unsupervised LDA model.")
            NAlda =nonAnchoredLDA()
            NAlda.setData(deconv_df, self.genes, self.cells, self.ann_list, alpha,eta,self.out_dir)
            if n_ensemble<1:
                NAlda.conduct_train(self.gene2id,self.model_dir,ntopics,n_iter=n_iter,random_state =seed)
                self.model = NAlda
                self.genes_dict = NAlda.genes_dict
                self.topic_cell_mat = NAlda.topic_cell_mat
                self.topic_celltype_df = NAlda.topic_celltype_df
                self.celltype_num_dict = NAlda.celltype_num_dict
                self.norm_selection = NAlda.norm_selection
            else:
                if benchmark:
                    NAlda.benchmark(self.gene2id,self.model_dir,self.ntopics_list,n_ensemble,n_iter)
                    self.benchmark_metrics_dict = NAlda.benchmark_metrics_dict
                else:
                    NAlda.ensemble_train(self.gene2id,self.model_dir,self.ntopics_list,indicator_selection,n_ensemble,benchmark=benchmark)
                    self.genes_dict = NAlda.genes_dict
                    self.topic_cell_mat = NAlda.topic_cell_mat
                    self.topic_celltype_df = NAlda.topic_celltype_df
                    self.sc_celltype_prediction = NAlda.sc_celltype_prediction
                    self.celltype_num_dict = NAlda.celltype_num_dict
                    self.norm_selection = NAlda.norm_selection
                    self.ntopics_selection = NAlda.ntopics_selection
                    self.seed_selection = NAlda.seed_selection
                    self.indicator_selection = indicator_selection
                    
        else:
            print("Selecting the Anchored LDA model.")
            seed_topics = {}
            for k,v in self.seed_topics.items():
                seed_topics[k] = v[0]  
            with open(os.path.join(self.model_dir,'seed_topics.pkl'), 'wb') as f:
                pickle.dump(seed_topics, f)
            f.close()

            seed_k=np.array(self.seed_k)
            np.save(os.path.join(self.model_dir,'seed_k.npy'),seed_k)

            Alda =AnchoredLDA()
            Alda.setData(sc_corpus=deconv_df, genes=self.genes, sc_count_cells=self.cells, ann_list=self.ann_list,anchor_dict = self.anchor_dict,
                         out_dir = self.out_dir,alpha = alpha,eta=eta,refresh=refresh,seed_confidence=seed_confidence)
            if benchmark:
                Alda.benchmark(self.model_dir,self.ntopics_list,seed_topics,self.gene2id,n_ensemble,n_iter)
                self.benchmark_metrics_dict = Alda.benchmark_metrics_dict
            else:
                Alda.conduct_train(self.model_dir,ntopics,seed_topics,n_iter,ll_plot,seed)

                self.topic_cell_mat = Alda.topic_cell_mat
                self.topic_celltype_df = Alda.topic_celltype_df
                self.sc_celltype_prediction = Alda.sc_celltype_prediction
                self.celltype_num_dict = Alda.celltype_num_dict
                self.norm_selection = Alda.norm_selection
                self.model = Alda
            
    def trainCorEx(self,indicator_selection="accuracy",seed=9,n_iter=100,n_ensemble=10,ntopics=None,
                   anchor_strength=2,ifint=False,iflog=False,tc_plot=True,hierarchy_topic =[6,2],benchmark =False):
        """_summary_
        Args:
            n_ensemble (int, optional): The number of ensembles. Defaults to 100.
            n_iter (int, optional): The number of iterations for each run. Defaults to 100.
            refresh (int, optional): _description_. Defaults to 10.
            ct_plot (bool, optional): _description_. Defaults to True.
        """
        if ifint:
            deconv_df = self.target_int
        else:
            deconv_df = self.mm_df

        if len(self.anchor_list) < 1:
            print("Selecting the unsupervised CorEx model.")
        else:
            print("Selecting the Anchored CorEx model.")
        
        Cex = CorEx()
        Cex.setData(deconv_df, self.genes, self.cells, self.ann_list,self.anchor_list,self.out_dir)
        if benchmark:
            # print(Cex.sc_corpus.shape) #doc*word
            Cex.benchmark(self.model_dir,self.ntopics_list,anchor_strength,n_ensemble,n_iter,log=iflog)
            self.accuracy_bayes_dict = Cex.accuracy_bayes_dict
            self.accuracy_bayesnorm_dict = Cex.accuracy_bayesnorm_dict
            self.accuracy_lg_dict = Cex.accuracy_lg_dict
            self.nmi_dict =Cex.nmi_dict
            self.umass_coherence_dict = Cex.umass_coherence_dict
            self.cv_coherence_dict = Cex.cv_coherence_dict
            self.final_tcs_dict = Cex.final_tcs_dict
            self.model = Cex.model
        else:
            if n_ensemble>0:
                Cex.ensemble_train(self.model_dir,self.ntopics_list,anchor_strength,indicator_selection,n_ensemble,n_iter,log=iflog)
        
                if len(hierarchy_topic)>0:
                    Cex.hierarchy_topic(n_hidden_list=hierarchy_topic,max_edges=200,plot =False,figfile=None)
                if tc_plot:
                    Cex.tc_plot()

                self.topic_cell_mat = Cex.topic_cell_mat
                self.topic_celltype_df = Cex.topic_celltype_df
                self.sc_celltype_prediction = Cex.sc_celltype_prediction
                self.celltype_num_dict = Cex.celltype_num_dict
                self.norm_selection = Cex.norm_selection
                self.ntopics_selection = Cex.ntopics_selection
                self.seed_selection = Cex.seed_selection
                self.topic_layers = Cex.topic_layers
                self.indicator_selection = indicator_selection
            else:
                Cex.conduct_train(self.model_dir,ntopics,anchor_strength,n_iter,seed,iflog)
        
                if len(hierarchy_topic)>0:
                    hier_topics = Cex.hierarchy(n_hidden_list=hierarchy_topic,max_edges=200,plot=True,figfile=None)
                    self.hier_topics = hier_topics
                #   fig = Cex.plot_hierarchy(hier_topics)
                if tc_plot:
                    Cex.tc_plot()

                self.topic_cell_mat = Cex.topic_cell_mat
                self.topic_celltype_df = Cex.topic_celltype_df
                self.sc_celltype_prediction = Cex.sc_celltype_prediction
                self.celltype_num_dict = Cex.celltype_num_dict
                self.model = Cex
                self.norm_selection = Cex.norm_selection

    