import os
import copy
import random
import numpy as np
import pandas as pd
from umap import UMAP

import hdbscan
from hdbscan import HDBSCAN
import datetime as dt

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import scipy.sparse as ss
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import __version__ as sklearn_version

from packaging import version
from collections import defaultdict,Counter
from typing import  Union, Tuple,List,Mapping, Callable
from scipy.cluster import hierarchy as sch

from .BertFunction import *
from .performer_pytorch import *
from .epiDecon import *
from .plot_utils import *
from .hierarchy import *
from .vectorizers._ctfidf import ClassTfidfTransformer

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_top_words(topic_word, top_n=30):
    if isinstance(topic_word, ss.spmatrix):
        sparse_mat_T = topic_word.T
        # 初始化存储索引结果
        top_indices = []

        # 遍历每列
        for col_idx in range(sparse_mat_T.shape[1]):
            # 提取每列非零元素的索引和值
            col_data = sparse_mat_T.getcol(col_idx)
            nonzero_indices = col_data.nonzero()[0]
            nonzero_values = col_data.data

            # 排序非零元素并获取最大30个的索引
            sorted_nonzero_indices = nonzero_indices[np.argsort(nonzero_values)[-top_n:]]
            top_indices.append(sorted_nonzero_indices)

        # 转为NumPy数组
        return np.array(top_indices, dtype=object)  # 每列的索引可能长度不一
    else:
        return np.argsort(topic_word, axis=1)[:, -top_n:]

def is_supported_hdbscan(model):
    """Check whether the input model is a supported HDBSCAN-like model."""
    if isinstance(model, HDBSCAN):
        return True

    str_type_model = str(type(model)).lower()
    if "cuml" in str_type_model and "hdbscan" in str_type_model:
        return True

    return False

def check_is_fitted(topic_model):
    msg = (
        "This %(name)s instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )

    if topic_model.topics_ is None:
        raise ValueError(msg % {"name": type(topic_model).__name__})

def hdbscan_delegator(model, func: str, embeddings: np.ndarray = None):
    """Function used to select the HDBSCAN-like model for generating
    predictions and probabilities.

    Arguments:
        model: The cluster model.
        func: The function to use. Options:
                - "approximate_predict"
                - "all_points_membership_vectors"
                - "membership_vector"
        embeddings: Input embeddings for "approximate_predict"
                    and "membership_vector"
    """
    # Approximate predict
    if func == "approximate_predict":
        if isinstance(model, HDBSCAN):
            predictions, probabilities = hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan

            predictions, probabilities = cuml_hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities

        predictions = model.predict(embeddings)
        return predictions, None

    # All points membership
    if func == "all_points_membership_vectors":
        if isinstance(model, hdbscan.HDBSCAN):
            return hdbscan.all_points_membership_vectors(model)

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan

            return cuml_hdbscan.all_points_membership_vectors(model)

        return None

    # membership_vector
    if func == "membership_vector":
        if isinstance(model, hdbscan.HDBSCAN):
            probabilities = hdbscan.membership_vector(model, embeddings)
            return probabilities

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan

            probabilities = cuml_hdbscan.membership_vector(model, embeddings)
            return probabilities

        return None

def get_unique_distances(dists: np.array, noise_max=1e-7) -> np.array:
    """Check if the consecutive elements in the distance array are the same. If so, a small noise
    is added to one of the elements to make sure that the array does not contain duplicates.

    Arguments:
        dists: distance array sorted in the increasing order.
        noise_max: the maximal magnitude of noise to be added.

    Returns:
         Unique distances sorted in the preserved increasing order.
    """
    dists_cp = dists.copy()

    for i in range(dists.shape[0] - 1):
        if dists[i] == dists[i + 1]:
            # returns the next unique distance or the current distance with the added noise
            next_unique_dist = next((d for d in dists[i + 1 :] if d != dists[i]), dists[i] + noise_max)

            # the noise can never be large then the difference between the next unique distance and the current one
            curr_max_noise = min(noise_max, next_unique_dist - dists_cp[i])
            dists_cp[i + 1] = np.random.uniform(low=dists_cp[i] + curr_max_noise / 2, high=dists_cp[i] + curr_max_noise)
    return dists_cp

class BaseDimensionalityReduction:
    def fit(self, X: np.ndarray = None):
        return self
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

class scTopic_BERT():
    def __init__(self):
        self.genes = None
        self.use_genes =None
        self.cells = None
        self.ann_dict = None
        self.input_df = None
        self.anchor_list = None
        self.supervise = None
        self.y = None
        self.device_name = None
        self.calculate_probabilities = True
        self.topic_embeddings_ = None
        self.min_cluster_size =10
        self.top_n_words = 30
        self.verbose = False

    def setData(self,object,out_dir,outname=None,custom_labels_= None,ntopics_list = "auto"):

        self.genes = object.sc_genes
        if object.use_genes is None:
            self.use_genes = self.genes
            self.gene_select = False
        else:
            self.use_genes = object.use_genes
            self.gene_select = True
        self.cells = object.sc_cells
        self.ann_dict = object.ann_dict
        self.input_df = object.input_df
        self.input_int = object.input_int

        self.ntopics_list = ntopics_list
        self.custom_labels_ = custom_labels_

        if object.use_anchor_dict is not None:
            self.anchor_dict = object.use_anchor_dict
            self.anchor_list = object.use_anchor_list
            self.seed_mat = object.seed_mat

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

        self.celltypes = celltypes
        self.ann_list = cell_celltype_list

        if outname==None:
            time_str= dt.datetime.strftime(dt.datetime.now(),'%Y%m%d_%H%M%S')
            outname = time_str
        self.outname = outname

    def finetune_set(self,CLASS = 7,dropout=0., h_dim=128,gene2vec_file=None,seed=9,
                              device_name = "cuda:0"):
        self.seed = seed
        self.device_name = device_name
        device = torch.device(self.device_name)
        label_dict, label = np.unique(np.array(self.ann_list), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored

        class_num = np.unique(label, return_counts=True)[1].tolist()

        class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
        label = torch.from_numpy(label)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for index_train, index_val in sss.split(self.input_df, label):
            data_train, label_train = self.input_df[index_train], label[index_train]
            data_val, label_val = self.input_df[index_val], label[index_val]
            train_dataset = SCDataset(data_train, label_train,CLASS,device)
            val_dataset = SCDataset(data_val, label_val,CLASS,device)

        if gene2vec_file is not None:
            # self.gene_embeddings = np.load(gene2vec_file)
            model = PerformerLM(
                num_tokens = CLASS,
                dim = 200,
                depth = 6,
                max_seq_len = len(self.genes)+1,
                heads = 10,
                gene_weight_file=gene2vec_file,
                local_attn_heads = 0,
                g2v_position_emb = True)
        else:
            model = PerformerLM(
                num_tokens = CLASS,
                dim = 200,
                depth = 6,
                max_seq_len = len(self.genes)+1,
                heads = 10,
                local_attn_heads = 0,
                g2v_position_emb = False)

        model.to_out = Identity(dropout=dropout, h_dim=h_dim, out_dim=label_dict.shape[0])

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_dict = label_dict
        self.CLASS = CLASS
        
    def finetune(self,pretrain_file=None,learning_rate=1e-4,seed=9,
                       trigger_times=0,epochs=10,batch_size=8,
                       GRADIENT_ACCUMULATION =60,VALIDATE_EVERY =1,
                       PATIENCE = 10,UNASSIGN_THRES = 0.0):
        self.seed = seed
        torch.manual_seed(self.seed)
        ckpt = torch.load(pretrain_file,map_location=torch.device(self.device_name))
        model_name = os.path.join(self.model_dir,"scBERT_finetune_best")
        self.model_name = model_name

        if self.device_name !="mps":
            model.load_state_dict(ckpt['model_state_dict'])
            for param in model.parameters():
                param.requires_grad = False
            for param in model.norm.parameters():
                param.requires_grad = True
            for param in model.performer.net.layers[-2].parameters():
                param.requires_grad = True
        else:
            for key, value in ckpt['model_state_dict'].items():
                if value.dtype == torch.float64:
                    ckpt['model_state_dict'][key] = value.to(torch.float32)
            model.load_state_dict(ckpt['model_state_dict'])

            for param in model.parameters():
                if param.dtype == torch.float64:
                    param.data = param.data.float()
                param.requires_grad = False
            for param in model.norm.parameters():
                if param.dtype == torch.float64:
                    param.data = param.data.float()
                param.requires_grad = True
            for param in model.performer.net.layers[-2].parameters():
                if param.dtype == torch.float64:
                    param.data = param.data.float()
                param.requires_grad = True

        device = torch.device(self.device_name)
        model = self.model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=learning_rate,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        loss_fn = nn.CrossEntropyLoss(weight=None).to(device)
        max_acc = 0.0
        train_sampler = SimpleSampler(self.train_dataset)
        val_sampler = SimpleSampler(self.val_dataset)
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, sampler=val_sampler)

        for i in range(1, epochs+1):
            train_loader.sampler.set_epoch(i)
            model.train()
            running_loss = 0.0
            cum_acc = 0.0
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                data, labels = data.to(device), labels.to(device)
                if index % GRADIENT_ACCUMULATION != 0:
                    
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    
                if index % GRADIENT_ACCUMULATION == 0:      
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                
                final = softmax(logits)
                final = final.argmax(dim=-1)
                cum_acc += torch.eq(final, labels).sum().item() / labels.size(0)
                
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * cum_acc / len(train_loader)
            print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:4f}%  ==')
            
            scheduler.step()

            if i % VALIDATE_EVERY == 0:
                model.eval()
                running_loss = 0.0
                predictions = []
                truths = []
                with torch.no_grad():
                    for index, (data_v, labels_v) in enumerate(val_loader):
                
                        data_v, labels_v = data_v.to(device), labels_v.to(device)
                    
                        logits = model(data_v)
                        loss = loss_fn(logits, labels_v)
                        running_loss += loss.item()
                        
                        softmax = nn.Softmax(dim=-1)
                        final_prob = softmax(logits)
                        final = final_prob.argmax(dim=-1)
                        final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                        
                        predictions.append(final.cpu())
                        truths.append(labels_v.cpu())
                    
                    predictions = torch.cat(predictions)
                    truths = torch.cat(truths)
                    no_drop = predictions != -1
                    predictions = predictions[no_drop]
                    truths = truths[no_drop]
                    
                    cur_acc = accuracy_score(truths, predictions)
                    f1 = f1_score(truths, predictions, average='macro')
                    val_loss = running_loss / len(val_loader)

                    print(f'Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}')
                    # print(confusion_matrix(truths, predictions))
                    # print(classification_report(truths,predictions, target_names=self.label_dict.tolist(), digits=4))
                    
                    if cur_acc > max_acc:
                        max_acc = cur_acc
                        trigger_times = 0
                        save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, self.model_dir)
                    else:
                        trigger_times += 1
                        if trigger_times > PATIENCE:
                            break
            del predictions, truths

    def getEmbeddingModel(self,pretrain_file=None, gene2vec_file=None,
                          encodings=False,CLS = False,last_layer=False,tie=False,CLASS = 7,dropout=0., h_dim=128):

        if pretrain_file is None:
            pretrain_file = self.model_name
        
        if gene2vec_file is not None:
            self.gene2vec_file = gene2vec_file
            self.g2v = True
        else:
            self.g2v = False
        
        self.CLS =CLS
        self.encodings = encodings
        self.CLASS = CLASS
        self.max_seq_len = max_seq_len = len(self.genes)+1
        if self.encodings:
            if self.CLS:
                self.outname = "CLS"+"_"+self.outname
            else:
                self.outname = "Pooling"+"_"+self.outname
        else:
            if last_layer:
                self.outname = "Last"+"_"+self.outname
            else:
                self.outname = "Conv"+"_"+self.outname

        model = PerformerLM_modified(
            num_tokens = CLASS,
            dim = 200,
            depth = 6,
            max_seq_len = max_seq_len,
            heads = 10,
            gene_weight_file=gene2vec_file,
            local_attn_heads = 0,
            tie_embed = tie,
            return_encodings = encodings,
            return_cls = CLS,
            return_last=last_layer,
            g2v_position_emb = self.g2v)
        
        model.to_out = Identity(dropout=dropout, h_dim=h_dim, out_dim=len(self.celltypes))

        ckpt = torch.load(pretrain_file,map_location=torch.device("cpu"))
        model.load_state_dict(ckpt['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.performer.net.layers[-2].parameters():
            param.requires_grad = True

        self.embedding_model = model

    def getUmapModel(self,supervise = True,n_neighbors=15,n_components=2,min_dist=0.0,seed=9):
        self.seed =seed

        if supervise:
            self.supervise = True
            umap_model = BaseDimensionalityReduction()
        else:
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric='cosine',
                # random_state=self.seed
                )
        self.umap_model = umap_model

    def getHdbscanModel(self,supervise=True,min_samples=3,min_cluster_size=50,metric='euclidean',cluster_selection_method='eom'):
        if self.supervise is None:
            self.supervise =supervise
        if self.supervise:
            hdbscan_model  = LogisticRegression(random_state=self.seed)
        else:
            hdbscan_model = HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size,
                                    metric= metric,cluster_selection_method=cluster_selection_method,prediction_data=True)
        self.hdbscan_model = hdbscan_model
    
    def getCtfidfModel(self,ctfidf_model=None,bm25_weighting=True):
        if ctfidf_model is None:
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting)
        self.ctfidf_model = ctfidf_model
    
    def _guided_topic_modeling(self,seed_embeddings,embeddings):
        """Apply Guided Topic Modeling.
        Returns:
            y: The labels for each seeded topic embeddings: Updated embeddings
        """
        # Create embeddings from the seeded topics
        seed_topic_num = seed_embeddings.shape[0]
        seed_topic_embeddings = np.vstack([seed_embeddings, embeddings.mean(axis=0)])

        # Label documents that are most similar to one of the seeded topics
        sim_matrix = cosine_similarity(embeddings, seed_topic_embeddings)
        y = [np.argmax(sim_matrix[index]) for index in range(sim_matrix.shape[0])]
        y = [val if val != seed_topic_num else -1 for val in y]

        # Average the document embeddings related to the seeded topics with the
        # embedding of the seeded topic to force the documents in a cluster
        for seed_topic in range(seed_topic_num):
            indices = [index for index, topic in enumerate(y) if topic == seed_topic]
            embeddings[indices] = embeddings[indices] * 0.75 + seed_topic_embeddings[seed_topic] * 0.25
        return y, embeddings

    def getEmbedding(self,batch_size=50,input_df= None,seed_embedding=None,seed=9,device_name = "cuda:0"):
        self.device_name = device_name
        self.seed = seed
        torch.manual_seed(self.seed)
        device = torch.device(self.device_name)

        if input_df is None:
            input_df = self.input_df

        model = self.embedding_model.to(device)

        if self.gene_select:
            gene_mask = [self.genes.index(i) for i in self.genes if i in self.use_genes]
            input_df[:,gene_mask]= 0
            self.outname = self.outname+"_"+"GeneSelect"

        dataset = SCDataset(input_df,np.array(range(len(self.cells))),self.CLASS,torch.device("cpu"))
        data_loader = DataLoader(dataset, batch_size=batch_size)

        model.eval()
        embedding_list = []
        data_list = []
        cell_list = []
        for index, (data, labels) in enumerate(data_loader):
            cell_list+=[labels.cpu().numpy()]
            data_list.append(data)
            data = data.to(device)

            with torch.no_grad():
                embeddings = model(data)
                embedding_list.append(embeddings.cpu().numpy())

        shape = (batch_size-embedding_list[index].shape[0], embedding_list[index].shape[1])
        pad_array = np.zeros(shape)
        embedding_list[index] = np.concatenate((embedding_list[index], pad_array), axis=0)
        embeddings_np = np.concatenate(embedding_list,axis=0)
        embeddings_np = embeddings_np[:len(self.cells),:]

        pad_array = np.zeros((shape[0],data_list[index].shape[1]))
        data_list[index] = np.concatenate((data_list[index], pad_array), axis=0)
        input_int = np.concatenate(data_list,axis=0)
        input_int = input_int[:len(self.cells),:]

        pad_array = np.zeros((shape[0],))
        cell_list[index] = np.concatenate((cell_list[index], pad_array), axis=0)
        cellid = np.concatenate(cell_list,axis=0)
        cellid = cellid[:len(self.cells)].astype(int)

        self.input_int = copy.deepcopy(input_int)
        self.cell_id = copy.deepcopy(cellid)

        del embeddings,data
        if self.device_name.startswith("cuda"):
            torch.cuda.empty_cache()

        if self.anchor_list is not None:
            self.outname = self.outname +"_seeded" 
            if seed_embedding is None:
                seed_dataset = SCDataset(self.seed_mat,np.array(range(self.seed_mat.shape[0])),self.CLASS,torch.device("cpu"))
                seeddata_loader = DataLoader(seed_dataset, batch_size=self.seed_mat.shape[0])
            
                model.eval()
                seed_embedding_list = []
                for index, (seed_data, labels) in enumerate(seeddata_loader):
                    seed_data = seed_data.to(device)
                    with torch.no_grad():
                        seed_embeddings = model(seed_data)
                        seed_embedding_list.append(seed_embeddings.cpu().numpy())
                
                shape = (batch_size-seed_embedding_list[index].shape[0], seed_embedding_list[index].shape[1])
                pad_array = np.zeros(shape)
                seed_embedding_list[index] = np.concatenate((seed_embedding_list[index], pad_array), axis=0)
                seed_embeddings_np = np.concatenate(seed_embedding_list,axis=0)
                seed_embeddings_np = seed_embeddings_np[:len(self.cells),:]

                
                self.seed_embeddings = seed_embeddings_np
                np.save(os.path.join(self.model_dir,"embeddings_seed_"+self.outname+".npy"), seed_embeddings_np)
                gc.collect()
            else:
                self.seed_embeddings = seed_embeddings
            
            y, embeddings_np = self._guided_topic_modeling(self.seed_embeddings,embeddings_np)
            self.y = y
            print("use seed!")

        self.embeddings = copy.deepcopy(embeddings_np)
        np.save(os.path.join(self.model_dir,"embeddings_"+self.outname+".npy"), embeddings_np)
        np.save(os.path.join(self.model_dir,"input_"+self.outname+".npy"), input_int)
        np.save(os.path.join(self.model_dir,"cellindex_embeddings_"+self.outname+".npy"), cellid) 
        gc.collect()

        if self.device_name.startswith("cuda"):
            torch.cuda.empty_cache()

    def _extract_topics(
        self,X):

        # input_int = copy.deepcopy(self.input_int)
    
        # topic_list = []
        # newcluster_list = sorted(set(documents_meta.Topic))
        # self.unique_topics_ = list(newcluster_list)

        # for i in self.unique_topics_:
        #     cell_ind = [j for j in range(documents_meta.shape[0]) if i==documents_meta.Topic[j]]
        #     topic_list.append(input_int[cell_ind,:].sum(0))
        # X_arr = np.vstack(topic_list)
        # c_tf_idf = self.ctfidf_model.fit_transform(np.array(X_arr))
        c_tf_idf = self.ctfidf_model.fit_transform(X)

        # try:
        #     c_tf_idf.data = np.nan_to_num(c_tf_idf.data,np.NINF)
        # except:
        #     c_tf_idf = np.nan_to_num(c_tf_idf,np.NINF)

        return csr_matrix(c_tf_idf)

    def _update_topic_size(self, documents_meta: pd.DataFrame):
        """Calculate the topic sizes.
        Arguments:
            documents_meta: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        self.topic_sizes_ = collections.Counter(documents_meta.Topic.values.tolist())
        self.topics_ = documents_meta.Topic.astype(int).tolist()

    @property
    def _outliers(self):
        """Some algorithms have outlier labels (-1) that can be tricky to work
        An integer indicating whether outliers are present in the topic model
        """
        return 1 if -1 in self.topic_sizes_ else 0
    
    def getTopicEmbedding(self,documents_meta,topic_embeddings_=None,mappings=None):
        if topic_embeddings_ is None:
            topic_embeddings = []
            topics = self.unique_topics_

            for topic in topics:
                indices = documents_meta.loc[documents_meta.Topic == topic, "ID"].values
                indices = [int(index) for index in indices]
                topic_embedding = np.mean(self.embeddings[indices], axis=0)
                topic_embeddings.append(topic_embedding)
            topic_embeddings_ = np.array(topic_embeddings)

        if mappings is not None:
            topic_embeddings_dict = {}
            for topic_to, topics_from in mappings.items():
                topic_ids = topics_from["topics_from"]
                topic_sizes = topics_from["topic_sizes"]
                
                embds = np.array(topic_embeddings_)[np.array(topic_ids) + self._outliers]
                topic_embedding = np.average(embds, axis=0, weights=topic_sizes) 
                topic_embeddings_dict[topic_to] = topic_embedding

            unique_topics =  sorted(list(set(topic_embeddings_dict.keys())))
            topic_embeddings_ = np.array([topic_embeddings_dict[topic] for topic in unique_topics])

        # return np.nan_to_num(topic_embeddings_, nan=0.0)
        return topic_embeddings_
    
    def _reduce_to_n_topics(self,topics, nr_topics,distance_matrix):
        # Cluster the gene(topic) embeddings using AgglomerativeClustering
        if version.parse(sklearn_version) >= version.parse("1.4.0"):
            cluster = AgglomerativeClustering(nr_topics - self._outliers, metric="precomputed", linkage="average")
        else:
            cluster = AgglomerativeClustering(
                nr_topics - self._outliers,
                affinity="precomputed",
                linkage="average",
            )
        cluster.fit(distance_matrix)
        new_topics = [cluster.labels_[topic] if topic != -1 else -1 for topic in topics]

        # Track mappings and sizes of topics for merging topic embeddings
        mapped_topics = {from_topic: to_topic \
                         for from_topic, to_topic in zip(topics, new_topics)}

        basic_mappings = defaultdict(list)
        for key, val in sorted(mapped_topics.items()):
            basic_mappings[val].append(key)

        mappings = {
            topic_to: {
                "topics_from": topics_from,
                "topic_sizes": [self.topic_sizes_[topic] for topic in topics_from],
            }
            for topic_to, topics_from in basic_mappings.items()
        }

        return new_topics,mappings,mapped_topics
    
    def _reduce_topics(self, documents_meta,use_ctfidf = False,ntopics_list="auto"):
        initial_nr_topics = len(self.unique_topics_)
        initial_documents_meta = copy.deepcopy(documents_meta)

        if isinstance(ntopics_list, list):
            accuracy_list = []
            topics_list = []

            topics = initial_documents_meta.Topic.tolist().copy()

            # Create topic distance matrix
            topic_embeddings = select_topic_representation(
                self.c_tf_idf_, self.topic_embeddings_, use_ctfidf, output_ndarray=True
            )[0][self._outliers:] # delete -1

            # Create topic distance matrix
            distance_matrix = 1 - cosine_similarity(topic_embeddings)
            np.fill_diagonal(distance_matrix, 0)

            for nr_topics in ntopics_list:
                if nr_topics <=initial_nr_topics:
                    topic_mapper_ = copy.deepcopy(self.topic_mapper_)
                    topic_embeddings_  = copy.deepcopy(self.topic_embeddings_)
                    dm = copy.deepcopy(initial_documents_meta)

                    if nr_topics <initial_nr_topics:
                        new_topics,mappings,mapped_topics= self._reduce_to_n_topics(topics=topics,nr_topics=nr_topics,distance_matrix=distance_matrix)
                        # Map topics
                        dm.Topic = new_topics
                        # Update representations
                        topic_mapper_.add_mappings(mapped_topics, topic_model=self)
                        topic_embeddings_ = self.getTopicEmbedding(dm,self.topic_embeddings_,mappings)

                    y=dm.Truth.to_list()
                    uni_topics = list(dm.sort_values("Topic").Topic.unique())
                    indices = np.array([uni_topics.index(i) for i in dm['Topic']], dtype=int)
                    cell_topic_mat = topic_embeddings_[indices, :]

                    model_l = LogisticRegression(random_state=self.seed)
                    # cell_topic_mat = StandardScaler().fit_transform(cell_topic_mat)
                    model_l.fit(cell_topic_mat,y=y)
                    y_p = model_l.predict(cell_topic_mat)
                    accuracy = accuracy_score(y, y_p)
                    accuracy_list.append(accuracy)
                    topics_list.append(nr_topics)

            if len(accuracy_list)>0:
                self.ntopics_selection = topics_list[accuracy_list.index(max(accuracy_list))]
                print(f"get {self.ntopics_selection} topics,can get {max(accuracy_list)} accuracy for predicting celltyes in scRNA-seq.")

                new_topics,mappings,mapped_topics = self._reduce_to_n_topics(topics=topics,nr_topics=self.ntopics_selection,distance_matrix=distance_matrix)
                documents_meta.Topic = new_topics

                self.topic_mapper_.add_mappings(mapped_topics, topic_model=self)
                self.topic_embeddings_ = self.getTopicEmbedding(documents_meta,self.topic_embeddings_,mappings)

        elif isinstance(ntopics_list, str):
            if ntopics_list == "auto":
                documents_meta,mappings,self.topic_mapper_= self._auto_reduce_topics(initial_documents_meta, self.topic_embeddings_,use_ctfidf)
                self.ntopics_selection = len(set(documents_meta.Topic))
            else:
                self.ntopics_selection = initial_nr_topics
        else:
            raise ValueError("nr_topics needs to be an list or strings! ")
        
        
        return documents_meta
    
    @staticmethod
    def _top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
        """Return indices of top n values in each row of a sparse matrix.
        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix
        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row
        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
            indices.append(values)
        return np.array(indices)

    @staticmethod
    def _top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
        """Return the top n values for each row in a sparse matrix.

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row

        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        return np.array(top_values)

    def _extract_words_per_topic(
        self,
        words: List[str],
        documents_meta: pd.DataFrame,
        c_tf_idf: csr_matrix = None,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Based on tf_idf scores per topic, extract the top n words per topic."""
        if c_tf_idf is None:
            c_tf_idf = csr_matrix(self.c_tf_idf_)

        labels = sorted(list(documents_meta.Topic.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = self._top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)


        # Get top 30 words per topic based on c-TF-IDF score
        base_topics = {
            label: [
                (words[word_index], score) if word_index is not None and score > 0 else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }

        topics = base_topics.copy()
        topics = {label: values[: self.top_n_words] for label, values in topics.items()}
        return topics

    def fit(self,embedding_file = None,y=None,cellid_file = None,use_ctfidf=True,use_bin = False):
        self.use_bin = use_bin
        self.use_ctfidf = use_ctfidf

        if embedding_file is None:
            embeddings = self.embeddings
            cell_list = self.cell_id.astype(int).tolist()
        else:
            embeddings = np.load(embedding_file)

            if self.use_bin:
                try:
                    input_int_file = os.path.join(os.path.dirname(embedding_file),"input_"+os.path.basename(embedding_file).lstrip("embeddings_"))
                    self.input_int =  np.load(input_int_file)[:,:len(self.genes)]
                except:
                    print("use gene expression for c-TF-IDF.")    
                    self.use_bin = False   

            self.embeddings = embeddings
            if cellid_file is None:
                cellid_file= os.path.join(os.path.dirname(embedding_file),"cellindex_"+os.path.basename(embedding_file))
                cell_list = np.load(cellid_file).astype(int).tolist()
            else:
                cell_list = np.load(cellid_file).astype(int).tolist()
        
        self.cell_list  = cell_list
        
        if self.use_bin==False:
            self.input_df = self.input_df[cell_list,:].copy()
            input_int = copy.deepcopy(self.input_df)
            input_int.data = np.round(input_int.data).astype(int) 
            self.input_int = input_int
        
        col_sums = self.input_int.sum(axis=0)
        col_sums_array = np.array(col_sums).flatten()

        self.gene_zero_dict = {
            "zero":np.where(col_sums_array == 0)[0],
            "non_zero":np.where(col_sums_array != 0)[0]
        }
        self.input_int = csr_matrix(self.input_int)

        documents = []
        for row in range(self.input_int.shape[0]):
            start_idx = self.input_int.indptr[row]
            end_idx = self.input_int.indptr[row + 1]
            
            # 获取该行所有非零元素的值和列索引
            nonzero_vals = self.input_int.data[start_idx:end_idx]
            nonzero_cols = self.input_int.indices[start_idx:end_idx]
            doc = " ".join([" ".join([word] * int(freq)) for word,freq in \
            zip([self.use_genes[i] for i in nonzero_cols.tolist()],nonzero_vals)])
            documents.append(doc)
        self.documents = documents
        # self.input_int = self.input_int[:,self.gene_zero_dict["non_zero"]]
        # self.use_genes = [self.use_genes[i] for i in self.gene_zero_dict["non_zero"]]

        if y is None:
            if self.y is None:
                pass
            else:
                y = self.y 

        try:
            y = np.array(y) if y is not None else None
            reduced_embeddings = self.umap_model.fit_transform(embeddings, y=y)
        except TypeError:
            reduced_embeddings = self.umap_model.fit_transform(embeddings)

        try:
            self.hdbscan_model.fit(reduced_embeddings, y=y)
        except TypeError:
            self.hdbscan_model.fit(reduced_embeddings)

        if self.supervise:
            newcluster_array = self.hdbscan_model.predict(reduced_embeddings)
            newcluster_list = newcluster_array.tolist()
            cm = confusion_matrix(y_true=y, y_pred=newcluster_array)
        else:
            newcluster_list = self.hdbscan_model.labels_.tolist()

        documents_meta= pd.DataFrame({"ID": cell_list, "Topic": newcluster_list,"Truth":[self.ann_list[i] for i in cell_list]})
        self.unique_topics_ = sorted(list(set(newcluster_list)))
        self.documents_meta = documents_meta
        self.reduced_embeddings =reduced_embeddings

        self.vectorizer_model = CountVectorizer()
        X = self.vectorizer_model.fit_transform(self.documents)
        self.c_tf_idf_ = self._extract_topics(X)

        self._update_topic_size(documents_meta)
        self.topic_mapper_ = TopicMapper(self.topics_)
        self.topic_embeddings_ = self.getTopicEmbedding(documents_meta)

        if self.ntopics_list != "pass":
            initial_documents_meta = copy.deepcopy(documents_meta)
            documents_meta = self._reduce_topics(initial_documents_meta,use_ctfidf=use_ctfidf,ntopics_list=self.ntopics_list)
            self._update_topic_size(documents_meta)
            # self.c_tf_idf_ = self._extract_topics(documents_meta)

        self.documents_meta = documents_meta
        self.topic_representations_ = self._extract_words_per_topic(self.use_genes, documents_meta)
        celltype_topic_mat,_= self.approximate_distribution(self.documents, self.documents_meta.Truth.tolist(),calculate_tokens=False)
        _,gene_topic_mat= self.approximate_distribution(self.documents,["all"]*len(self.ann_list),calculate_tokens=True)
        self.gene_topic_mat = gene_topic_mat[0]
        # self.gene_topic_mat = np.nan_to_num(self.c_tf_idf_.T, nan=0)
        self.cv_coherence = self._cv_coherence(self.gene_topic_mat.T,self.gene2vec_file)

        gene_topic_df = pd.DataFrame(self.gene_topic_mat,index =self.use_genes,columns=["Topic%s" %i for i in self.unique_topics_])
        gene_topic_df.to_csv(os.path.join(self.model_dir, "gene_topic_mat_"+self.outname+".txt"),sep="\t")

        # if self.supervise:
        #     self.sc_celltype_prediction = newcluster_list
        #     self.topic_celltype_mat = cm.T
        #     self.ntopics_selection = len(set(newcluster_list))
        #     pd.DataFrame(self.topic_celltype_mat).to_csv(os.path.join(self.model_dir, "topic_celltype_mat_"+self.outname+".txt"),sep="\t")
            
        # cm= pd.crosstab(documents_meta['Topic'], documents_meta['Truth'])
        # topic_celltype_mat = cm.div(cm.sum(axis=0), axis=1)
        # self.topic_celltype_mat = topic_celltype_mat.loc[self.unique_topics_,:]
        self.topic_celltype_mat = celltype_topic_mat.T
        pd.DataFrame(self.topic_celltype_mat,\
                     index=["Topic%s" %i for i in self.unique_topics_],columns = sorted(list(set(documents_meta.Truth)))).to_csv(os.path.join(self.model_dir, "topic_celltype_mat_"+self.outname+".txt"),sep="\t")

        vocab = self.use_genes
        x = np.array(self.c_tf_idf_.todense())
        for i in range(x.shape[0]):
            important_words_index = np.argsort(x[i])[::-1]  # 将稀疏矩阵转换为数组并排序
            top_30_words = [vocab[index] for index in important_words_index[:10]]  # 获取前10个词
            print(top_30_words)
   
    def benchmark(self,embedding_file = None,y=None,cellid_file = None,use_ctfidf=True,use_bin = False):
        self.use_bin = use_bin
        self.use_ctfidf = use_ctfidf

        if embedding_file is None:
            embeddings = self.embeddings
            cell_list = self.cell_id.astype(int).tolist()
        else:
            embeddings = np.load(embedding_file)

            if self.use_bin:
                try:
                    input_int_file = os.path.join(os.path.dirname(embedding_file),"input_"+os.path.basename(embedding_file).lstrip("embeddings_"))
                    self.input_int =  np.load(input_int_file)[:,:len(self.genes)]
                except:
                    print("use gene expression for c-TF-IDF.")    
                    self.use_bin = False   

            self.embeddings = embeddings
            if cellid_file is None:
                cellid_file= os.path.join(os.path.dirname(embedding_file),"cellindex_"+os.path.basename(embedding_file))
                cell_list = np.load(cellid_file).astype(int).tolist()
            else:
                cell_list = np.load(cellid_file).astype(int).tolist()
        
        self.input_df = self.input_df[cell_list,:].copy()

        if self.use_bin==False:
            input_int = copy.deepcopy(self.input_df)
            input_int.data = np.round(input_int.data).astype(int) 
            self.input_int = input_int

        if y is None:
            if self.y is None:
                pass
            else:
                y = self.y 

        try:
            y = np.array(y) if y is not None else None
            reduced_embeddings = self.umap_model.fit_transform(embeddings, y=y)
        except TypeError:
            reduced_embeddings = self.umap_model.fit_transform(embeddings)

        try:
            self.hdbscan_model.fit(reduced_embeddings, y=y)
        except TypeError:
            self.hdbscan_model.fit(reduced_embeddings)

        if self.supervise:
            newcluster_array = self.hdbscan_model.predict(reduced_embeddings)
            newcluster_list = newcluster_array.tolist()
            cm = confusion_matrix(y_true=y, y_pred=newcluster_array)
        else:
            newcluster_list = self.hdbscan_model.labels_.tolist()

        documents_meta= pd.DataFrame({"ID": cell_list, "Topic": newcluster_list,"Truth":[self.ann_list[i] for i in cell_list]})
        self.c_tf_idf_ = self._extract_topics(documents_meta)
        self._update_topic_size(documents_meta)
        self.topic_mapper_ = TopicMapper(self.topics_)
        self.topic_embeddings_ = self.getTopicEmbedding(documents_meta)

        initial_documents_meta = copy.deepcopy(documents_meta)
        initial_nr_topics = len(self.unique_topics_)

        if isinstance(self.ntopics_list, list):
            accuracy_list = []
            cv_coherence_list = []
            umass_coherence_list =[]
            topics_list = []

            topics = initial_documents_meta.Topic.tolist().copy()

            # Create topic distance matrix
            topic_embeddings = select_topic_representation(
                self.c_tf_idf_, self.topic_embeddings_, use_ctfidf, output_ndarray=True
            )[0][self._outliers:] # delete -1

            # Create topic distance matrix
            distance_matrix = 1 - cosine_similarity(topic_embeddings)
            np.fill_diagonal(distance_matrix, 0)

            for nr_topics in self.ntopics_list:
                if nr_topics <=initial_nr_topics:
                    topic_mapper_ = copy.deepcopy(self.topic_mapper_)
                    topic_embeddings_  = copy.deepcopy(self.topic_embeddings_)
                    dm = copy.deepcopy(initial_documents_meta)

                    if nr_topics <initial_nr_topics:
                        new_topics,mappings,mapped_topics= self._reduce_to_n_topics(topics=topics,nr_topics=nr_topics,distance_matrix=distance_matrix)
                        # Map topics
                        dm.Topic = new_topics
                        # Update representations
                        topic_mapper_.add_mappings(mapped_topics, topic_model=self)
                        topic_embeddings_ = self.getTopicEmbedding(dm,self.topic_embeddings_,mappings)

                    y=dm.Truth.to_list()
                    uni_topics = list(dm.sort_values("Topic").Topic.unique())
                    indices = np.array([uni_topics.index(i) for i in dm['Topic']], dtype=int)
                    cell_topic_mat = topic_embeddings_[indices, :]

                    model_l = LogisticRegression(random_state=self.seed)
                    # cell_topic_mat = StandardScaler().fit_transform(cell_topic_mat)
                    model_l.fit(cell_topic_mat,y=y)
                    y_p = model_l.predict(cell_topic_mat)
                    accuracy = accuracy_score(y, y_p)
                    accuracy_list.append(accuracy)
                    topics_list.append(nr_topics)
                    
                    gene_topic_mat = np.nan_to_num(self._extract_topics(dm).T, nan=0)
                    # umc = self._umass_coherence(gene_topic_mat.T,self.input_int, top_n=30, epsilon=1e-12)
                    # umass_coherence_list.append(umc)
                    if self.g2v:
                        cvc = self._cv_coherence(gene_topic_mat.T,self.gene2vec_file)
                        cv_coherence_list.append(cvc)

            if len(accuracy_list)>0:
                self.ntopics_selection = topics_list[accuracy_list.index(max(accuracy_list))]
                print(f"get {self.ntopics_selection} topics,can get {max(accuracy_list)} accuracy for predicting celltyes in scRNA-seq.")

                new_topics,mappings,mapped_topics = self._reduce_to_n_topics(topics=topics,nr_topics=self.ntopics_selection,distance_matrix=distance_matrix)
                documents_meta.Topic = new_topics

                self.topic_mapper_.add_mappings(mapped_topics, topic_model=self)
                self.topic_embeddings_ = self.getTopicEmbedding(documents_meta,self.topic_embeddings_,mappings)

            self.accuracy_list = accuracy_list
            self.cv_coherence_list = cv_coherence_list
            self.umass_coherence_list = umass_coherence_list
            self.topics_list = topics_list
        

        self._update_topic_size(documents_meta)
        self.c_tf_idf_ = self._extract_topics(documents_meta)

        self.documents_meta = documents_meta
        self.topic_representations_ = self._extract_words_per_topic(self.use_genes, documents_meta)
        self.gene_topic_mat = np.nan_to_num(self.c_tf_idf_.T, nan=0)

        gene_topic_df = pd.DataFrame(self.gene_topic_mat.todense(),index =self.genes,columns=["Topic%s" %i for i in self.unique_topics_])
        gene_topic_df.to_csv(os.path.join(self.model_dir, "gene_topic_mat_"+self.outname+".txt"),sep="\t")

        if self.supervise:
            self.sc_celltype_prediction = newcluster_list
            self.topic_celltype_mat = cm.T
            self.ntopics_selection = len(set(newcluster_list))
            pd.DataFrame(self.topic_celltype_mat).to_csv(os.path.join(self.model_dir, "topic_celltype_mat_"+self.outname+".txt"),sep="\t")
            
        cm= pd.crosstab(documents_meta['Topic'], documents_meta['Truth'])
        topic_celltype_mat = cm.div(cm.sum(axis=0), axis=1)
        self.topic_celltype_mat = topic_celltype_mat.loc[self.unique_topics_,:]
        pd.DataFrame(self.topic_celltype_mat).to_csv(os.path.join(self.model_dir, "topic_celltype_mat_"+self.outname+".txt"),sep="\t")

    def _auto_reduce_topics(self, documents_meta: pd.DataFrame, topic_embeddings_,use_ctfidf: bool = False) -> pd.DataFrame:
        """Reduce the number of topics automatically using HDBSCAN.
        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, the
                        embeddings from the embedding model are used.

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """

        topics = documents_meta.Topic.tolist().copy()
        # Find similar topics
        embeddings = select_topic_representation(
            self.c_tf_idf_, topic_embeddings_, use_ctfidf, output_ndarray=True
        )[0]
        norm_data = normalize(embeddings, norm="l2")

        if self._outliers==1:
            unique_topics = self.unique_topics_[1:]
            norm_data = norm_data[1:]
        else:
            unique_topics = self.unique_topics_
        max_topic = unique_topics[-1]


        predictions = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ).fit_predict(norm_data)

        # Map similar topics
        mapped_topics = {
            unique_topics[index]: prediction + max_topic
            for index, prediction in enumerate(predictions)
            if prediction != -1
        }
        documents_meta.Topic = documents_meta.Topic.map(mapped_topics).fillna(documents_meta.Topic).astype(int)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, documents_meta.Topic.tolist())}

        # Track mappings and sizes of topics for merging topic embeddings
        mappings = defaultdict(list)
        for key, val in sorted(mapped_topics.items()):
            mappings[val].append(key)

        mappings = {
            topic_to: {
                "topics_from": topics_from,
                "topic_sizes": [self.topic_sizes_[topic] for topic in topics_from],
            }
            for topic_to, topics_from in mappings.items()
        }

        # Update documents and topics
        self.topic_mapper_.add_mappings(mapped_topics, topic_model=self)
        self.c_tf_idf_ = self._extract_topics(documents_meta)
        documents_meta = self._sort_mappings_by_frequency(documents_meta)
        #topic_embeddings_=self.getTopicEmbedding(documents_meta,mappings=mappings)
        self._update_topic_size(documents_meta)
        return documents_meta,mappings
    
    def _map_probabilities(
        self, probabilities: Union[np.ndarray, None], original_topics: bool = False
    ) -> Union[np.ndarray, None]:
        """Map the probabilities to the reduced topics.
        This is achieved by adding together the probabilities of all topics that are mapped to the same topic. Then,
        the topics that were mapped from are set to 0 as they were reduced.
        Arguments:
            probabilities: An array containing probabilities
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.
        Returns:
            mapped_probabilities: Updated probabilities
        """
        mappings = self.topic_mapper_.get_mappings(original_topics)

        # Map array of probabilities (probability for assigned topic per document)
        if probabilities is not None:
            if len(probabilities.shape) == 2:
                mapped_probabilities = np.zeros(
                    (
                        probabilities.shape[0],
                        len(set(mappings.values())) - self._outliers,
                    )
                )
                for from_topic, to_topic in mappings.items():
                    if to_topic != -1 and from_topic != -1:
                        mapped_probabilities[:, to_topic] += probabilities[:, from_topic]

                return mapped_probabilities

        return probabilities

    def _map_predictions(self, predictions: List[int]) -> List[int]:
        """Map predictions to the correct topics if topics were reduced."""
        mappings = self.topic_mapper_.get_mappings(original_topics=True)
        mapped_predictions = [mappings[prediction] if prediction in mappings else -1 for prediction in predictions]
        return mapped_predictions

    def transform(self,input_df,batch_size=50,seed=9,similarity="cluster",device_name="cuda:0"):
        self.seed = seed
        torch.manual_seed(self.seed)
        device = torch.device(device_name)
        model = self.embedding_model.to(device)
        sampel_num = input_df.shape[0]

        if sampel_num<batch_size:
            batch_size=sampel_num

        if self.gene_select:
            gene_mask = [self.genes.index(i) for i in self.genes if i in self.use_genes]
            input_df[:,gene_mask]= 0

        dataset = SCDataset(input_df,np.array(range(sampel_num)),self.CLASS,torch.device("cpu"))
        data_loader = DataLoader(dataset, batch_size=batch_size)

        model.eval()
        embedding_list = []
        data_list = []
        sample_list = []
        for index, (data, labels) in enumerate(data_loader):
            sample_list+=[labels.cpu().numpy()]
            data_list.append(data)
            data = data.to(device)
            with torch.no_grad():
                embeddings = model(data)
                embedding_list.append(embeddings.cpu().numpy())

        shape = (batch_size-embedding_list[index].shape[0], embedding_list[index].shape[1])
        pad_array = np.zeros(shape)
        embedding_list[index] = np.concatenate((embedding_list[index], pad_array), axis=0)
        embeddings_np = np.concatenate(embedding_list,axis=0)
        embeddings_np = embeddings_np[:sampel_num,:]

        pad_array = np.zeros((shape[0],data_list[index].shape[1]))
        data_list[index] = np.concatenate((data_list[index], pad_array), axis=0)
        input_int = np.concatenate(data_list,axis=0)
        input_int = input_int[:sampel_num,:]

        pad_array = np.zeros((shape[0],))
        sample_list[index] = np.concatenate((sample_list[index], pad_array), axis=0)
        sampleid = np.concatenate(sample_list,axis=0)
        sampleid = sampleid[:sampel_num].astype(int)

        np.save(os.path.join(self.model_dir,"sampleinput_embeddings_"+self.outname+".npy"), input_int)
        np.save(os.path.join(self.model_dir,"sample_embeddings_"+self.outname+".npy"), embeddings_np)
        np.save(os.path.join(self.model_dir,"sampleindex_embeddings_"+self.outname+".npy"), sampleid) 
        gc.collect()

        self.sample_id = sampleid
        self.sample_embeddings = embeddings_np

        if similarity=="cosine":
            sim_matrix = cosine_similarity(embeddings.cpu(), np.array(self.topic_embeddings_))

            # 归一化相似性值，使其总和为1
            adjusted_similarities = [(s + 1) / 2 for s in sim_matrix]
            total_similarity = sum(adjusted_similarities)
            normalized_similarities = [s / total_similarity for s in adjusted_similarities]
            normalized_similarities = np.array(normalized_similarities).reshape(1, -1)
            predictions = np.argmax(normalized_similarities, axis=1) - self._outliers

            if self.calculate_probabilities:
                probabilities = normalized_similarities
            else:
                probabilities = np.max(normalized_similarities, axis=1)

        elif similarity=="euclidean":
            distances = [euclidean(embeddings.cpu().flatten(), b.flatten()) for b in np.array(self.topic_embeddings_)]
            similarities = [1 / (d + 1e-10) for d in distances]  # 添加小常数防止除零
            total_similarity = sum(similarities)
            normalized_similarities = [s / total_similarity for s in similarities]
            normalized_similarities = np.array(normalized_similarities).reshape(1, -1)
            predictions = np.argmax(normalized_similarities, axis=1) - self._outliers

            if self.calculate_probabilities:
                probabilities = normalized_similarities
            else:
                probabilities = np.max(normalized_similarities, axis=1)

        else:
            if type(self.hdbscan_model) == HDBSCAN:
                umap_embeddings = self.umap_model.transform(torch.tensor(embeddings_np))

                if is_supported_hdbscan(self.hdbscan_model):
                    predictions, probabilities = hdbscan_delegator(
                        self.hdbscan_model, "approximate_predict", umap_embeddings
                    )

                    if self.calculate_probabilities:
                        probabilities = hdbscan_delegator(self.hdbscan_model, "membership_vector", umap_embeddings)
                else:
                    predictions = self.hdbscan_model.predict(umap_embeddings)
                    probabilities = None

                probabilities = self._map_probabilities(probabilities, original_topics=True)
                predictions = self._map_predictions(predictions)

        self.predictions = predictions
        self.topic_sample_mat = probabilities.T
        topic_sample_file = os.path.join(self.model_dir, "topic_sample_mat_"+self.outname+".npz")
        np.savez(topic_sample_file, self.topic_sample_mat)
        
        celltype_topic_norm_df = np.divide(self.topic_celltype_mat.values.T, np.array([self.topic_celltype_mat.values.T.sum(axis=0)]))
        celltype_sample_array = np.dot(celltype_topic_norm_df[:, 0:self.topic_sample_mat.shape[0]],self.topic_sample_mat)
        sample_celltype_array = celltype_sample_array.transpose()

        sample_celltype_array_df = pd.DataFrame(sample_celltype_array,index=self.sample_id.tolist(),columns=self.topic_celltype_mat.columns)
        sample_celltype_array_norm = np.divide(sample_celltype_array, np.array([sample_celltype_array.sum(axis=1)]).T)
        sample_celltype_array_norm_df = pd.DataFrame(sample_celltype_array_norm,index=self.sample_id.tolist(),columns=self.topic_celltype_mat.columns)

        return sample_celltype_array_df, sample_celltype_array_norm_df
    
    def get_topic_freq(self, topic: int = None) -> Union[pd.DataFrame, int]:
        check_is_fitted(self)
        if isinstance(topic, int):
            return self.topic_sizes_[topic]
        else:
            return pd.DataFrame(self.topic_sizes_.items(), columns=["Topic", "Count"]).sort_values(
                "Count", ascending=False
            )

    def get_topics(self, full: bool = False) -> Mapping[str, Tuple[str, float]]:
        check_is_fitted(self)

        if full:
            topic_representations = {"Main": self.topic_representations_}
            topic_representations.update(self.topic_aspects_)
            return topic_representations
        else:
            return self.topic_representations_

    def get_topic(self, topic: int, full: bool = False) -> Union[Mapping[str, Tuple[str, float]], bool]:
        """Return top n words for a specific topic and their c-TF-IDF scores.

        Arguments:
            topic: A specific topic for which you want its representation
            full: If True, returns all different forms of topic representations
                  for a topic, including aspects

        Returns:
            The top n words for a specific word and its respective c-TF-IDF scores

        Examples:
        ```python
        topic = topic_model.get_topic(12)
        ```
        """
        check_is_fitted(self)
        if topic in self.topic_representations_:
            if full:
                representations = {"Main": self.topic_representations_[topic]}
                aspects = {aspect: representations[topic] for aspect, representations in self.topic_aspects_.items()}
                representations.update(aspects)
                return representations
            else:
                return self.topic_representations_[topic]
        else:
            return False

    def hierarchical_topics(
        self,
        documents_meta,
        top_n_words=30,
        use_ctfidf: bool = True,
        linkage_function: Callable[[csr_matrix], np.ndarray] = None,
        distance_function: Callable[[csr_matrix], csr_matrix] = None,
    ) -> pd.DataFrame:
        
        self.top_n_words=top_n_words

        if distance_function is None:
            distance_function = lambda x: 1 - cosine_similarity(x)

        if linkage_function is None:
            linkage_function = lambda x: sch.linkage(x, "ward", optimal_ordering=True)

        # Calculate distance
        embeddings = select_topic_representation(self.c_tf_idf_, self.topic_embeddings_, use_ctfidf)[0][self._outliers:]
        # embeddings = np.nan_to_num(embeddings, nan=0.0)

        X = distance_function(embeddings)
        # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # X = imputer.fit_transform(X)
        X = validate_distance_matrix(X, embeddings.shape[0])

        # Use the 1-D condensed distance matrix as an input instead of the raw distance matrix
        Z = linkage_function(X)

        # Ensuring that the distances between clusters are unique otherwise the flatting of the hierarchy with
        # `sch.fcluster(...)` would produce incorrect values for "Topics" for these clusters
        if len(Z[:, 2]) != len(np.unique(Z[:, 2])):
            Z[:, 2] = get_unique_distances(Z[:, 2])

        # Calculate basic bag-of-words to be iteratively merged later
        documents_per_topic = documents_meta.groupby("Topic", as_index=False).agg({"ID": lambda x: list(x)})
        documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.

        bow  = copy.deepcopy(self.input_int)
        words = copy.deepcopy(self.use_genes)
        
        # Extract clusters
        hier_topics = pd.DataFrame(
            columns=[
                "Parent_ID",
                "Parent_Name",
                "Topics",
                "Child_Left_ID",
                "Child_Left_Name",
                "Child_Right_ID",
                "Child_Right_Name",
            ]
        )

        for index in range(len(Z)):
            # Find clustered documents
            clusters = sch.fcluster(Z, t=Z[index][2], criterion="distance") - self._outliers
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

            # Group bow per cluster, calculate c-TF-IDF and extract words
            grouped = csr_matrix(bow[clustered_topics].sum(axis=0))
            c_tf_idf = self.ctfidf_model.transform(grouped)

            selection = documents_meta.loc[documents_meta.Topic.isin(clustered_topics), :]
            selection.Topic = 0

            words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf)# calculate_aspects=False

            # Extract parent's name and ID
            parent_id = index + len(clusters)
            parent_name = "_".join([x[0] for x in words_per_topic[0]][:5])

            # Extract child's name and ID
            Z_id = Z[index][0]
            child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_left_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
            else:
                child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

            # Extract child's name and ID
            Z_id = Z[index][1]
            child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_right_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
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
        orientation: str = "left",
        topics: List[int] = None,
        top_n_topics: int = None,
        use_ctfidf: bool = True,
        custom_labels: bool = False,
        title: str = "<b>Hierarchical Clustering</b>",
        width: int = 1000,
        height: int = 600,
        hierarchical_topics: pd.DataFrame = None,
        linkage_function: Callable[[csr_matrix], np.ndarray] = None,
        distance_function: Callable[[csr_matrix], csr_matrix] = None,
        color_threshold: int = 1,
    ) -> go.Figure:

        check_is_fitted(self)
        return visualize_hierarchy(
            self,
            orientation=orientation,
            topics=topics,
            top_n_topics=top_n_topics,
            use_ctfidf=use_ctfidf,
            custom_labels=custom_labels,
            title=title,
            width=width,
            height=height,
            hierarchical_topics=hierarchical_topics,
            linkage_function=linkage_function,
            distance_function=distance_function,
            color_threshold=color_threshold,
        )
    
    def compute_umass_coherence(self, top_n=30, epsilon=1e-12):
        return self._umass_coherence(self.gene_topic_mat.T,self.input_int,top_n,epsilon)

    def _umass_coherence(self,topic_word_matrix,X, top_n=30, epsilon=1e-12):
        # Get the top N words for each topic
        top_words = get_top_words(topic_word_matrix, top_n)

        # Convert document-term matrix to a binary matrix (word appears or not in each document)
        binary_matrix = (X > 0).astype(int)

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
                        freq_wj = word_doc_freq[words[j]]
                        freq_wi_wj = pair_doc_freq[(words[i], words[j])]
                        score += np.log((freq_wi_wj + epsilon) / (freq_wj + epsilon))
                coherence_scores.append(score)

        # Return the average coherence score
        return np.mean(coherence_scores)
    
    def _cv_coherence(self,topic_word_matrix,gene2vec_file, top_n=30):
        g2v = np.load( gene2vec_file)
        top_words = get_top_words(topic_word_matrix, top_n)

        coherence_scores = []

        for words in top_words:
            word_vecs = []
            for w in words:
                word_vecs.append(g2v[w, :].flatten()[np.newaxis, :])  # 提取单词向量
                # if  self.encodings:
                #     word_vecs.append(embeddings[:, w+1].toarray().flatten()[np.newaxis, :])  # 提取单词向量
                # else:
                #     word_vecs.append(topic_word_matrix[:, w].toarray().flatten()[np.newaxis, :])

            if len(word_vecs) < 2:
                continue
            
            word_vecs = np.vstack(word_vecs)  # 转换为矩阵格式
            sim_matrix = cosine_similarity(word_vecs)

            upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            coherence_scores.append(np.mean(upper_triangle))

        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def approximate_distribution(
        self,
        documents=None,
        label=None,
        window: int = 4,
        stride: int = 1,
        min_similarity: float = 0.1,
        batch_size: int = 1000,
        padding: bool = False,
        use_embedding_model: bool = False,
        calculate_tokens: bool = False,
        separator: str = " ",
    ) -> Tuple[np.ndarray, Union[List[np.ndarray], None]]:
        """A post-hoc approximation of topic distributions across documents.
        """
        if documents is None:
            documents = self.documents
        if label is not None:
            ct_documents = []
            celltype = sorted(list(set(label)))
            for i in celltype:
                cell_ind = [j for j in range(len(label)) if i==label[j]]
                docs = []
                for j in cell_ind:
                    docs.append(documents[j])
                ct_documents.append(" ".join(docs))
            documents = ct_documents
    
        if batch_size is None:
            batch_size = len(documents)
            batches = 1
        else:
            batches = math.ceil(len(documents) / batch_size)

        topic_distributions = []
        topic_token_distributions = []

        for i in range(batches):
            doc_set = documents[i * batch_size : (i + 1) * batch_size]

            # Extract tokens
            analyzer = self.vectorizer_model.build_tokenizer()
            tokens = [analyzer(document) for document in doc_set]

            # Extract token sets
            all_sentences = []
            all_indices = [0]
            all_token_sets_ids = []

            for tokenset in tokens:
                if len(tokenset) < window:
                    token_sets = [tokenset]
                    token_sets_ids = [list(range(len(tokenset)))]
                else:
                    # Extract tokensets using window and stride parameters
                    stride_indices = list(range(len(tokenset)))[::stride]
                    token_sets = []
                    token_sets_ids = []
                    for stride_index in stride_indices:
                        selected_tokens = tokenset[stride_index : stride_index + window]

                        if padding or len(selected_tokens) == window:
                            token_sets.append(selected_tokens)
                            token_sets_ids.append(
                                list(
                                    range(
                                        stride_index,
                                        stride_index + len(selected_tokens),
                                    )
                                )
                            )

                    # Add empty tokens at the beginning and end of a document
                    if padding:
                        padded = []
                        padded_ids = []
                        t = math.ceil(window / stride) - 1
                        for i in range(math.ceil(window / stride) - 1):
                            padded.append(tokenset[: window - ((t - i) * stride)])
                            padded_ids.append(list(range(0, window - ((t - i) * stride))))

                        token_sets = padded + token_sets
                        token_sets_ids = padded_ids + token_sets_ids

                # Join the tokens
                sentences = [separator.join(token) for token in token_sets]
                all_sentences.extend(sentences)
                all_token_sets_ids.extend(token_sets_ids)
                all_indices.append(all_indices[-1] + len(sentences))

            # Calculate similarity between embeddings of token sets and the topics
            if use_embedding_model:
                embeddings = self._extract_embeddings(all_sentences, method="document", verbose=True)
                similarity = cosine_similarity(embeddings, self.topic_embeddings_[self._outliers :])

            # Calculate similarity between c-TF-IDF of token sets and the topics
            else:
                bow_doc = self.vectorizer_model.transform(all_sentences)
                c_tf_idf_doc = self.ctfidf_model.transform(bow_doc)
                similarity = cosine_similarity(c_tf_idf_doc, self.c_tf_idf_[self._outliers :])

            # Only keep similarities that exceed the minimum
            similarity[similarity < min_similarity] = 0

            # Aggregate results on an individual token level
            if calculate_tokens:
                topic_distribution = []
                topic_token_distribution = []
                for index, token in enumerate(tokens):
                    start = all_indices[index]
                    end = all_indices[index + 1]

                    if start == end:
                        end = end + 1

                    # Assign topics to individual tokens
                    token_id = [i for i in range(len(token))]
                    token_val = {index: [] for index in token_id}
                    for sim, token_set in zip(similarity[start:end], all_token_sets_ids[start:end]):
                        for token in token_set:
                            if token in token_val:
                                token_val[token].append(sim)

                    matrix = []
                    for _, value in token_val.items():
                        matrix.append(np.add.reduce(value))

                    # Take empty documents into account
                    matrix = np.array(matrix)
                    if len(matrix.shape) == 1:
                        matrix = np.zeros((1, len(self.topic_labels_) - self._outliers))

                    topic_token_distribution.append(np.array(matrix))
                    topic_distribution.append(np.add.reduce(matrix))

                topic_distribution = normalize(topic_distribution, norm="l1", axis=1)

            # Aggregate on a tokenset level indicated by the window and stride
            else:
                topic_distribution = []
                for index in range(len(all_indices) - 1):
                    start = all_indices[index]
                    end = all_indices[index + 1]

                    if start == end:
                        end = end + 1
                    group = similarity[start:end].sum(axis=0)
                    topic_distribution.append(group)
                topic_distribution = normalize(np.array(topic_distribution), norm="l1", axis=1)
                topic_token_distribution = None

            # Combine results
            topic_distributions.append(topic_distribution)
            if topic_token_distribution is None:
                topic_token_distributions = None
            else:
                topic_token_distributions.extend(topic_token_distribution)

        topic_distributions = np.vstack(topic_distributions)

        return topic_distributions, topic_token_distributions

class TopicMapper:
    def __init__(self, topics: List[int]):
        """Initialization of Topic Mapper.
        Arguments:
            topics: A list of topics per document
        """
        base_topics = np.array(sorted(set(topics)))
        topics = base_topics.copy().reshape(-1, 1)
        self.mappings_ = np.hstack([topics.copy(), topics.copy()]).tolist()

    def get_mappings(self, original_topics: bool = True) -> Mapping[int, int]:
        """Get mappings from either the original topics or
        the second-most recent topics to the current topics.
        Arguments:
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.
        Returns:
            mappings: The mappings from old topics to new topics

        Examples:
        To get mappings, simply call:
        ```python
        mapper = TopicMapper(topics)
        mappings = mapper.get_mappings(original_topics=False)
        ```
        """
        if original_topics:
            mappings = np.array(self.mappings_)[:, [0, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        else:
            mappings = np.array(self.mappings_)[:, [-3, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        return mappings

    def add_new_topics(self, mappings: Mapping[int, int]):
        """Add new row(s) of topic mappings.
        Arguments:
            mappings: The mappings to add
        """
        length = len(self.mappings_[0])
        for key, value in mappings.items():
            to_append = [key] + ([None] * (length - 2)) + [value]
            self.mappings_.append(to_append)
    
    def add_mappings(self, mappings: Mapping[int, int], topic_model: scTopic_BERT):
        """Add new column(s) of topic mappings.
        Arguments:
            mappings: The mappings to add
            topic_model: The topic model this TopicMapper belongs to
        """
        for topics in self.mappings_:
            topic = topics[-1]
            if topic in mappings:
                topics.append(mappings[topic])
            else:
                topics.append(-1)