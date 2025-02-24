import os
import copy
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
import hdbscan
from hdbscan import HDBSCAN
import datetime as dt

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import plotly.io as pio
import plotly.graph_objects as go

from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from typing import  Union, Tuple,List,Mapping, Callable,Union
from scipy.cluster import hierarchy as sch

from .BertFunction import *
from .performer_pytorch import *
from .epiDecon import *
from .plot_utils import *

import gc
import leidenalg
import igraph as ig

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

def is_supported_hdbscan(model):
    """Check whether the input model is a supported HDBSCAN-like model."""
    if isinstance(model, HDBSCAN):
        return True

    str_type_model = str(type(model)).lower()
    if "cuml" in str_type_model and "hdbscan" in str_type_model:
        return True

    return False

def visualize_documents_celltype(
    topic_model,
    docs: List[str],
    topics: List[int] = None,
    ann_list: List[int] = None,
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    sample: float = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Documents and Topics</b>",
    width: int = 1200,
    height: int = 750,
):
    topic_per_doc = ann_list

    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine").fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        names = unique_topics
        # names = [
        #     f"{topic}_" + "_".join([word for word, value in topic_model.get_topic(topic)][:3])
        #     for topic in unique_topics
        # ]

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [
        None,
        None,
        selection.x.mean(),
        selection.y.mean(),
        "Other documents",
    ]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode="markers+text",
            name="other",
            showlegend=False,
            marker=dict(color="#CFD8DC", size=5, opacity=0.5),
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [
                    None,
                    None,
                    selection.x.mean(),
                    selection.y.mean(),
                    name,
                ]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode="markers+text",
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5),
                )
            )

    # Add grid in a 'plus' shape
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def save_figure(fig, file_path):
    """通用保存图形的函数"""
    if file_path.endswith('.html'):
        fig.write_html(file_path)
    else:
        fig.write_image(file_path,format="pdf")

class BaseDimensionalityReduction:
    def fit(self, X: np.ndarray = None):
        return self
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
    
class DimensionalityReduction:
    def __init__(self,cells,celltypes,n_neighbors=15,n_components=2,seed=9,reduced_embedding=False) :
        self.adata = ad.AnnData(obs=pd.DataFrame(zip(cells,celltypes)))
        self.n_neighbors =n_neighbors
        self.n_components = n_components
        self.seed = seed
        self.reduced_embedding=reduced_embedding

    def fit_transform(self, X):
        adata = self.adata.copy()
        adata.obsm = {"X_embeddings":X}
        sc.pp.neighbors(adata,use_rep="X_embeddings",n_neighbors=self.n_neighbors)
        sc.tl.umap(adata,n_components=self.n_components,random_state=self.seed)
        self.adata =adata
        if self.reduced_embedding:
            return adata.obsm['X_umap']
        else:
            return adata.obsp['connectivities']
    def fit(self, X):
        adata = self.adata.copy()
        adata.obsm = {"X_embeddings":X}
        sc.pp.neighbors(adata,use_rep="X_embeddings",n_neighbors=self.n_neighbors)
        sc.tl.umap(adata,n_components=self.n_components,random_state=self.seed)
        self.adata =adata
        if self.reduced_embedding:
            return adata.obsm['X_umap']
        else:
            return adata.obsp['connectivities']
    
    def transform(self, X):
        adata = self.adata.copy()
        adata.obsm = {"X_embeddings":X}
        sc.pp.neighbors(adata,use_rep="X_embeddings",n_neighbors=self.n_neighbors)
        sc.tl.umap(adata,n_components=self.n_components,random_state=self.seed)
        self.adata =adata
        if self.reduced_embedding:
            return adata.obsm['X_umap']
        else:
            return adata.obsp['connectivities']

    def transform_sample(self,X):
        umap_model = UMAP(n_neighbors=self.n_neighbors,n_components=self.n_components,
             min_dist=0.5,spread=1,random_state=self.seed)
        return umap_model.fit_transform(X)
    
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es["weight"] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        print(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g

class ClusterModel:
    def __init__(self,resolution=1,directed=True,seed=9) :
        self.clustering_args = dict()
        self.clustering_args["seed"] =seed
        self.clustering_args["resolution_parameter"] = resolution
        self.directed=directed
        self.partition_type = leidenalg.RBConfigurationVertexPartition

    def fit(self, X):
        clustering_args =self.clustering_args
        g = get_igraph_from_adjacency(X, directed=self.directed)
        part = leidenalg.find_partition(g, self.partition_type, **clustering_args)
        groups = np.array(part.membership)
        self.labels_ = groups.tolist()
        self.g = g
        return self
    def predict(self, X):
        return X

class scTopic_BERT():
    def __init__(self) :
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
        self.min_cluster_size =10
        self.top_n_words = 30
        self.verbose = False

    def setData(self,object,out_dir,outname=None,custom_labels_= None,ntopics_list = "auto",seed_multiplier=20):

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
        else:
            self.anchor_list = None

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_dir = os.path.join(out_dir, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.seed_multiplier =seed_multiplier
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
                          encodings=False,CLS = False,last_layer=0,tie=False,CLASS = 7,dropout=0., h_dim=128):

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
            self.outname = "Last"+str(last_layer)+"_"+self.outname

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
        
        #model.to_out = Identity(dropout=dropout, h_dim=h_dim, out_dim=len(self.celltypes))
        model.to_out = BioClassifier(seq_len=16907, embed_dim=200, num_classes=len(self.celltypes))
        
        ckpt = torch.load(pretrain_file,map_location=torch.device("cpu"))
        model.load_state_dict(ckpt['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.performer.net.layers[-2].parameters():
            param.requires_grad = True

        self.embedding_model = model

    def getClusterModel(self,supervise = True,cluster_method=None,resolution=0.4,
                        min_samples=3,min_cluster_size=50,metric='euclidean',
                        cluster_selection_method='eom',n_neighbors=15,n_components=2,seed=9):
        self.seed =seed

        if supervise:
            self.supervise = True
            umap_model = BaseDimensionalityReduction()
            hdbscan_model  = LogisticRegression(random_state=self.seed)
        else:
            if cluster_method is None:
                reduced_embedding=True
                hdbscan_model = HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size,
                                        metric= metric,cluster_selection_method=cluster_selection_method,
                                        prediction_data=True)
            else:
                reduced_embedding=False
                hdbscan_model = ClusterModel(resolution=resolution,directed=True,seed=self.seed)
            umap_model = DimensionalityReduction(self.cells,self.ann_list,n_neighbors=n_neighbors,
                                                 n_components=n_components,seed=seed,
                                                 reduced_embedding=reduced_embedding)
        self.umap_model = umap_model
        self.hdbscan_model = hdbscan_model
    
    def getCtfidfModel(self,ctfidf_model=None,bm25_weighting=True,reduce_frequent_words=True):
        if ctfidf_model is None:
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting)
        if self.anchor_list is not None:
            seed_words = []
            for i in self.anchor_list:
                seed_words += i
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting,
                                                 reduce_frequent_words=reduce_frequent_words,
                                                 seed_words=seed_words,
                                                 seed_multiplier=self.seed_multiplier)
            self.seed_words =seed_words
        self.ctfidf_model = ctfidf_model
    
    def getEmbedding(self,batch_size=50,input_df= None,seed_embedding=None,seed=9,device_name = "cuda:0"):
        self.device_name = device_name
        self.seed = seed
        torch.manual_seed(self.seed)
        device = torch.device(self.device_name)

        if input_df is None:
            input_df = self.input_df

        model = self.embedding_model.to(device)

        # if self.gene_select:
        #     gene_mask = [self.genes.index(i) for i in self.genes if i in self.use_genes]
        #     input_df[:,gene_mask]= 0
        #     self.outname = self.outname+"_"+"GeneSelect"

        dataset = SCDataset_forTransform(input_df,np.array(range(len(self.cells))),self.CLASS,torch.device("cpu"))
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

    def getDocs(self, sparse_matrix, use_genes=None):
        gene_names = self.genes
        docs = []
        
        if use_genes is not None:
            gene_to_index = {gene: idx for idx, gene in enumerate(gene_names) if gene in use_genes}
        else:
            gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}

        for row in range(sparse_matrix.shape[0]):
            start_idx = sparse_matrix.indptr[row]
            end_idx = sparse_matrix.indptr[row + 1]

            nonzero_vals = sparse_matrix.data[start_idx:end_idx]
            nonzero_cols = sparse_matrix.indices[start_idx:end_idx]

            doc_words = []
            for col, freq in zip(nonzero_cols, nonzero_vals):
                gene = gene_names[col]
                if gene in gene_to_index:
                    doc_words.extend([gene] * int(freq))
            
            docs.append(" ".join(doc_words))
        return docs
    
    def fit(self,embedding_file = None,cellid_file = None,use_bin = False,nr_topics=None):
        self.use_bin = use_bin

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

        if self.gene_select:
            docs = self.getDocs(self.input_int,self.use_genes)
        else:
            docs = self.getDocs(self.input_int)
        self.docs = docs

        topic_model = BERTopic(
            embedding_model = None,
            vectorizer_model = None,
            hdbscan_model = self.hdbscan_model,
            umap_model = self.umap_model,
            ctfidf_model = self.ctfidf_model)
        
        self.model = topic_model
        
        topic_model.nr_topics = nr_topics
        topics, probs  = topic_model.fit_transform(docs, embeddings=embeddings) #传入训练好的词向量
        self.representation_meta = topic_model.get_topic_info()
        self.ann_list = [self.ann_list[i] for i in cell_list]
        self.celltype_meta = topic_model.topics_per_class(docs, classes=self.ann_list)
        self.topics = topics

        # self.Plot(embeddings=embeddings,docs=docs,topics_per_class=self.celltype_meta)

        # cell_topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
        scaler = MinMaxScaler(feature_range=(0, 5))
        celltypes =  sorted(list(set(self.ann_list)))
        celltype_mats = []
        for celltype in celltypes:
            indices = [id_ for id_,i in enumerate(self.ann_list) if i==celltype]
            sub_matrix = self.input_int[indices, :]
            celltype_mat = np.array(sub_matrix.mean(axis=0).A1)
            matrix_reshaped = celltype_mat.reshape(-1, 1)
            scaled_matrix = scaler.fit_transform(matrix_reshaped)
            scaled_matrix = scaled_matrix.reshape(celltype_mat.shape)
            celltype_mats.append(scaled_matrix.astype(int))
        celltype_int = csr_matrix(np.array(celltype_mats).astype(int))
        self.celltypes = celltypes

        all_mat = np.array(self.input_int.mean(axis=0).A1)
        all_int = csr_matrix(scaler.fit_transform(all_mat.reshape(-1, 1)).reshape(all_mat.shape).astype(int))

        if self.gene_select:
            celltype_docs = self.getDocs(celltype_int,self.use_genes)
            all_docs = self.getDocs(all_int,self.use_genes)
        else:
            celltype_docs = self.getDocs(celltype_int)
            all_docs = self.getDocs(all_int)

        # celltype_topic_mat,_= topic_model.approximate_distribution(celltype_docs,calculate_tokens=False)
        # _,gene_topic_mat= topic_model.approximate_distribution(all_docs,calculate_tokens=True)

        self.gene_topic_mat = np.nan_to_num(topic_model.c_tf_idf_.T, nan=0)
        self.unique_topics_ = sorted(set(list(topic_model.topics_)))
        topic_model.vectorizer_model.get_feature_names_out().tolist()
        self.use_genes = [i.upper() for i in topic_model.vectorizer_model.get_feature_names_out().tolist()]
        gene_topic_df = pd.DataFrame(self.gene_topic_mat.todense(),
                                     index =self.use_genes,columns=["Topic%s" %i for i in self.unique_topics_])
        gene_topic_df.to_csv(os.path.join(self.model_dir, 
                                          "gene_topic_mat_"+self.outname+".txt"),sep="\t")

        if self.supervise:
            self.sc_celltype_prediction = topic_model.topics_
            self.topic_celltype_mat = cm.T
            self.ntopics_selection = len(set(topic_model.topics_))
            pd.DataFrame(self.topic_celltype_mat).to_csv(os.path.join(self.model_dir, "topic_celltype_mat_"+self.outname+".txt"),sep="\t")
            
        cm= pd.crosstab(topic_model.topics_, self.ann_list)
        topic_celltype_mat = cm.div(cm.sum(axis=0), axis=1)
        self.topic_celltype_mat = topic_celltype_mat.loc[self.unique_topics_,:]
        pd.DataFrame(self.topic_celltype_mat).to_csv(os.path.join(self.model_dir, "topic_celltype_mat_"+self.outname+".txt"),sep="\t")

        self.gene_topic_mat = gene_topic_df.values
        self.celltype_topic_mat =  topic_celltype_mat.T

    def Plot(
        self,
        embeddings=None,
        docs=None,
        topics_per_class=None,
        linkage_function=None,
    ):
        linkage_function = linkage_function or (lambda x: sch.linkage(x, 'single', optimal_ordering=True))
        # topics = sorted(list(self.model.get_topics().keys()))
        if docs is not None:
            # 生成层次主题图
            self.hierarchical_topics = self.model.hierarchical_topics(docs, linkage_function=linkage_function)
            save_figure(self.model.visualize_hierarchy(hierarchical_topics=self.hierarchical_topics), 
                        os.path.join(self.model_dir, "topics_hierarchy_fig.pdf"))
            
            if embeddings is not None:
                # 使用 UMAP 降维
                
                reduced_embeddings = DimensionalityReduction(self.cells,self.ann_list,reduced_embedding=True).fit_transform(embeddings)
                #UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

                # 生成 UMAP 可视化图
                save_figure(self.model.visualize_documents(docs, reduced_embeddings=reduced_embeddings), 
                            os.path.join(self.model_dir, "topics_umap_fig.pdf"))

                # 生成细胞类型的 UMAP 可视化图
                save_figure(visualize_documents_celltype(
                                topic_model=self.model,
                                docs=docs,
                                ann_list=self.ann_list,
                                reduced_embeddings=reduced_embeddings),
                            os.path.join(self.model_dir, "celltype_umap_fig.pdf"))

        if topics_per_class is not None:
            # 生成主题分类可视化图
            save_figure(self.model.visualize_topics_per_class(topics_per_class), 
                        os.path.join(self.model_dir, "topics_per_celltype_fig.html"))

        # 生成主题间距离图
        save_figure(self.model.visualize_topics(), 
                    os.path.join(self.model_dir, "Intertopic_distance_fig.html"))

        # 生成主题基因图
        save_figure(self.model.visualize_barchart(), 
                    os.path.join(self.model_dir, "topics_genes_fig.pdf"))

    def transform(self,input_df,batch_size=50,seed=9,similarity="cluster",device_name="cuda:0"):
        self.seed = seed
        torch.manual_seed(self.seed)
        device = torch.device(device_name)
        model = self.embedding_model.to(device)
        sampel_num = input_df.shape[0]

        if sampel_num<batch_size:
            batch_size=sampel_num

        # if self.gene_select:
        #     gene_mask = [self.genes.index(i) for i in self.genes if i in self.use_genes]
        #     input_df[:,gene_mask]= 0

        dataset = SCDataset_forTransform(input_df,np.array(range(sampel_num)),self.CLASS,torch.device("cpu"))
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
        try:
            self.sample_embeddings = embeddings_np.cpu()
        except:
            self.sample_embeddings = embeddings_np

        if similarity=="cluster":
            if type(self.model.hdbscan_model) == HDBSCAN:
                umap_embeddings = self.model.umap_model.transform_sample(X=self.sample_embeddings)

                if is_supported_hdbscan(self.model.hdbscan_model):
                    predictions, probabilities = hdbscan_delegator(
                        self.model.hdbscan_model, "approximate_predict", umap_embeddings
                    )

                    if self.calculate_probabilities:
                        probabilities = hdbscan_delegator(self.model.hdbscan_model, "membership_vector", umap_embeddings)
                else:
                    predictions = self.model.hdbscan_model.predict(umap_embeddings)
                    probabilities = None

                probabilities = self.model._map_probabilities(probabilities, original_topics=True)
                predictions = self.model._map_predictions(predictions)
        else:
            similarity ="cosine"
        
        if similarity=="cosine":
            sim_matrix = cosine_similarity(self.sample_embeddings, np.array(self.model.topic_embeddings_))

            # 归一化相似性值，使其总和为1
            # adjusted_similarities = [(s + 1) / 2 for s in sim_matrix]
            # total_similarity = sum(adjusted_similarities)
            # normalized_similarities = [s / total_similarity for s in adjusted_similarities]
            # normalized_similarities = np.array(normalized_similarities).reshape(1, -1)

            if not isinstance(sim_matrix, np.ndarray):
                sim_matrix = np.array(sim_matrix)
            sim_matrix = sim_matrix[:,self.model._outliers:]
            column_sums = sim_matrix.sum(axis=0)
            epsilon = 1e-10
            column_sums = np.where(column_sums == 0, epsilon, column_sums)
            normalized_similarities = sim_matrix / column_sums[np.newaxis, :]
            normalized_similarities = np.nan_to_num(normalized_similarities, nan=0.0, posinf=0.0, neginf=0.0)
            predictions = np.argmax(normalized_similarities, axis=1) - self.model._outliers

            if self.calculate_probabilities:
                probabilities = normalized_similarities#[:,self.model._outliers:]
            else:
                probabilities = np.max(normalized_similarities, axis=1)

        elif similarity=="euclidean":
            distances = [euclidean(self.sample_embeddings.flatten(), b.flatten()) for b in np.array(self.model.topic_embeddings_)]
            similarities = [1 / (d + 1e-10) for d in distances]  # 添加小常数防止除零
            total_similarity = sum(similarities)
            normalized_similarities = [s / total_similarity for s in similarities]
            normalized_similarities = np.array(normalized_similarities).reshape(1, -1)
            predictions = np.argmax(normalized_similarities, axis=1) - self.model._outliers

            if self.calculate_probabilities:
                probabilities = normalized_similarities[:,self.model._outliers:]
            else:
                probabilities = np.max(normalized_similarities, axis=1)

        self.predictions = predictions
        self.topic_sample_mat = probabilities.T
        topic_sample_file = os.path.join(self.model_dir, "topic_sample_mat_"+self.outname+".npz")
        np.savez(topic_sample_file, self.topic_sample_mat)
        
        celltype_topic_norm_df = np.divide(self.celltype_topic_mat, np.array([self.celltype_topic_mat.sum(axis=0)])+1e-10)
        try:
            celltype_sample_array = np.dot(celltype_topic_norm_df[:, self.model._outliers:],self.topic_sample_mat)
        except:
            celltype_sample_array = np.dot(celltype_topic_norm_df.iloc[:, self.model._outliers:],self.topic_sample_mat)
        sample_celltype_array = celltype_sample_array.transpose()

        sample_celltype_array_df = pd.DataFrame(sample_celltype_array,index=self.sample_id.tolist(),columns=self.celltypes)
        sample_celltype_array_norm = np.divide(sample_celltype_array, np.array([sample_celltype_array.sum(axis=1)]).T+1e-10)
        sample_celltype_array_norm_df = pd.DataFrame(sample_celltype_array_norm,index=self.sample_id.tolist(),columns=self.celltypes)

        return sample_celltype_array_df, sample_celltype_array_norm_df