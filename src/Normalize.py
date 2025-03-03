import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import CountVectorizer

from .utils import *
from .data import *

def standardz_sample(x):
    pop_mean = x.mean(axis=0)
    pop_std = x.std(axis=0)+ np.spacing(1) # np.spacing(1) == np.finfo(np.float64).eps
    df = (x - pop_mean).divide(pop_std)
    df = df.replace(np.inf,np.nan)
    df = df.replace(-np.inf,np.nan)
    df = df.dropna()
    print('standardz population control')
    return df

def ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, cell_celltype_list):
    # calculate accuracy according to backward
    celltype_topic_df = topic_celltype_df.transpose()
    celltype_cell_array = np.dot(celltype_topic_df, topic_cell_mat.toarray())
    cell_celltype_array = celltype_cell_array.transpose()
    cell_celltype_array_norm = np.divide(cell_celltype_array, np.array([cell_celltype_array.sum(axis=1)]).T)
    cell_celltype_array_norm_df = pd.DataFrame(cell_celltype_array_norm).fillna(0)
    cell_celltype_array_norm_df.columns = celltype_topic_df.index
    cell_celltype_array_norm_df_macloc = cell_celltype_array_norm_df.idxmax(axis=1).tolist()
    iftrue = np.array(cell_celltype_array_norm_df_macloc) == np.array(cell_celltype_list)
    iftrue_num = np.flatnonzero(iftrue).shape[0]
    accuracy = iftrue_num / len(cell_celltype_array_norm_df_macloc)
    # kmeans clustering based on dimension reduction
    cell_topic_mat = topic_cell_mat.transpose()
    kmeans = KMeans(n_clusters=len(set(cell_celltype_list)), random_state=0, max_iter=1000).fit(cell_topic_mat)
    cell_kmeans_label = kmeans.labels_
    nmi = normalized_mutual_info_score(np.array(cell_celltype_list), cell_kmeans_label)

    return ({"nmi": nmi, "accuracy": accuracy, "celltype_prediction": cell_celltype_array_norm_df_macloc})


def ModelEvaluateNorm(topic_cell_mat, topic_celltype_df, cell_celltype_list):
    # calculate accuracy according to backward
    celltype_topic_df = topic_celltype_df.transpose()
    celltype_topic_norm_df = np.divide(celltype_topic_df, np.array([celltype_topic_df.sum(axis=0)]))
    celltype_topic_norm_df = celltype_topic_norm_df.fillna(0)
    celltype_cell_array = np.dot(celltype_topic_norm_df, topic_cell_mat.toarray())
    cell_celltype_array = celltype_cell_array.transpose()
    cell_celltype_array_norm = np.divide(cell_celltype_array, np.array([cell_celltype_array.sum(axis=1)]).T)
    cell_celltype_array_norm_df = pd.DataFrame(cell_celltype_array_norm).fillna(0)
    cell_celltype_array_norm_df.columns = celltype_topic_df.index
    cell_celltype_array_norm_df_macloc = cell_celltype_array_norm_df.idxmax(axis=1).tolist()
    iftrue = np.array(cell_celltype_array_norm_df_macloc) == np.array(cell_celltype_list)
    iftrue_num = np.flatnonzero(iftrue).shape[0]
    accuracy = iftrue_num / len(cell_celltype_array_norm_df_macloc)

    return ({"accuracy": accuracy, "celltype_prediction": cell_celltype_array_norm_df_macloc})


def ModelEvaluateNormBySD(topic_cell_mat, topic_celltype_df, cell_celltype_list):
    # calculate accuracy according to backward
    celltype_topic_df = topic_celltype_df.transpose()
    celltype_topic_norm_df = StandardScaler(with_mean=False).fit_transform(celltype_topic_df)
    celltype_topic_norm_df = pd.DataFrame(celltype_topic_norm_df, columns=celltype_topic_df.columns,
                                          index=celltype_topic_df.index)
    celltype_topic_norm_df = celltype_topic_norm_df.fillna(0)
    celltype_cell_array = np.dot(celltype_topic_norm_df, topic_cell_mat.toarray())
    cell_celltype_array = celltype_cell_array.transpose()
    cell_celltype_array_norm = np.divide(cell_celltype_array, np.array([cell_celltype_array.sum(axis=1)]).T)
    cell_celltype_array_norm_df = pd.DataFrame(cell_celltype_array_norm).fillna(0)
    cell_celltype_array_norm_df.columns = celltype_topic_df.index
    cell_celltype_array_norm_df_macloc = cell_celltype_array_norm_df.idxmax(axis=1).tolist()
    iftrue = np.array(cell_celltype_array_norm_df_macloc) == np.array(cell_celltype_list)
    iftrue_num = np.flatnonzero(iftrue).shape[0]
    accuracy = iftrue_num / len(cell_celltype_array_norm_df_macloc)

    return ({"accuracy": accuracy, "celltype_prediction": cell_celltype_array_norm_df_macloc})


def ModelEvaluateBayes(topic_cell_mat, topic_celltype_df, cell_celltype_list, celltype_num_dict, model_dir):
    # bayes
    topic_prob_array = np.array([topic_celltype_df.sum(axis=1)]).T
    topic_prob_array = topic_prob_array / topic_prob_array.sum()
    celltype_prob_list = []
    for celltype in celltype_num_dict:
        celltype_prob_list.append(celltype_num_dict[celltype] / len(cell_celltype_list))
    celltype_prob_array = np.array(celltype_prob_list)
    topic_celltype_prob = np.divide(np.multiply(topic_celltype_df, celltype_prob_array), topic_prob_array)
    celltype_topic_bayes_df = topic_celltype_prob.transpose().fillna(0)
    # calculate accuracy according to backward
    celltype_cell_array = np.dot(celltype_topic_bayes_df, topic_cell_mat.toarray())
    cell_celltype_array = celltype_cell_array.transpose()
    cell_celltype_array_norm = np.divide(cell_celltype_array, np.array([cell_celltype_array.sum(axis=1)]).T)
    cell_celltype_array_norm_df = pd.DataFrame(cell_celltype_array_norm).fillna(0)
    cell_celltype_array_norm_df.columns = celltype_topic_bayes_df.index
    cell_celltype_array_norm_df_macloc = cell_celltype_array_norm_df.idxmax(axis=1).tolist()
    iftrue = np.array(cell_celltype_array_norm_df_macloc) == np.array(cell_celltype_list)
    iftrue_num = np.flatnonzero(iftrue).shape[0]
    accuracy = iftrue_num / len(cell_celltype_array_norm_df_macloc)

    return ({"accuracy": accuracy, "celltype_topic_bayes_df": celltype_topic_bayes_df,
             "celltype_prediction": cell_celltype_array_norm_df_macloc})


def ModelEvaluateBayesNorm(topic_cell_mat, celltype_topic_bayes_df, cell_celltype_list):
    # bayes
    celltype_topic_norm_df = np.divide(celltype_topic_bayes_df, np.array([celltype_topic_bayes_df.sum(axis=0)]))
    celltype_topic_norm_df = celltype_topic_norm_df.fillna(0)
    # calculate accuracy according to backward
    celltype_cell_array = np.dot(celltype_topic_norm_df, topic_cell_mat.toarray())
    cell_celltype_array = celltype_cell_array.transpose()
    cell_celltype_array_norm = np.divide(cell_celltype_array, np.array([cell_celltype_array.sum(axis=1)]).T)
    cell_celltype_array_norm_df = pd.DataFrame(cell_celltype_array_norm).fillna(0)
    cell_celltype_array_norm_df.columns = celltype_topic_norm_df.index
    cell_celltype_array_norm_df_macloc = cell_celltype_array_norm_df.idxmax(axis=1).tolist()
    iftrue = np.array(cell_celltype_array_norm_df_macloc) == np.array(cell_celltype_list)
    iftrue_num = np.flatnonzero(iftrue).shape[0]
    accuracy = iftrue_num / len(cell_celltype_array_norm_df_macloc)

    return ({"accuracy": accuracy, "celltype_prediction": cell_celltype_array_norm_df_macloc})
