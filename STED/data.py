import os
import h5py
import tables
import collections
import numpy as np
import anndata as ad
import pandas as pd

from scipy.sparse import (csr_matrix,csc_matrix)

FeatureBCMatrix = collections.namedtuple('FeatureBCMatrix', ['ids', 'names', 'barcodes', 'matrix'])

def search_file_a(dirPath, a,b):
    files=[]
    for currentFile in os.listdir(dirPath):
        if currentFile.endswith(b) and currentFile.startswith(a):
            File = os.path.join(dirPath,currentFile)
            files.append(File)
    return files

def getNewDataFrame(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def remove_zero_rows(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]

def remove_zero_cols(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    _, nonzero_col_indice = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_col_indice)
    return X[:,unique_nonzero_indice]

def read_10X_h5(filename):
    """Read 10X HDF5 files, support both gene expression and peaks."""
    with tables.open_file(filename, 'r') as f:
        try:
            group = f.get_node(f.root, 'matrix')
        except tables.NoSuchNodeError:
            print("Matrix group does not exist in this file.")
            return None
        feature_group = getattr(group, 'features')
        ids = getattr(feature_group, 'id').read()
        names = getattr(feature_group, 'name').read()
        barcodes = getattr(group, 'barcodes').read()
        data = getattr(group, 'data').read()
        indices = getattr(group, 'indices').read()
        indptr = getattr(group, 'indptr').read()
        shape = getattr(group, 'shape').read()
        matrix = csc_matrix((data, indices, indptr), shape=shape)
        return FeatureBCMatrix(ids, names, barcodes, matrix)

def read_count(count_file, separator = "tab"):
    """Read count table as matrix."""
    if separator == "tab":
        sep = "\t"
    elif separator == "space":
        sep = " "
    elif separator == "comma":
        sep = ","
    else:
        raise Exception("Invalid separator!")

    infile = open(count_file, 'r').readlines()
    barcodes = infile[0].strip().split(sep)
    features = []
    matrix = []
    for line in infile[1:]:
        line = line.strip().split(sep)
        features.append(line[0])
        matrix.append([float(t) for t in line[1:]])
    if len(barcodes) == len(matrix[0]) + 1:
        barcodes = barcodes[1:]

    return {"matrix": matrix, "features": features, "barcodes": barcodes}

def read_10X_h5ad(filename):
    """Read processed H5AD files, support analysed scRNA-seq matrix"""
    adata =ad.read_h5ad(filename)
    ids = list(range(len(adata.var_names)))
    names = adata.var_names
    barcodes = adata.obs_names
    matrix = adata.X
    return FeatureBCMatrix(ids, names, barcodes, matrix)

def write_10X_h5(filename, matrix, features, barcodes):
    """Write 10X HDF5 files, support both gene expression and peaks."""
    f = h5py.File(filename, 'w')
    datatype = "Gene"
    M = csc_matrix(matrix, dtype=np.float32)
    B = np.array(barcodes, dtype='|S200')
    P = np.array(features, dtype='|S100')
    FT = np.array([datatype]*len(features), dtype='|S100')
    mat = f.create_group('matrix')
    mat.create_dataset('barcodes', data=B)
    mat.create_dataset('data', data=M.data)
    mat.create_dataset('indices', data=M.indices)
    mat.create_dataset('indptr', data=M.indptr)
    mat.create_dataset('shape', data=M.shape)
    fet = mat.create_group('features')
    fet.create_dataset('id', data=P)
    fet.create_dataset('name', data=P)
    f.close()
    
def GeneScore_Process(gs_count_file, gs_scale_factor=None):
    # read spatial count file
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
        gs_count_mat = gs_count['Bulk'].tolist()
        gs_count_mat = csc_matrix(gs_count_mat, dtype=np.float32)
        gs_count_genes = gs_count.index.to_list()
        gs_count_samples = gs_count.columns.to_list()

    # scale the count matrix
#     count_per_sample = np.asarray(gs_count_mat.sum(axis=0))  # gene expression total count
#     count_per_sample = np.array(count_per_sample.tolist()[0])
#     idx = np.where(count_per_sample == 0)
#     count_per_sample = np.delete(count_per_sample, idx, axis=None)
#     gs_count_mat = remove_zero_cols(gs_count_mat)
#     gs_count_genes = [gs_count_genes[i] for i in range(len(gs_count_genes)) if i not in idx[0].tolist()]

#     if not gs_scale_factor:
#         gs_scale_factor = np.round(np.quantile(count_per_sample, 0.75) / 1000, 0) * 1000
#     r, c = gs_count_mat.nonzero()
#     count_per_sample_sp = csr_matrix(((1.0 / count_per_sample)[c], (r, c)), shape=(gs_count_mat.shape))
#     gs_count_scale_mat = gs_count_mat.multiply(count_per_sample_sp) * gs_scale_factor
#     gs_count_scale_mat = csc_matrix(gs_count_scale_mat)

    return ({"raw_matrix": gs_count_mat, "genes": gs_count_genes,
             "spots": gs_count_samples})