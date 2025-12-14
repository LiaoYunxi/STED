import copy
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from combat.pycombat import pycombat

def array_imputer(df,threshold=0.9,strategy="median",trim=1.0,batch=False,lst_batch=[], trim_red=True):
    """
    imputing nan and trim the values less than 1
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    threshold: float, default 0.9
        determine whether imupting is done or not dependent on ratio of not nan
        
    strategy: str, default median
        indicates which statistics is used for imputation
        candidates: "median", "most_frequent", "mean"
    
    lst_batch : lst, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    Returns
    ----------
    res: a dataframe
    
    """
    df_c = copy.deepcopy(df)
    if (type(trim)==float) or (type(trim)==int):
        df_c = df_c.where(df_c > trim)
    else:
        pass
    df_c = df_c.replace(0,np.nan)
    if batch:
        lst = []
        ap = lst.append
        for b in range(max(lst_batch)+1):
            place = [i for i, x in enumerate(lst_batch) if x == b]
            print("{0} ({1} sample)".format(b,len(place)))
            temp = df_c.iloc[:,place]
            if temp.shape[1]==1:
                ap(pd.DataFrame(temp))
            else:
                thresh = int(threshold*float(len(list(temp.columns))))
                temp = temp.dropna(thresh=thresh)
                imr = SimpleImputer(strategy=strategy)
                imputed = imr.fit_transform(temp.values.T) # impute in columns
                ap(pd.DataFrame(imputed.T,index=temp.index,columns=temp.columns))
        if trim_red:
            df_res = pd.concat(lst,axis=1)
            df_res = df_res.replace(np.nan,0) + 1
            print("redundancy trimming")
        else:
            df_res = pd.concat(lst,axis=1,join="inner")
    else:            
        thresh = int(threshold*float(len(list(df_c.columns))))
        df_c = df_c.dropna(thresh=thresh)
        imr = SimpleImputer(strategy=strategy)
        imputed = imr.fit_transform(df_c.values.T) # impute in columns
        df_res = pd.DataFrame(imputed.T,index=df_c.index,columns=df_c.columns)
    return df_res

def trimming(df, log=True, trimming=True, batch=False, lst_batch=[], trim_red=False, threshold=0.9):
    df_c = copy.deepcopy(df)
    # same index median
    df_c.index = [str(i) for i in df_c.index]
    df2 = pd.DataFrame()
    dup = df_c.index[df_c.index.duplicated(keep="first")]
    gene_list = pd.Series(dup).unique().tolist()
    if len(gene_list) != 0:
        for gene in gene_list:
            new = df_c.loc[:,gene].median()
            df2.loc[gene] = new
        df_c = df_c.drop(gene_list)
        df_c = pd.concat([df_c,df2.T])
    
    if trimming:
        if len(df_c.T) != 1:    
            df_c = array_imputer(df_c,lst_batch=lst_batch,batch=batch,trim_red=trim_red,threshold=threshold)
        else:
            df_c = df_c.where(df_c>1)
            df_c = df_c.dropna()
    else:
        df_c = df_c.dropna()

    # log conversion
    if log:
        df_c = df_c.where(df_c>=0)
        df_c = df_c.dropna()
        df_c = np.log2(df_c+1)
    else:
        pass
    return df_c

# TODO: 注释掉了
# def batch_norm(df,lst_batch=[]):
#     """
#     batch normalization with combat
    
#     Parameters
#     ----------
#     df: a dataframe # genes X samples
#         a dataframe to be analyzed
    
#     lst_batch : lst, int
#         indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
#     """
#     comb_df = pycombat(df,lst_batch)
#     return comb_df

# def multi_batch_norm(df,lst_lst_batch=[[],[]],do_plots=True):
#     """
#     batch normalization with combat for loop
    
#     Note that the order of normalization is important. Begin with the broadest batch and move on to more specific batches of corrections.
    
#     e.g. sex --> area --> country
    
#     Parameters
#     ----------
#     df: a dataframe: genes X samples
#         a dataframe to be analyzed
    
#     lst_batch : lst, int
#         indicates batch like : [[0,0,1,1,1,1],[0,0,1,1,2,2]]
    
#     """
#     df_c = df.copy() # deep copy
#     for lst_batch in tqdm(lst_lst_batch):
#         comb = batch_norm(df_c,lst_batch)
#         df_c = comb # update
#         if do_plots:
#             for i in range(5):
#                 plt.hist(df_c.iloc[:,i],bins=200,alpha=0.8)
#             plt.show()
#         else:
#             pass
#     return df_c

# def quantile(df,method="median"):
#     """
#     quantile normalization of dataframe (variable x sample)
    
#     Parameters
#     ----------
#     df: dataframe genes X samples
#         a dataframe subjected to QN
    
#     method: str, default "median"
#         determine median or mean values are employed as the template    

#     """
#     #print("quantile normalization (QN)")
#     df_c = df.copy() # deep copy
#     lst_index = list(df_c.index)
#     lst_col = list(df_c.columns)
#     n_ind = len(lst_index)
#     n_col = len(lst_col)

#     ### prepare mean/median distribution
#     x_sorted = np.sort(df_c.values,axis=0)[::-1]
#     if method=="median":
#         temp = np.median(x_sorted,axis=1)
#     else:
#         temp = np.mean(x_sorted,axis=1)
#     temp_sorted = np.sort(temp)[::-1]

#     ### prepare reference rank list
#     x_rank_T = np.array([rankdata(v,method="ordinal") for v in df_c.T.values])

#     ### conversion
#     rank = sorted([v + 1 for v in range(n_ind)],reverse=True)
#     converter = dict(list(zip(rank,temp_sorted)))
#     converted = []
#     converted_ap = converted.append  
#     for i in range(n_col):
#         transient = [converter[v] for v in list(x_rank_T[i])]
#         converted_ap(transient)

#     np_data = np.matrix(converted).T
#     df2 = pd.DataFrame(np_data)
#     df2.index = lst_index
#     df2.columns = lst_col
#     return df2

# def quantile_sparse(sparse_matrix, method="median"):
#     """
#     Quantile normalization for a sparse matrix (csr_matrix).
    
#     Parameters
#     ----------
#     sparse_matrix : csr_matrix
#         A sparse matrix (genes x samples) to be quantile normalized.
    
#     method : str, default "median"
#         The method to use for quantile normalization. Either "median" or "mean".
    
#     Returns
#     -------
#     csr_matrix
#         Quantile normalized sparse matrix.
#     """
#     # Convert sparse matrix to a dense matrix for processing
#     dense_matrix = sparse_matrix.toarray()

#     # Prepare mean/median distribution
#     x_sorted = np.sort(dense_matrix, axis=0)[::-1]
#     if method == "median":
#         temp = np.median(x_sorted, axis=1)
#     else:
#         temp = np.mean(x_sorted, axis=1)
#     temp_sorted = np.sort(temp)[::-1]

#     # Prepare reference rank list
#     x_rank_T = np.array([rankdata(v, method="ordinal") for v in dense_matrix.T])

#     # Conversion
#     rank = sorted([v + 1 for v in range(dense_matrix.shape[0])], reverse=True)
#     converter = dict(list(zip(rank, temp_sorted)))

#     converted = np.zeros_like(dense_matrix)
#     for i in range(dense_matrix.shape[1]):
#         transient = [converter[v] for v in list(x_rank_T[i])]
#         converted[:, i] = transient

#     # Convert back to sparse matrix format
#     sparse_normalized = csr_matrix(converted)
    
#     return sparse_normalized

def quantile(df, method="mean"):
    """
    Correct quantile normalization for a DataFrame (genes x samples).
    
    Parameters
    ----------
    df : pd.DataFrame
        Expression matrix with shape (genes, samples).
    method : str, default "mean"
        "mean" or "median" for reference distribution.
    
    Returns
    -------
    pd.DataFrame
        Quantile-normalized matrix with same index/columns as input.
    """
    mat = df.values  # (n_genes, n_samples)
    
    # Step 1: Sort each column (sample) in ascending order
    sorted_mat = np.sort(mat, axis=0)
    
    # Step 2: Compute reference distribution across samples
    if method == "median":
        ref = np.median(sorted_mat, axis=1)
    else:
        ref = np.mean(sorted_mat, axis=1)
    
    # Step 3: For each sample, map original values to reference by rank
    normalized = np.empty_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        # Use 'min' to assign same rank to ties
        ranks = rankdata(col, method="min").astype(int) - 1  # 0-based
        normalized[:, j] = ref[ranks]
    
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)

def batch_norm(df, lst_batch=[]):
    # df: genes x samples → transpose to samples x genes
    df_t = df.T
    comb_t = pycombat(df_t, lst_batch)  # assuming pycombat expects samples x genes
    return comb_t.T  # back to genes x samples


def multi_batch_norm(df, lst_lst_batch=[[], []], do_plots=True):
    """
    Sequential batch correction using ComBat.
    Assumes df is genes x samples.
    """
    df_c = df.copy()
    
    for i, lst_batch in enumerate(lst_lst_batch):
        print(f"Correcting batch level {i+1}...")
        df_c = batch_norm(df_c, lst_batch)  # ensure batch_norm handles genes x samples correctly
        
        if do_plots:
            # Better: plot PCA or sample distributions
            plt.figure(figsize=(8, 4))
            for j in range(min(5, df_c.shape[1])):
                plt.hist(df_c.iloc[:, j], bins=100, alpha=0.6, label=f'Sample {j}')
            plt.title(f'After batch correction {i+1}')
            plt.legend()
            plt.show()
    
    return df_c

def quantile_normalize_dense(matrix, method="mean"):
    """
    Correct quantile normalization for dense matrix (genes x samples).
    """
    # Step 1: Sort each column (sample) in ascending order
    sorted_mat = np.sort(matrix, axis=0)
    
    # Step 2: Compute reference distribution (mean or median across samples)
    if method == "median":
        ref_dist = np.median(sorted_mat, axis=1)
    else:
        ref_dist = np.mean(sorted_mat, axis=1)
    
    # Step 3: For each sample, replace values by reference based on rank
    normalized = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        # Use 'min' to handle ties: same value → same rank
        ranks = rankdata(col, method="min").astype(int) - 1  # 0-based index
        normalized[:, j] = ref_dist[ranks]
    
    return normalized

def quantile_sparse(sparse_matrix, method="mean", force=False):
    """
    Quantile normalization for sparse matrix (genes x samples).
    
    ⚠️ Warning: Converts to dense! Only use for small matrices.
    Not recommended for scRNA-seq due to biological heterogeneity.
    """
    if not force and sparse_matrix.shape[0] * sparse_matrix.shape[1] > 1e7:
        raise ValueError(
            "Matrix too large for quantile normalization (would require dense conversion). "
            "Consider alternative normalization (e.g., library size + log)."
        )
    
    dense = sparse_matrix.toarray()
    normalized_dense = quantile_normalize_dense(dense, method=method)
    return csr_matrix(normalized_dense)

def log2(df):
    f_add = lambda x: x+1
    log_df = df.apply(f_add)
    log_df = np.log2(log_df)
    return log_df

def log2_sparse(sparse_matrix):
    sparse_matrix.data = np.log2(sparse_matrix.data + 1)
    return sparse_matrix
    
def low_cut(df,threshold=1.0):
    df_c = copy.deepcopy(df)
    if (type(threshold)==float) or (type(threshold)==int):
        cut_df = df_c.where(df_c > threshold)
    else:
        pass
    return cut_df

def standardz_sample(x):
    pop_mean = x.mean(axis=0)
    pop_std = x.std(axis=0)+ np.spacing(1) # np.spacing(1) == np.finfo(np.float64).eps
    df = (x - pop_mean).divide(pop_std)
    df = df.replace(np.inf,np.nan)
    df = df.replace(-np.inf,np.nan)
    df = df.dropna()
    print('standardz population control')
    return df

def ctrl_norm(df,ctrl="C"):
    """normalization with ctrl samples"""
    ctrl_samples = []
    for t in df.index.tolist():
        if t.split("_")[0]==ctrl:
            ctrl_samples.append(t)
    ctrl_df = df.loc[ctrl_samples]
    
    ctrl_mean = ctrl_df.mean() # mean value of ctrl
    ctrl_std = ctrl_df.std() # std of ctrl
    
    norm_df = (df-ctrl_mean)/ctrl_std
    return norm_df

def drop_all_missing(df):
    replace = df.replace(0,np.nan)
    drop = replace.dropna(how="all") # remove rows whose all values are missing
    res = drop.fillna(0)
    print(len(df)-len(res),"rows are removed")
    return res

def freq_norm(df,anchor_dict,ignore_others=True):
    """
    Normalize by sum of exression
    ----------
    df : DataFrame
        Genes in row and samples in column.
             PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
    AIF1          9.388634       8.354677  ...       8.848500       9.149019
    AIM2          4.675251       4.630904  ...       4.830909       4.831925
    ALOX5AP       9.064822       8.891569  ...       9.420134       9.192017
    APBA2         4.313265       4.455105  ...       4.309868       4.338142
    APEX1         7.581810       7.994079  ...       7.604995       7.706539
                   ...            ...  ...            ...            ...
    VCAN          8.213386       7.018457  ...       9.050750       8.263430
    VIPR1         6.436875       6.281543  ...       5.973437       6.622016
    ZBTB16        4.687727       4.618193  ...       4.730128       4.546280
    ZFP36        12.016052      11.514114  ...      11.538242      12.271717
    ZNF101        5.288079       5.250802  ...       5.029970       5.141903
    
    anchor_dict : dict

    """
    others = sorted(list(set(df.index.tolist()) - set(itertools.chain.from_iterable(anchor_dict.values()))))
    if len(others)>0:
        other_dict = {'others':others}
        #anchor_dict = anchor_dict | other_dict # Python 3.9
        anchor_dict = {**anchor_dict,**other_dict}

    # normalize
    use_k = []
    use_v = []
    for i,k in enumerate(anchor_dict):
        if len(anchor_dict.get(k))>0:
            use_k.append(k)
            use_v.append(anchor_dict.get(k))
        else:
            pass
    anchor_dict = dict(zip(use_k,use_v))
    
    cell_sums = []
    for i,k in enumerate(anchor_dict):
        if ignore_others:
            if k == 'others':
                cell_sums.append(-1)
            else:
                common_v = sorted(list(set(anchor_dict.get(k)) & set(df.index.tolist())))
                tmp_df = df.loc[common_v] # expression of markers
                tmp_sum = tmp_df.T.sum() # sum of expression level
                cell_sum = sum(tmp_sum)
                cell_sums.append(cell_sum)
        else:
            common_v = sorted(list(set(anchor_dict.get(k)) & set(df.index.tolist())))
            tmp_df = df.loc[common_v] # expression of markers
            tmp_sum = tmp_df.T.sum() # sum of expression level
            cell_sum = sum(tmp_sum)
            cell_sums.append(cell_sum)
    
    base = max(cell_sums) # unify to maximum value
    r = [base/t for t in cell_sums]
    
    norm_df = pd.DataFrame()
    for i,k in enumerate(anchor_dict):
        common_v = sorted(list(set(anchor_dict.get(k)) & set(df.index.tolist())))
        tmp_df = df.loc[common_v] # expression of markers
        if ignore_others:
            if k == 'others':
                tmp_norm = tmp_df
            else:
                tmp_norm = tmp_df*r[i]
        else:
            tmp_norm = tmp_df*r[i]
        norm_df = pd.concat([norm_df,tmp_norm])
    
    # for multiple marker origin
    sample_name = norm_df.columns.tolist()[0]
    sort_norm = norm_df.sort_values(sample_name,ascending=False)

    # Gene duplications caused by multiple corrections are averaged.
    sort_norm['gene_name'] = sort_norm.index.tolist()
    trim_df= sort_norm.groupby("gene_name").mean() 

    #trim_df = sort_norm[~sort_norm.index.duplicated(keep='first')] # pick up the 1st one.
    return trim_df

def freq_norm_sparse(sparse_df, gene_names, anchor_dict, ignore_others=True):
    """
    Normalize by sum of expression for sparse matrix
    ----------
    sparse_df : csr_matrix
        Sparse matrix with genes in row and samples in column.
        
    gene_names : list
        List of gene names corresponding to the rows of the sparse matrix.
        
    anchor_dict : dict
        Dictionary specifying the anchors.
        
    ignore_others : bool, optional
        Whether to ignore "others" category, by default True
    """
    # Find 'others' category
    others = sorted(list(set(gene_names) - set(itertools.chain.from_iterable(anchor_dict.values()))))
    if len(others) > 0:
        other_dict = {'others': others}
        anchor_dict = {**anchor_dict, **other_dict}

    # Filter and prepare anchor_dict
    use_k = []
    use_v = []
    for k in anchor_dict:
        if len(anchor_dict[k]) > 0:
            use_k.append(k)
            use_v.append(anchor_dict[k])
    
    anchor_dict = dict(zip(use_k, use_v))
    
    # Normalize
    cell_sums = []
    for k in anchor_dict:
        if ignore_others and k == 'others':
            cell_sums.append(-1)
        else:
            common_v = sorted(list(set(anchor_dict[k]) & set(gene_names)))
            indices = [gene_names.index(gene) for gene in common_v if gene in gene_names]
            if len(indices) > 0:
                tmp_df = sparse_df[indices, :]  # select rows
                tmp_sum = np.array(tmp_df.sum(axis=0)).flatten()  # sum across rows (genes) for each sample
                cell_sum = tmp_sum.sum()  # sum across all samples
                cell_sums.append(cell_sum)
            else:
                cell_sums.append(0)

    base = max(cell_sums) if len(cell_sums) > 0 else 1  # avoid division by zero
    r = [base / t if t != -1 else 1 for t in cell_sums]
    
    # Normalize each group by the corresponding factor
    norm_data = []
    norm_index = []
    for i, k in enumerate(anchor_dict):
        common_v = sorted(list(set(anchor_dict[k]) & set(gene_names)))
        indices = [gene_names.index(gene) for gene in common_v if gene in gene_names]
        if len(indices) > 0:
            tmp_df = sparse_df[indices, :]
            if ignore_others and k == 'others':
                tmp_norm = tmp_df  # No normalization for 'others'
            else:
                tmp_norm = tmp_df.multiply(r[i])  # Element-wise multiplication for normalization
            norm_data.append(tmp_norm)
            norm_index.extend(common_v)

    # Concatenate normalized data
    if len(norm_data) > 0:
        norm_matrix = csr_matrix.vstack(norm_data)
    else:
        norm_matrix = csr_matrix((0, sparse_df.shape[1]))  # Empty matrix

    return norm_matrix,norm_index

def size_norm(df,marker_dic):
    """
    Normalize by gene size (number).
    ----------
    df : DataFrame
        Genes in row and samples in column.
             PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
    AIF1          9.388634       8.354677  ...       8.848500       9.149019
    AIM2          4.675251       4.630904  ...       4.830909       4.831925
    ALOX5AP       9.064822       8.891569  ...       9.420134       9.192017
    APBA2         4.313265       4.455105  ...       4.309868       4.338142
    APEX1         7.581810       7.994079  ...       7.604995       7.706539
                   ...            ...  ...            ...            ...
    VCAN          8.213386       7.018457  ...       9.050750       8.263430
    VIPR1         6.436875       6.281543  ...       5.973437       6.622016
    ZBTB16        4.687727       4.618193  ...       4.730128       4.546280
    ZFP36        12.016052      11.514114  ...      11.538242      12.271717
    ZNF101        5.288079       5.250802  ...       5.029970       5.141903
    
    marker_dic : dict

    """
    max_size = max([len(t) for t in marker_dic.values()])
    norm_df = pd.DataFrame()
    for i,k in enumerate(marker_dic):
        common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
        tmp_size = len(common_v)
        r = max_size / tmp_size
        tmp_df = df.loc[common_v] # expression of markers
        tmp_norm = tmp_df*r
        norm_df = pd.concat([norm_df,tmp_norm])
    return norm_df

def norm_total_res(total_res,base_names=['Monocytes', 'NK cells', 'B cells naive', 'B cells memory', 'T cells CD4 naive', 'T cells CD4 memory', 'T cells CD8', 'T cells gamma delta']):
    norm_total_res = []
    for tmp_df in total_res:
        tmp_df = tmp_df[base_names]
        tmp_sum = tmp_df.T.sum()
        r = 1/tmp_sum
        norm_res = (tmp_df.T*r).T
        norm_total_res.append(norm_res)
    return norm_total_res

def norm_val(val_df,base_names=['Naive B', 'Memory B', 'CD8 T', 'Naive CD4 T', 'Gamma delta T', 'NK', 'Monocytes']):
    tmp_df = val_df[base_names]
    tmp_sum = tmp_df.T.sum()
    r = 1/tmp_sum
    norm_res = (tmp_df.T*r).T
    return norm_res

def check_random_state(seed):
    if seed is None:
        # i.e., use existing RandomState
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("{} cannot be used as a random seed.".format(seed))

def matrix_to_lists(doc_word):  
    """Convert a (sparse) matrix of counts into arrays of word and doc indices  
  
    Parameters  
    ----------  
    doc_word : array or sparse matrix (D, V)  
        document-term matrix of counts  
  
    Returns  
    -------  
    (WS, DS) : tuple of two arrays  
        WS[k] contains the kth word in the corpus  
        DS[k] contains the document index for the kth word  
    """  
    sparse = False 
  
    try:  
        doc_word = doc_word.copy().tolil()  
        sparse = True  
    except AttributeError:  
        pass  

    if sparse and not np.issubdtype(doc_word.dtype, np.integer):  
        raise ValueError("expected sparse matrix with integer values, found float values")  
  
    ii, jj = np.nonzero(doc_word)  
   
    if sparse:  
        ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))  
    else:  
        ss = doc_word[ii, jj]  
   
    DS = np.repeat(ii, ss).astype(np.intc)  
    WS = np.repeat(jj, ss).astype(np.intc)  
  
    return WS, DS

def lists_to_matrix(WS, DS):
    """Convert array of word (or topic) and document indices to doc-term array

    Parameters
    -----------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    Returns
    -------
    doc_word : array (D, V)
        document-term array of counts

    """
    D = max(DS) + 1
    V = max(WS) + 1
    doc_word = np.zeros((D, V), dtype=np.intc)
    indices, counts = np.unique(list(zip(DS, WS)), axis=0, return_counts=True)
    doc_word[indices[:, 0], indices[:, 1]] += counts
    return doc_word

def dtm2ldac(dtm, offset=0):
    """Convert a document-term matrix into an LDA-C formatted file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    doclines : iterable of LDA-C lines suitable for writing to file

    Notes
    -----
    If a format similar to SVMLight is desired, `offset` of 1 may be used.
    """
    try:
        dtm = dtm.tocsr()
    except AttributeError:
        pass
    assert np.issubdtype(dtm.dtype, np.integer)
    n_rows = dtm.shape[0]
    for i, row in enumerate(dtm):
        try:
            row = row.toarray().squeeze()
        except AttributeError:
            pass
        unique_terms = np.count_nonzero(row)
        if unique_terms == 0:
            raise ValueError("dtm row {} has all zero entries.".format(i))
        term_cnt_pairs = [(i + offset, cnt) for i, cnt in enumerate(row) if cnt > 0]
        docline = str(unique_terms) + ' '
        docline += ' '.join(["{}:{}".format(i, cnt) for i, cnt in term_cnt_pairs])
        if (i + 1) % 1000 == 0:
            logger.info("dtm2ldac: on row {} of {}".format(i + 1, n_rows))
        yield docline

def ldac2dtm(stream, offset=0):
    """Convert an LDA-C formatted file to a document-term array

    Parameters
    ----------
    stream: file object
        File yielding unicode strings in LDA-C format.

    Returns
    -------
    dtm : array of shape N,V

    Notes
    -----
    If a format similar to SVMLight is the source, an `offset` of 1 may be used.
    """
    doclines = stream

    # We need to figure out the dimensions of the dtm.
    N = 0
    V = -1
    data = []
    for l in doclines:  # noqa
        l = l.strip()  # noqa
        # skip empty lines
        if not l:
            continue
        unique_terms = int(l.split(' ')[0])
        term_cnt_pairs = [s.split(':') for s in l.split(' ')[1:]]
        for v, _ in term_cnt_pairs:
            # check that format is indeed LDA-C with the appropriate offset
            if int(v) == 0 and offset == 1:
                raise ValueError("Indexes in LDA-C are offset 1")
        term_cnt_pairs = tuple((int(v) - offset, int(cnt)) for v, cnt in term_cnt_pairs)
        np.testing.assert_equal(unique_terms, len(term_cnt_pairs))
        V = max(V, *[v for v, cnt in term_cnt_pairs])
        data.append(term_cnt_pairs)
        N += 1
    V = V + 1
    dtm = np.zeros((N, V), dtype=np.intc)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_equal(dtm[i, v], 0)
            dtm[i, v] = cnt
    return dtm

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)