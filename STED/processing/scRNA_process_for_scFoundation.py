import sys
import pandas as pd
import numpy as np
import os
import argparse
import scanpy as sc
# add scFoundation path, your path may be different
sys.path.append("/data/lyx/software/scFoundation/model")
sys.path.append("/data/lyx/software/scFoundation")
from STED.Preprocessing import *
from preprocessing.scRNA_workflow import *

####################################Settings#################################
parser = argparse.ArgumentParser(description='Pre_for_scFoundation')
parser.add_argument('--data_dir', type=str, default='/data/lyx/scCHiP/scATAC/LDA/PBMC_10k/processed_data', help='data_dir')
parser.add_argument('--count_file', type=str, default='10k_PBMC_Multiome_filtered_gene_count.h5ad', help='scRNA-seq counts')
parser.add_argument('--anno_file', type=str, default='MainCelltype.txt', help='cell annotation')
parser.add_argument('--out_dir', type=str, default='PBMC_Main', help='out_dir')
parser.add_argument('--save_path', type=str, default='PBMC_Main_demo.h5ad', help='save path')

args = parser.parse_args()

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

def main():
    data_dir = args.data_dir
    sc_count_file = os.path.join(data_dir,args.count_file)
    sc_anno_file = os.path.join(data_dir,args.anno_file)
    sc_batch_file = None

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    print("Reading single-cell count matrix...")
    scp = scPreProcessing()
    scp.set_data(sc_count_file=sc_count_file,sc_anno_file=sc_anno_file,batch_info=sc_batch_file)
    scp.cell_selection()

    gene_list_df = pd.read_csv('/data/lyx/software/scFoundation/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    # TODO:20260128
    # X_df = pd.DataFrame(scp.mix_raw.todense(),index=scp.sc_cells,columns = scp.sc_genes)
    X_df = pd.DataFrame(scp.mix_raw.todense().T, index=scp.sc_cells, columns=scp.sc_genes)
    X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)

    adata = sc.AnnData(X_df,obs = pd.DataFrame(index = X_df.index),var =pd.DataFrame(index=X_df.columns) )

    adata_uni = BasicFilter(adata,qc_min_genes=200,qc_min_cells=0) # filter cell and gene by lower limit
    adata_uni = QC_Metrics_info(adata)

    save_path = os.path.join(out_dir,args.save_path)
    save_adata_h5ad(adata,save_path)

if __name__=='__main__':
    main()