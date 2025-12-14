import numpy as np
import pandas as pd
import anndata as ad
import pybedtools
from scipy.sparse import (coo_matrix,csr_matrix)
import matplotlib.pyplot as plt

def _uniquify(seq, sep='-'):
    """Uniquify a list of strings.

    Adding unique numbers to duplicate values.

    Parameters
    ----------
    seq : `list` or `array-like`
        A list of values
    sep : `str`
        Separator

    Returns
    -------
    seq: `list` or `array-like`
        A list of updated values
    """

    dups = {}

    for i, val in enumerate(seq):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            seq[i] += (sep+str(dups[val][1]))

    return(seq)

class GeneScores:
    """A class used to represent gene scores

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self,
                 gene_anno_file,
                 peaks=None,
                 adatatype=True,
                 return_dataframe=True,
                 n_batch=100,
                 tss_upstream=1e5,
                 tss_downsteam=1e5,
                 gb_upstream=5000,
                 cutoff_weight=1,
                 use_top_pcs=True,
                 use_precomputed=False,
                 use_gene_weigt=True,
                 min_w=1,
                 max_w=5):
        """
        Parameters
        ----------
        peaks: `bed`
        """
        self.peaks = peaks
        self.adatatype = adatatype
        self.return_dataframe = return_dataframe
        self.gene_anno_file = gene_anno_file
        self.n_batch = n_batch
        self.tss_upstream = tss_upstream
        self.tss_downsteam = tss_downsteam
        self.gb_upstream = gb_upstream
        self.cutoff_weight = cutoff_weight
        self.use_top_pcs = use_top_pcs
        self.use_precomputed = use_precomputed
        self.use_gene_weigt = use_gene_weigt
        self.min_w = min_w
        self.max_w = max_w

    def _read_gene_anno(self):
        """Read in gene annotation

        Parameters
        ----------

        Returns
        -------

        """

        gene_anno = pd.read_csv(self.gene_anno_file,
                                encoding='utf8',
                                sep='\t',
                                header=None,
                                names=['chr', 'start', 'end',
                                       'symbol', 'strand'])
        self.gene_anno = gene_anno
        return self.gene_anno

    def _extend_tss(self, pbt_gene):
        """Extend transcription start site in both directions

        Parameters
        ----------

        Returns
        -------

        """
        ext_tss = pbt_gene
        if (ext_tss['strand'] == '+'):
            ext_tss.start = max(0, ext_tss.start - self.tss_upstream)
            ext_tss.end = max(ext_tss.end, ext_tss.start + self.tss_downsteam)
        else:
            ext_tss.start = max(0, min(ext_tss.start,
                                       ext_tss.end - self.tss_downsteam))
            ext_tss.end = ext_tss.end + self.tss_upstream
        return ext_tss

    def _extend_genebody(self, pbt_gene):
        """Extend gene body upstream

        Parameters
        ----------

        Returns
        -------

        """
        ext_gb = pbt_gene
        if (ext_gb['strand'] == '+'):
            ext_gb.start = max(0, ext_gb.start - self.gb_upstream)
        else:
            ext_gb.end = ext_gb.end + self.gb_upstream
        return ext_gb

    def _weight_genes(self):
        """Weight genes

        Parameters
        ----------

        Returns
        -------

        """
        gene_anno = self._read_gene_anno()
        gene_size = gene_anno['end'] - gene_anno['start']
        w = 1 / gene_size
        w_scaled = (self.max_w - self.min_w) * (w - min(w)) / (max(w) - min(w)) + self.min_w
        return w_scaled

    def cal_gene_scores(self):
        """Calculate gene scores

        Parameters
        ----------

        Returns
        -------

        """
        gene_ann = self._read_gene_anno()
        if self.adatatype == False:
            fragments = self.peaks

            if ('gene_scores' not in fragments.columns):
                print('Gene scores are being calculated for the first time')
                print('`use_precomputed` has been ignored')
                self.use_precomputed = False

            sample_IDs = list(set(fragments['sample']))

            gene_scores_dict = dict()

            df_gene_ann = gene_ann.copy()
            df_gene_ann.index = _uniquify(df_gene_ann['symbol'].values)
            if self.use_top_pcs:
                pass
            else:
                mask_p = pd.Series(True, index=fragments.index)
            df_peaks = fragments[mask_p][['chr', 'start', 'end', 'sample','score']].copy()
        else:
            adata = self.peaks

            sample_IDs = adata.obs_names.to_list()

            df_gene_ann = gene_ann.copy()
            df_gene_ann.index = _uniquify(df_gene_ann['symbol'].values)

            if self.use_top_pcs:
                mask_p = adata.var['top_pcs']
            else:
                mask_p = pd.Series(True, index=adata.var_names)
            df_peaks = adata.var[mask_p][['chr', 'start', 'end']].copy()
            df_peaks['score'] = adata.X.toarray().squeeze().tolist()

        if (self.use_precomputed):
            print('Using precomputed overlap')
            if self.adatatype == True:
                df_overlap_updated = df_peaks
            else:
                df_overlap_updated = fragments['gene_scores']
        else:
            # add the fifth column
            # so that pybedtool can recognize the sixth column as the strand
            df_gene_ann_for_pbt = df_gene_ann.copy()
            df_gene_ann_for_pbt['score'] = 0
            df_gene_ann_for_pbt = df_gene_ann_for_pbt[['chr', 'start', 'end','symbol', 'score','strand']]
            df_gene_ann_for_pbt['id'] = range(df_gene_ann_for_pbt.shape[0])

            for sample_ID in sample_IDs:
                if self.adatatype==True:
                    df_peaks_for_pbt = df_peaks.copy()
                else:
                    df_peaks_for_pbt = df_peaks[df_peaks['sample'] == sample_ID].copy()

                # print(np.array(peaks.loc[mask_p, ['score']]).flatten())
                df_peaks_for_pbt['id'] = range(df_peaks_for_pbt.shape[0])

                pbt_gene_ann = pybedtools.BedTool.from_dataframe(df_gene_ann_for_pbt)
                # pbt_gene_ann_ext = pbt_gene_ann.each(self._extend_tss)
                # pbt_gene_gb_ext = pbt_gene_ann.each(self._extend_genebody)

                pbt_peaks = pybedtools.BedTool.from_dataframe(df_peaks_for_pbt)

                # peaks overlapping with extended TSS
                pbt_overlap = pbt_peaks.intersect(pbt_gene_ann.each(self._extend_tss),
                                                  wa=True,
                                                  wb=True)
                df_overlap = pbt_overlap.to_dataframe(
                    names=[x + '_p' for x in df_peaks_for_pbt.columns]
                          + [x + '_g' for x in df_gene_ann_for_pbt.columns])
                # peaks overlapping with gene body
                pbt_overlap2 = pbt_peaks.intersect(pbt_gene_ann.each(self._extend_genebody),
                                                   wa=True,
                                                   wb=True)
                df_overlap2 = pbt_overlap2.to_dataframe(
                    names=[x + '_p' for x in df_peaks_for_pbt.columns]
                          + [x + '_g' for x in df_gene_ann_for_pbt.columns])

                # add distance and weight for each overlap
                df_overlap_updated = df_overlap.copy()
                df_overlap_updated['dist'] = 0

                for i, x in enumerate(df_overlap['symbol_g'].unique()):
                    # peaks within the extended TSS
                    df_overlap_x = df_overlap[df_overlap['symbol_g'] == x].copy()
                    # peaks within the gene body
                    df_overlap2_x = df_overlap2[df_overlap2['symbol_g'] == x].copy()
                    # peaks that are not intersecting with the promoter
                    # and gene body of gene x
                    id_overlap = df_overlap_x.index[~np.isin(df_overlap_x['id_p'], df_overlap2_x['id_p'])]
                    mask_x = (df_gene_ann['symbol'] == x)
                    range_x = df_gene_ann[mask_x][['start', 'end']].values.flatten()

                    if (df_overlap_x['strand_g'].iloc[0] == '+'):
                        df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                            [abs(df_overlap_x.loc[id_overlap, 'start_p']
                                 - (range_x[1])),
                             abs(df_overlap_x.loc[id_overlap, 'end_p']
                                 - max(0, range_x[0] - self.gb_upstream))],
                            axis=1, sort=False).min(axis=1)
                    else:
                        df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                            [abs(df_overlap_x.loc[id_overlap, 'start_p']
                                 - (range_x[1] + self.gb_upstream)),
                             abs(df_overlap_x.loc[id_overlap, 'end_p']
                                 - (range_x[0]))],
                            axis=1, sort=False).min(axis=1)

                df_overlap_updated['dist'] = df_overlap_updated['dist'].astype(float)

                # adata.uns['gene_scores'] = dict()
                # adata.uns['gene_scores']['overlap'] = df_overlap_updated.copy()

                df_overlap_updated['weight'] = np.exp(-(df_overlap_updated['dist'].values / self.gb_upstream))
                mask_w = (df_overlap_updated['weight'] < self.cutoff_weight)
                df_overlap_updated.loc[mask_w, 'weight'] = 0
                # df_overlap_updated['weight'].astype(int)
                # construct genes-by-peaks matrix
                mat_GP = csr_matrix(coo_matrix((df_overlap_updated['weight'],
                                                (df_overlap_updated['id_g'],
                                                 df_overlap_updated['id_p'])),
                                               shape=(df_gene_ann.shape[0],
                                                      df_peaks.shape[0])))
                if self.use_gene_weigt:
                    gene_weights = self._weight_genes()
                    gene_scores = np.array(df_peaks.loc[mask_p, ['score']]).flatten() * (mat_GP.T.multiply(gene_weights))
                else:
                    gene_scores = np.array(df_peaks.loc[mask_p, ['score']]).flatten() * mat_GP.T

                if self.adatatype !=True:
                    gene_scores_dict[sample_ID] = gene_scores
                    if (len(gene_scores_dict.keys()) % self.n_batch == 0):
                        print(f'Processing: {len(gene_scores_dict.keys()) / len(sample_IDs):.1%}')

        if self.adatatype:
            gene_scores_df = pd.DataFrame(gene_scores)
            adata_CG_atac = ad.AnnData(np.array(gene_scores_df).T,
                                       var=df_gene_ann.copy())
        else:
            gene_scores_df = pd.DataFrame.from_dict(gene_scores_dict)
            adata_CG_atac = ad.AnnData(np.array(gene_scores_df).T,
                                       obs=sample_IDs,
                                       var=df_gene_ann.copy())
        if self.return_dataframe:
            df = pd.DataFrame(adata_CG_atac.X.astype(int).squeeze().tolist())
            df.index = adata_CG_atac.var_names
            df.columns = ['Bulk']
            return df
        else:
            return adata_CG_atac

class Gene2Peaks:
    def __init__(self,
                 gene_anno_file,
                 peaks,
                 tss_upstream=1e5,
                 tss_downsteam=1e5,
                 gb_upstream=5000,
                 cutoff_weight=1,
                 min_w=1,
                 max_w=5):
        """
        peaks: `bed`
        genome : `str` The genome name
        """
        self.gene_anno_file = gene_anno_file
        self.peaks = peaks
        self.tss_upstream = tss_upstream
        self.tss_downsteam = tss_downsteam
        self.gb_upstream = gb_upstream
        self.cutoff_weight = cutoff_weight
        self.min_w = min_w
        self.max_w = max_w

    def _read_gene_anno(self):
        """Read in gene annotation
        """
        gene_anno = pd.read_csv(self.gene_anno_file,
                                encoding='utf8',
                                sep='\t',
                                header=None,
                                names=['chr', 'start', 'end',
                                       'symbol', 'strand'])
        self.gene_anno = gene_anno
        return self.gene_anno

    def _extend_tss(self, pbt_gene):
        """Extend transcription start site in both directions
        """
        ext_tss = pbt_gene
        if (ext_tss['strand'] == '+'):
            ext_tss.start = max(0, ext_tss.start - self.tss_upstream)
            ext_tss.end = max(ext_tss.end, ext_tss.start + self.tss_downsteam)
        else:
            ext_tss.start = max(0, min(ext_tss.start,
                                       ext_tss.end - self.tss_downsteam))
            ext_tss.end = ext_tss.end + self.tss_upstream
        return ext_tss

    def _extend_genebody(self, pbt_gene):
        """Extend gene body upstream
        """
        ext_gb = pbt_gene
        if (ext_gb['strand'] == '+'):
            ext_gb.start = max(0, ext_gb.start - self.gb_upstream)
        else:
            ext_gb.end = ext_gb.end + self.gb_upstream
        return ext_gb

    def _weight_genes(self):
        """Weight genes
        """
        gene_anno = self._read_gene_anno()
        gene_size = gene_anno['end'] - gene_anno['start']
        w = 1 / gene_size
        w_scaled = (self.max_w - self.min_w) * (w - min(w)) / (max(w) - min(w)) + self.min_w
        return w_scaled

    def gwt_gene_peak_match(self):
        gene_ann = self._read_gene_anno()
        df_gene_ann = gene_ann.copy()
        df_gene_ann.index = _uniquify(df_gene_ann['symbol'].values)
        df_gene_ann_for_pbt = df_gene_ann.copy()
        df_gene_ann_for_pbt = df_gene_ann_for_pbt[['chr', 'start', 'end', 'symbol', 'strand']]
        df_gene_ann_for_pbt['id'] = range(df_gene_ann_for_pbt.shape[0])

        df_peaks = self.peaks[['chr', 'start', 'end']].copy()
        df_peaks_for_pbt = df_peaks.copy()
        df_peaks_for_pbt['id'] = range(df_peaks_for_pbt.shape[0])

        pbt_gene_ann = pybedtools.BedTool.from_dataframe(df_gene_ann_for_pbt)
        pbt_gene_ann_ext = pbt_gene_ann.each(self._extend_tss)
        pbt_gene_gb_ext = pbt_gene_ann.each(self._extend_genebody)

        pbt_peaks = pybedtools.BedTool.from_dataframe(df_peaks_for_pbt)

        # peaks overlapping with extended TSS
        pbt_overlap = pbt_peaks.intersect(pbt_gene_ann_ext, wa=True, wb=True)
        df_overlap = pbt_overlap.to_dataframe(
            names=[x + '_p' for x in df_peaks_for_pbt.columns]
                  + [x + '_g' for x in df_gene_ann_for_pbt.columns])

        # peaks overlapping with gene body
        pbt_overlap2 = pbt_peaks.intersect(pbt_gene_gb_ext,
                                           wa=True,
                                           wb=True)
        df_overlap2 = pbt_overlap2.to_dataframe(
            names=[x + '_p' for x in df_peaks_for_pbt.columns]
                  + [x + '_g' for x in df_gene_ann_for_pbt.columns])

        # add distance and weight for each overlap
        df_overlap_updated = df_overlap.copy()
        df_overlap_updated['dist'] = 0

        for i, x in enumerate(df_overlap['symbol_g'].unique()):
            # peaks within the extended TSS
            df_overlap_x = df_overlap[df_overlap['symbol_g'] == x].copy()
            # peaks within the gene body
            df_overlap2_x = df_overlap2[df_overlap2['symbol_g'] == x].copy()
            # peaks that are not intersecting with the promoter
            # and gene body of gene x
            id_overlap = df_overlap_x.index[~np.isin(df_overlap_x['id_p'], df_overlap2_x['id_p'])]
            mask_x = (df_gene_ann['symbol'] == x)
            range_x = df_gene_ann[mask_x][['start', 'end']].values.flatten()

            if (df_overlap_x['strand_g'].iloc[0] == '+'):
                df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                    [abs(df_overlap_x.loc[id_overlap, 'start_p']
                         - (range_x[1])),
                     abs(df_overlap_x.loc[id_overlap, 'end_p']
                         - max(0, range_x[0] - self.gb_upstream))],
                    axis=1, sort=False).min(axis=1)
            else:
                df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                    [abs(df_overlap_x.loc[id_overlap, 'start_p']
                         - (range_x[1] + self.gb_upstream)),
                     abs(df_overlap_x.loc[id_overlap, 'end_p']
                         - (range_x[0]))],
                    axis=1, sort=False).min(axis=1)

        df_overlap_updated['dist'] = df_overlap_updated['dist'].astype(float)
        df_overlap_updated['weight'] = np.exp(-(df_overlap_updated['dist'].values / self.gb_upstream))

        mask_w = (df_overlap_updated['weight'] < self.cutoff_weight)
        df_overlap_updated.loc[mask_w, 'weight'] = 0

        df = df_overlap_updated.copy()
        # construct genes-by-peaks matrix
        mat_GP = csr_matrix(coo_matrix((df_overlap_updated['weight'],
                                        (df_overlap_updated['id_g'],
                                         df_overlap_updated['id_p'])),
                                       shape=(df_gene_ann.shape[0], df_peaks.shape[0])))

        return df, mat_GP
