from .Genescore import Gene2Peaks
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def Predict_Peak(gene_anno_file, sc_anno_file,
                 sc_count_file, peak_count_file,
                 predict, celltype_col,
                 scATAC_file=None,
                 estimate=False, pseudo=False):
    if peak_count_file.endswith(".h5ad"):
        adata_CP = ad.read_h5ad(peak_count_file)
    else:
        print('you better transform the data to h5ad format')

    if sc_count_file.endswith(".h5ad"):
        adata = ad.read_h5ad(sc_count_file)
    else:
        print('you better transform the data to h5ad format')

    cell_data = pd.read_table(sc_anno_file, header=None,index_col=0)
    cell_data.columns = [celltype_col]
    cell_type = list(set(cell_data[celltype_col]))

    mat = adata.X
    row_name = adata.obs_names.to_list()
    col_name = adata.var_names.to_list()
    dic_exp = {}
    for type_ in cell_type:
        names = cell_data.index[cell_data[celltype_col] == type_].to_list()
        tmp_mat = mat[[row_name.index(i) for i in names], :]
        tmp_exp = tmp_mat.sum(axis=0).flatten().tolist()[-1]
        # print(len(tmp_exp))
        dic_exp[type_] = tmp_exp
    celltype_exp = pd.DataFrame.from_dict(dic_exp, orient='index')
    celltype_exp.columns = col_name

    gs_scale_factor = celltype_exp.sum(axis=1) / 10000
    dic_exp = {}
    for idx, row in celltype_exp.iterrows():
        dic_exp[idx] = row / gs_scale_factor[idx]
    celltype_exp_libnorm = pd.DataFrame.from_dict(dic_exp, orient='index')
    celltype_exp_libnorm.columns = col_name

    peak = adata_CP.X.toarray()
    gene_anno = pd.read_csv(gene_anno_file,
                            encoding='utf8', sep='\t', header=None,
                            names=['chr', 'start', 'end', 'symbol', 'strand'])

    G2P = Gene2Peaks(gene_anno_file=gene_anno_file, peaks=adata_CP.var, cutoff_weight=0)
    df, mat_GP = G2P.gwt_gene_peak_match()
    adata_GS = ad.AnnData(mat_GP)
    row_names = adata_CP.var_names
    adata_GS.obs_names = pd.Index(gene_anno.symbol)
    adata_GS.var_names = adata_CP.var_names
    celltype_exp_libnorm_filtered = celltype_exp_libnorm.loc[:, celltype_exp_libnorm.columns.isin(adata_GS.obs_names)]
    ex = np.array(celltype_exp_libnorm_filtered)
    idx = [adata_GS.obs_names.to_list().index(i) for i in celltype_exp_libnorm_filtered.columns.to_list()]
    adata_GS = adata_GS[idx, :]
    gs = adata_GS.X.toarray()

    # get gene score·scRNA-seq lib-normalized count
    Ep2 = np.dot(ex, gs)
    Ep2_df = pd.DataFrame(Ep2)
    Ep2_df.columns = adata_CP.var_names
    Ep2_df.index = celltype_exp_libnorm_filtered.index

    gs_scale_factor = Ep2_df.sum(axis=0) / len(cell_type)
    dic_exp = {}
    for idx, row in Ep2_df.transpose().iterrows():
        if gs_scale_factor[idx] != 0:
            dic_exp[idx] = row / gs_scale_factor[idx]
        else:
            dic_exp[idx] = row
    # gene score·scRNA-seq lib-normalized count to celltype_peak_weight
    celltype_peak_weight = pd.DataFrame.from_dict(dic_exp, orient='index')
    per = np.array(predict.Bulk[celltype_exp_libnorm_filtered.index].to_list())
    per = np.expand_dims(per, axis=1)
    Ep1 = np.dot(per, peak)
    Ep1_df = pd.DataFrame(Ep1)
    Ep1_df.columns = adata_CP.var_names
    Ep1_df.index = celltype_exp_libnorm_filtered.index

    Ep = np.multiply(np.array(celltype_peak_weight.transpose()), Ep1_df)
    Ep_df = pd.DataFrame(Ep)
    Ep_df.columns = adata_CP.var_names
    Ep_df.index = celltype_exp_libnorm_filtered.index

    if pseudo:
        if scATAC_file==None:
            print('please give the scATAC-seq count h5ad data')
        else:
            adata_sc_CP = ad.read_h5ad(scATAC_file)
            adata_sc_CP = adata_sc_CP[cell_data.index.to_list()]
            adata_sc_CP.obs.loc[:, celltype_col] = cell_data[celltype_col].to_list()
            adata_sc_CP.var_names = [i.replace(':', '_').replace('-', '_') for i in adata_sc_CP.var_names]
            adata_sc_CP = adata_sc_CP[:, adata_CP.var_names]
            row_name = adata_sc_CP.obs_names.to_list()
            col_name = adata_sc_CP.var_names.to_list()
            dic_exp = {}
            for type_ in cell_type:
                names = adata_sc_CP.obs_names[adata_sc_CP.obs[celltype_col] == type_].to_list()
                tmp_mat = adata_sc_CP.X[[row_name.index(i) for i in names], :]
                tmp_exp = tmp_mat.mean(axis=0).flatten().tolist()[-1]  ### sum ###
                dic_exp[type_] = tmp_exp

            celltype_peak_exp = pd.DataFrame.from_dict(dic_exp, orient='index')
            celltype_peak_exp.columns = col_name
            celltype_peak_exp.index = cell_type

            gs_scale_factor = np.round(celltype_peak_exp.sum(axis=0) / 10000)
            dic_exp = {}
            for idx, row in celltype_peak_exp.transpose().iterrows():
                if gs_scale_factor[idx] != 0:
                    dic_exp[idx] = row / gs_scale_factor[idx]
                else:
                    dic_exp[idx] = row
            celltype_peak_exp_scaled = pd.DataFrame.from_dict(dic_exp, orient='index')

            truth = cell_data[celltype_col].value_counts().to_frame()
            truth['percent'] = truth[celltype_col] / sum(truth[celltype_col])
            predict['truth'] = truth['percent'][predict.index]

            value_cor_dic = {}
            Ep_df_filter = Ep_df.loc[:, Ep_df.sum(0) > 0]
            # y_predict = np.array(Ep_df_filter.sum(0).to_list())
            celltype_peak_exp_filter = celltype_peak_exp.loc[:, Ep_df.sum(0) > 0]
            for idx, value in celltype_peak_exp_filter.iterrows():
                value_cor_dic[idx] = Ep_df_filter.loc[idx].corr(value)
            final_p = pd.DataFrame.from_dict(value_cor_dic, orient='index').mean(0)[0]
            # print('modol:', model_selected, 'ntopics:', ntopics_selected, "person:", final_p)

            # y_test = np.array(celltype_peak_exp_filter.sum(0).to_list())
            # mae = mean_absolute_error(y_test, y_predict)
            # mse = mean_squared_error(y_test, y_predict)
            # rmse = sqrt(mean_squared_error(y_test, y_predict))
            # r2 = r2_score(y_test, y_predict)
            return {'PCC': final_p}, Ep_df
    else:
        if estimate:
            Ep_df_filter = Ep_df.loc[:, Ep_df.sum(0) > 0]
            y_predict = np.array(Ep_df_filter.sum(0).to_list())
            y_bulk_test = adata_CP[:, Ep_df.sum(0) > 0].X.toarray().flatten()
            mae_b = mean_absolute_error(y_bulk_test, y_predict)
            mse_b = mean_squared_error(y_bulk_test, y_predict)
            rmse_b = sqrt(mean_squared_error(y_bulk_test, y_predict))
            r2_b = r2_score(y_bulk_test, y_predict)
            return {'MAE':mae_b,'MSE':mse_b,'RMSE':rmse_b,'R2':r2_b},Ep_df
        else:
            return {},Ep_df



