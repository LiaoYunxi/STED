# -*- coding: utf-8 -*-
"""
Created on 2023-06-07 (Wed) 10:07:04

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

from logging import getLogger
logger = getLogger('plot_utils')

class DeconvPlot():
    def __init__(self,deconv_df,val_df,dec_name=['B cells naive'],val_name=['Naive B'],
                do_plot=True,figsize=(6,6),dpi=300,plot_size=100):
        self.deconv_df = deconv_df
        self.val_df = val_df
        self.dec_name = dec_name
        self.val_name = val_name
        self.do_plot = do_plot
        self.figsize = figsize
        self.dpi = dpi
        self.plot_size = plot_size

        self.xlabel = 'Estimated Proportion'
        self.ylabel = 'True Proportion'
        self.label_size = 20
        self.tick_size = 15

    def plot_simple_corr(self,color='tab:blue',title='Naive B',target_samples=None):
        """
        Correlation Scatter Plotting
        Format of both input dataframe is as follows
        Note that the targe data contains single treatment group (e.g. APAP treatment only)
        
                    B       CD4       CD8      Monocytes        NK  Neutrophils
        Donor_1 -0.327957 -0.808524 -0.768420   0.311360  0.028878     0.133660
        Donor_2  0.038451 -0.880116 -0.278970  -1.039572  0.865344    -0.437588
        Donor_3 -0.650633  0.574758 -0.498567  -0.796406 -0.100941     0.035709
        Donor_4 -0.479019 -0.005198 -0.675028  -0.787741  0.343481    -0.062349
        
        """
        total_x = self.deconv_df[self.dec_name].sum(axis=1).tolist()
        total_y = self.val_df[self.val_name].sum(axis=1).tolist()
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}
        
        x_min = min(min(total_x),min(total_y))
        x_max = max(max(total_x),max(total_y))
        
        if self.do_plot:
            fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
            if target_samples is None:
                plt.scatter(total_x,total_y,alpha=1.0,s=self.plot_size,c=color)
            else:
                markers1 = ["o", "^", "+", ",", "v",  "<", ">"]
                for mi,d in enumerate(target_samples):
                    tmp1 = self.deconv_df.filter(regex="^"+d+"_",axis=0)
                    tmp2 = self.val_df.filter(regex="^"+d+"_",axis=0)
                    res1 = tmp1[self.dec_name].sum(axis=1).tolist()
                    res2 = tmp2[self.val_name].sum(axis=1).tolist()
                    plt.scatter(res1,res2,alpha=1.0,s=self.plot_size,c=color,marker=markers1[mi])

            #plt.plot([0,x_max],[0,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
            
            #plt.legend(shadow=True)
            plt.xlabel(self.xlabel,fontsize=self.label_size)
            plt.ylabel(self.ylabel,fontsize=self.label_size)
            xlocs, _ = plt.xticks()
            ylocs, _ = plt.yticks()
            tick_min = max(0.0,min(xlocs[0],ylocs[0]))
            tick_max = max(xlocs[-1],ylocs[-1])
            step = (tick_max-tick_min)/5
            plt.xticks(np.arange(tick_min,tick_max,step),fontsize=self.tick_size)
            plt.yticks(np.arange(tick_min,tick_max,step),fontsize=self.tick_size)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            ax.set_axisbelow(True)
            ax.grid(color="#ababab",linewidth=0.5)
            plt.title(title,fontsize=self.label_size)
            plt.show()
        else:
            pass
        return performance,total_x,total_y

    def plot_group_corr(self,sort_index=[],sep=True,title=None):
        """
        Correlation Scatter Plotting
        Format of both input dataframe is as follows
        
                    B       CD4       CD8  Monocytes        NK  Neutrophils
        AN_1 -0.327957 -0.808524 -0.768420   0.311360  0.028878     0.133660
        AN_2  0.038451 -0.880116 -0.278970  -1.039572  0.865344    -0.437588
        AN_3 -0.650633  0.574758 -0.498567  -0.796406 -0.100941     0.035709
        AN_4 -0.479019 -0.005198 -0.675028  -0.787741  0.343481    -0.062349
        AP_1 -1.107050  0.574758  0.858366  -1.503722 -1.053643     1.010999
        
        """
        if title is None:
            title = str(self.dec_name)+" vs "+str(self.val_name)
        
        if len(sort_index)>0:
            drugs = sort_index
        elif sep:
            drugs = sorted(list(set([t.split("_")[0] for t in self.deconv_df.index.tolist()])))
        else:
            drugs = sorted(self.deconv_df.index.tolist())
        
        # align the index
        val_df = self.val_df.loc[self.deconv_df.index.tolist()]
        
        total_x = self.deconv_df[self.dec_name].sum(axis=1).tolist()
        total_y = val_df[self.val_name].sum(axis=1).tolist()
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}

        fig,ax = plt.subplots(figsize=(6,6),dpi=self.dpi)
        x_min = 100
        x_max = -100
        for i,d in enumerate(drugs):
            tmp1 = self.deconv_df.filter(regex="^"+d+"_",axis=0)
            tmp2 = val_df.filter(regex="^"+d+"_",axis=0)
            
            res1 = tmp1[self.dec_name].sum(axis=1).tolist()
            res2 = tmp2[self.val_name].sum(axis=1).tolist()
            tmp_cor = round(np.corrcoef(res1,res2)[0][1],3)
        
            #plt.scatter(res1,res2,label=d+" : "+str(tmp_cor),alpha=1.0,s=self.plot_size) # inner correlation
            plt.scatter(res1,res2,label=d,alpha=1.0,s=self.plot_size)
            xmin = min(min(res1),min(res2))
            if xmin < x_min:
                x_min = xmin
            xmax = max(max(res1),max(res2))
            if xmax > x_max:
                x_max = xmax
        
        plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
        plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
        
        #plt.legend(loc='upper center',shadow=True,fontsize=13,ncol=2,bbox_to_anchor=(.45, 1.12))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1),shadow=True,fontsize=13)
        plt.xlabel(self.xlabel,fontsize=self.label_size)
        plt.ylabel(self.ylabel,fontsize=self.label_size)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title(title,fontsize=15)
        plt.show()

        return performance,total_x,total_y
    
    def overlap_groups(self,evalxy,res_names=[['B cells naive'],['T cells CD4 naive'],['T cells CD8'],['NK cells'],['Monocytes']],
    ref_names=[['Naive B'],['Naive CD4 T'],['CD8 T'],['NK'],['Monocytes']],title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes'],
    color_list=None,target_samples=None,do_plot=True):
        if color_list is None:
            color_list = list(tab_colors.keys())
        # collect information
        total_x = []
        for t in evalxy[0]:
            total_x.extend(t)
        total_y = []
        for t in evalxy[1]:
            total_y.extend(t)
        
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}

        x_min = min(min(total_x),min(total_y))
        x_max = max(max(total_x),max(total_y))
        
        if do_plot:
            fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
            for i in range(len(res_names)):
                res_name = res_names[i]
                ref_name = ref_names[i]

                if target_samples is None:
                    res1 = self.deconv_df[res_name].sum(axis=1).tolist()
                    res2 = self.val_df[ref_name].sum(axis=1).tolist()
                    plt.scatter(res1,res2,alpha=1.0,s=self.plot_size,c=color_list[i],label=title_list[i])
                else:
                    markers1 = ["o", "^", "+", ",", "v",  "<", ">"]
                    for mi,d in enumerate(target_samples):
                        tmp1 = self.deconv_df.filter(regex="^"+d+"_",axis=0)
                        tmp2 = self.val_df.filter(regex="^"+d+"_",axis=0)
                        res1 = tmp1[res_name].sum(axis=1).tolist()
                        res2 = tmp2[ref_name].sum(axis=1).tolist()
                        if mi == 0:
                            plt.scatter(res1,res2,alpha=0.8,s=self.plot_size,c=color_list[i],marker=markers1[mi],label=title_list[i])
                        else:
                            plt.scatter(res1,res2,alpha=0.8,s=self.plot_size,c=color_list[i],marker=markers1[mi])

            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
            
            plt.xlabel(self.xlabel,fontsize=self.label_size)
            plt.ylabel(self.ylabel,fontsize=self.label_size)
            plt.xticks(fontsize=self.tick_size)
            plt.yticks(fontsize=self.tick_size)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            ax.set_axisbelow(True)
            ax.grid(color="#ababab",linewidth=0.5)
            plt.legend(shadow=True,bbox_to_anchor=(1.0, 1), loc='upper left')
            #plt.title(title,fontsize=self.label_size)
            plt.show()

        return total_cor
    
    def overlap_singles(self,evalxy, title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes']):
        total_x = []
        for t in evalxy[0]:
            total_x.extend(t)
        total_y = []
        for t in evalxy[1]:
            total_y.extend(t)
        
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}

        x_min = min(min(total_x),min(total_y))
        x_max = max(max(total_x),max(total_y))

        fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
        for idx in range(len(evalxy[0])):
            res1 = evalxy[0][idx]
            res2 = evalxy[1][idx]
            cell = title_list[idx]

            plt.scatter(res1,res2,alpha=0.8,s=60,label=cell)
            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)

        plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
        #plt.legend(shadow=True)
        plt.xlabel('Estimated Proportion',fontsize=12)
        plt.ylabel('True Proportion',fontsize=12)
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        #plt.title(title,fontsize=12)
        plt.legend(shadow=True,bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

    def estimation_var(total_res,cell='Neutrophil',dpi=100):
        summary_df = pd.DataFrame()
        for idx,tmp_df in enumerate(total_res):
            cell_df = tmp_df[[cell]]
            cell_df.columns = [idx] # rename column
            summary_df = pd.concat([summary_df,cell_df],axis=1)

        sample_names = summary_df.index.tolist()
        data = []
        for sample in sample_names:
            data.append(list(summary_df.loc[sample]))
        
        # plot bar
        plot_multi(data=data,names=sample_names,value='Deconvolution value (%)', title=str(cell)+" estimation variance",grey=False,dpi=dpi)
    
def plot_multi(data=[[11,50,37,202,7],[47,19,195,117,74],[136,69,33,47],[100,12,25,139,89]],names=["+PBS","+Nefopam","+Ketoprofen","+Cefotaxime"],value="ALT (U/I)",title="",grey=True,dpi=100,figsize=(12,6),lw=1,capthick=1,capsize=5):
    sns.set_style('whitegrid')
    if grey:
        sns.set_palette('gist_yarg')
        
    fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
    
    df = pd.DataFrame()
    for i in range(len(data)):
        tmp_df = pd.DataFrame({names[i]:data[i]})
        df = pd.concat([df,tmp_df],axis=1)
    error_bar_set = dict(lw=lw,capthick=capthick,capsize=capsize)
    if grey:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    else:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    # jitter plot
    df_melt = pd.melt(df)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax, size=3)
        
    ax.set_xlabel('')
    ax.set_ylabel(value)
    plt.title(title)
    plt.xticks(rotation=60)
    plt.show()

def plot_radar(data=[[0.3821, 0.6394, 0.8317, 0.7524],[0.4908, 0.7077, 0.8479, 0.7802]],labels=['Neutrophils', 'Monocytes', 'NK', 'Kupffer'],conditions=['w/o addnl. topic','w/ addnl. topic'],title='APAP Treatment',dpi=100):
    # preprocessing
    dft = pd.DataFrame(data,index=conditions)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=dpi)

    # Helper function to plot each car on the radar chart.
    def add_to_radar(name, color):
        values = dft.loc[name].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.25)
    # Add each car to the chart.
    add_to_radar(conditions[0], '#429bf4')
    add_to_radar(conditions[1], '#ec6e95')

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi/4)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles)[0:len(labels)], labels)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('left')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure radar range
    #ax.set_ylim(0, 0.9)
    ax.set_rlabel_position(180 / len(labels))
    ax.tick_params(colors='#222222')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(color='#AAAAAA')
    ax.spines['polar'].set_color('#222222')
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, y=1.02, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

def estimation_var(total_res,cell='Neutrophil',dpi=100):
    summary_df = pd.DataFrame()
    for idx,tmp_df in enumerate(total_res):
        cell_df = tmp_df[[cell]]
        cell_df.columns = [idx] # rename column
        summary_df = pd.concat([summary_df,cell_df],axis=1)

    sample_names = summary_df.index.tolist()
    data = []
    for sample in sample_names:
        data.append(list(summary_df.loc[sample]))
    
    # plot bar
    plot_multi(data=data,names=sample_names,value='Deconvolution value (%)', title=str(cell)+" estimation variance",grey=False,dpi=dpi)

def get_unique_distances(dists: np.array, noise_max=1e-7) -> np.array:
    dists_cp = dists.copy()

    for i in range(dists.shape[0] - 1):
        if dists[i] == dists[i + 1]:
            # returns the next unique distance or the current distance with the added noise
            next_unique_dist = next((d for d in dists[i + 1 :] if d != dists[i]), dists[i] + noise_max)

            # the noise can never be large then the difference between the next unique distance and the current one
            curr_max_noise = min(noise_max, next_unique_dist - dists_cp[i])
            dists_cp[i + 1] = np.random.uniform(low=dists_cp[i] + curr_max_noise / 2, high=dists_cp[i] + curr_max_noise)
    return dists_cp
def matrix_to_doc(doc_word):
    result_list = []
    for row_idx in range(doc_word.shape[0]):
        row = doc_word.getrow(row_idx)
        col_indices = row.indices
        values = row.data 
        row_list = []
        for col_idx, value in zip(col_indices, values):
            row_list.extend([col_idx] * value)

        result_list.append(row_list)
    return result_list
def top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
    indices = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)
    return np.array(indices)
def top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
        top_values.append(scores)
    return np.array(top_values)
def extract_words_per_topic(words,labels,c_tf_idf,top_n_words=30):
    """Based on tf_idf scores per topic, extract the top n words per topic."""
    indices = top_n_idx_sparse(c_tf_idf, top_n_words)
    scores = top_n_values_sparse(c_tf_idf, indices)
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
    topics = {label: values[: top_n_words] for label, values in topics.items()}
    return topics

def validate_distance_matrix(X, n_samples):
    # Make sure it is the 1-D condensed distance matrix with zeros on the diagonal
    s = X.shape
    if len(s) == 1:
        # check it has correct size
        n = s[0]
        if n != (n_samples * (n_samples - 1) / 2):
            raise ValueError("The condensed distance matrix must have " "shape (n*(n-1)/2,).")
    elif len(s) == 2:
        # check it has correct size
        if (s[0] != n_samples) or (s[1] != n_samples):
            raise ValueError("The distance matrix must be of shape " "(n, n) where n is the number of samples.")
        # force zero diagonal and convert to condensed
        np.fill_diagonal(X, 0)
        X = squareform(X)
    else:
        raise ValueError(
            "The distance matrix must be either a 1-D condensed "
            "distance matrix of shape (n*(n-1)/2,) or a "
            "2-D square distance matrix of shape (n, n)."
            "where n is the number of documents."
            "Got a distance matrix of shape %s" % str(s)
        )

    # Make sure its entries are non-negative
    if np.any(X < 0):
        raise ValueError("Distance matrix cannot contain negative values.")
    return X

def save_figure(fig, file_path):
    """通用保存图形的函数"""
    if file_path.endswith('.html'):
        fig.write_html(file_path)
    else:
        fig.write_image(file_path)