
import random
from tqdm.auto import tqdm
import copy

import numpy as np
import pandas as pd

from collections import defaultdict,Counter
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import gensim
from gensim import matutils
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from scipy.sparse import csr_matrix

import matplotlib
#matplotlib.use('TkAgg')  # 'Qt5Agg', 'GTK3Agg'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

from .Normalize import *

def RunBayesEvaluate(topic_cell_mat, topic_celltype_df,ann_list, celltype_num_dict, model_dir):
    bayes_res = ModelEvaluateBayes(topic_cell_mat, topic_celltype_df, ann_list, celltype_num_dict, model_dir)
    celltype_topic_bayes_df = bayes_res["celltype_topic_bayes_df"]
    bayesnorm_res = ModelEvaluateBayesNorm(topic_cell_mat, celltype_topic_bayes_df, ann_list)
    if bayesnorm_res["accuracy"] >= bayes_res["accuracy"]:
        v = bayesnorm_res["accuracy"]
        model_norm = "BayesNorm"
        celltype_prediction = bayesnorm_res["celltype_prediction"]
    else:
        v = bayes_res["accuracy"]
        model_norm = "Bayes"
        celltype_prediction = bayes_res["celltype_prediction"]
    return bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction

def RunCoherenceEvaluate(indicator_selection,lda,sc_corpus,genes,topic_cell_mat, topic_celltype_df,ann_list):
    if indicator_selection == "nmi":
                            v_dict = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, ann_list)
                            v = v_dict[["nmi"]]
    elif indicator_selection == "umass_coherence":
        cm = CoherenceModel(model = lda, corpus = sc_corpus, coherence='u_mass')
        v = cm.get_coherence()
    elif indicator_selection == "cv_coherence":
        cm = CoherenceModel(model = lda, corpus = sc_corpus, texts = genes, coherence='c_v')
        v = cm.get_coherence()
    return v

def matrix_to_lists(doc_word):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices."""
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

    DS = np.repeat(ii, ss).astype(np.int32)  # 确保为整数类型
    WS = np.repeat(jj, ss).astype(np.int32)

    return WS, DS

def GLDA(i,gene2id,seed_topics,ntopics,genes,cells,cell_celltype_list,sc_corpus,n_iter,alpha,eta):
        random.seed(i+1)
        re_order = random.sample(range(len(genes)), k=len(genes))# randomly sort the gene order
        genes = [genes[i] for i in re_order]
        genes_dict = Dictionary([genes]) 
        tmp = sc_corpus[:,re_order]

        sc_corpus = gensim.matutils.Sparse2Corpus(tmp.transpose())
        fake_docs = [[word] for word in gene2id.keys()]
        genes_dict = Dictionary(fake_docs)

        guided_lda = GuidedLdaModel(
            corpus=sc_corpus,
            num_topics=ntopics,
            alpha=alpha,
            eta=eta,
            iterations=n_iter,
            random_state=i,
            seed_topics=seed_topics,
            id2word=genes_dict,
        )

        # Inject precomputed counts
        guided_lda.inject_counts(csr_matrix(tmp, dtype=np.int64), initial_confidence=1.0)

        # 替换 inference 方法
        guided_lda.inference = lambda chunk: inference_with_seeds(
            guided_lda, chunk, seed_topics, seed_conf=0.9
        )

        # Update the model
        guided_lda.update(sc_corpus)

        topic_cell = guided_lda.get_document_topics(sc_corpus)
        topic_cell_mat = gensim.matutils.corpus2csc(topic_cell)

        topic_gene_mat_list = guided_lda.get_topics()
        topic_gene_mat = np.array(topic_gene_mat_list)
        gene_topic_mat = topic_gene_mat.transpose()
        gene_topic_mat_list = gene_topic_mat.tolist()

        # convert topic_cell_mat to topic_celltype_mat
        celltype_topic_dict = {}
        celltype_num_dict = {}
        celltypes = sorted(list(set(cell_celltype_list)))
        for celltype in celltypes:
            celltype_topic_dict[celltype] = [0]*ntopics
            celltype_num_dict[celltype] = 0
        for i in range(topic_cell_mat.shape[1]):
            cell_celltype = cell_celltype_list[i]
            celltype_topic_dict[cell_celltype] = [celltype_topic_dict[cell_celltype][j] + topic_cell_mat[j,i] for j in range(topic_cell_mat.shape[0])]
            celltype_num_dict[cell_celltype] = celltype_num_dict[cell_celltype] + 1
        celltype_topic_mean_dict = {}

        for celltype in celltypes:
            celltype_topic_mean_dict[celltype] = [i/celltype_num_dict[celltype] for i in celltype_topic_dict[celltype]]
        topic_celltype_df = pd.DataFrame(data = celltype_topic_mean_dict)
        return topic_cell_mat,topic_celltype_df,celltype_num_dict,gene_topic_mat_list,genes,sc_corpus,genes_dict,guided_lda

class GuidedLdaModel(LdaModel):
    def __init__(self, *args, seed_topics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_topics = seed_topics or {}

    def inject_counts(self, X, initial_confidence):
        """
        Initialize inject_counts with seed word assignments.

        Parameters:
        ----------
        X : array-like, shape (D, V)
            Document-term matrix.

        initial_confidence : float
            Probability of assigning a seed word to its predefined topic.
        """
        seed_topics = self.seed_topics  # {word_id: [topic_id1, topic_id2, ...]}

        # Initialize matrices
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.num_topics
        nzw_ = np.zeros((n_topics, W), dtype=np.float64)  # Topic-word counts
        ndz_ = np.zeros((D, n_topics), dtype=np.float64)  # Document-topic counts
        nz_ = np.zeros(n_topics, dtype=np.float64)        # Topic counts

        # Convert matrix to lists
        WS, DS = matrix_to_lists(X)
        ZS = np.empty_like(WS, dtype=np.float64)

        np.testing.assert_equal(N, len(WS))

        # Single Initialization Loop
        for i in range(N):
            w, d = WS[i], DS[i]

            # Check if word is a seed word
            if w in seed_topics:
                if random.random() < initial_confidence:
                    z_new = random.choice(seed_topics[w])  # Assign to predefined topic
                else:
                    z_new = i % n_topics  # Fallback to random assignment
            else:
                z_new = i % n_topics  # Random assignment for non-seed words

            # Update counts
            ZS[i] = z_new
            ndz_[int(d), int(z_new)] += 1
            nzw_[int(z_new), int(w)] += 1
            nz_[int(z_new)] += 1

        # Update Gensim's internal state
        self.state.sstats[:, :] = np.array(nzw_, dtype=np.float64) 
        self.state.sstats /= np.sum(self.state.sstats, axis=1)[:, np.newaxis]  # Normalize
        # self.state.sstats = self.state.sstats.astype(np.float64)

def inference_with_seeds(self, chunk, seed_topics, seed_conf=0.9):
    """
    Custom inference method with seed word constraints.

    Parameters:
    ----------
    chunk : iterable of lists of (int, float)
        Stream of document in BOW format.
    seed_topics : dict
        Mapping of word IDs to a list of possible topic IDs (e.g., {word_id: [topic_id1, topic_id2]}).
    seed_conf : float
        Confidence level for keeping seed words in their predefined topics (0 < seed_conf <= 1).
    """
    gamma, sstats = self.init_gammastats(len(chunk))

    for d, doc in enumerate(chunk):
        # Document-specific parameters
        gammad = gamma[d]
        expElogthetad = np.exp(self.dirichlet_expectation(gammad))
        expElogbetad = self.expElogbeta[:, [id for id, _ in doc]]

        for i in range(self.iterations):
            lastgamma = gammad.copy()
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-10

            for id, count in doc:
                # Check if the word is a seed word
                if id in seed_topics:
                    topic_ids = seed_topics[id]  # Allowed topics for the seed word
                    phi = expElogthetad * expElogbetad[:, id]
                    phi = phi / phinorm

                    # Apply constraints: boost probabilities for seed topics
                    boost = np.zeros_like(phi)
                    boost[topic_ids] = 1.0 / len(topic_ids)
                    phi = (1 - seed_conf) * phi + seed_conf * boost

                    phinorm[id] = np.sum(phi)
                    expElogbetad[:, id] = phi / phinorm[id]
                else:
                    # Regular word sampling
                    expElogbetad[:, id] = expElogthetad * expElogbetad[:, id] / phinorm[id]

            gammad = self.alpha + np.dot(count / phinorm, expElogbetad.T)

            # Check convergence
            if np.mean(abs(gammad - lastgamma)) < self.gamma_threshold:
                break

        # Update sufficient statistics
        sstats[:, [id for id, _ in doc]] += np.outer(expElogthetad, count / phinorm)

    return gamma, sstats

class AnchoredLDA():
    def __init__(self):
        self.sc_corpus = None
        self.genes = None
        self.cells = None
        self.celltypes = None
        self.write = False
    def setData(self,sc_corpus, genes, sc_count_cells, ann_list, alpha,eta,out_dir):
        self.sc_corpus = sc_corpus.tocsc()
        self.genes = copy.deepcopy(genes)
        self.cells = sc_count_cells
        self.ann_list = ann_list
        self.alpha = alpha
        self.eta = eta
        self.out_dir = out_dir
    def benchmark(self,gene2id,seed_topics,model_dir,ntopics_list,n_ensemble=10,n_iter=100):
        print("test model:will save all results!")
        accuracy_bayes_dict = defaultdict(list)
        accuracy_bayesnorm_dict = defaultdict(list)
        accuracy_lg_dict = defaultdict(list)
        umass_coherence_dict = defaultdict(list)
        cv_coherence_dict = defaultdict(list)
        nmi_dict = defaultdict(list)

        for ntopics in ntopics_list:
            print("Number of topics: %s" %(ntopics))
            if self.alpha is None:
                alpha = np.float64(1 / ntopics)
            if self.eta is None:
                eta = np.float64(1 / ntopics)
            for i in tqdm(range(n_ensemble)):
                topic_cell_mat,topic_celltype_df,celltype_num_dict,\
                gene_topic_mat_list,genes,sc_corpus,genes_dict,lda = GLDA(i,gene2id,seed_topics,ntopics,self.genes,self.cells,self.ann_list,self.sc_corpus,n_iter,alpha,eta)

                raw_res = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, self.ann_list)
                nmi_dict[ntopics].append(raw_res["nmi"])

                cm = CoherenceModel(model = lda, corpus = sc_corpus, coherence='u_mass')
                umass_coherence_dict[ntopics].append(cm.get_coherence())
                cm = CoherenceModel(model = lda, corpus = sc_corpus, texts = genes, coherence='c_v')
                cv_coherence_dict[ntopics].append(cm.get_coherence())

                bayes_res,bayesnorm_res,celltype_topic_bayes_df,model_norm,v,celltype_prediction = \
                    RunBayesEvaluate(topic_cell_mat, topic_celltype_df, self.ann_list, celltype_num_dict, model_dir)
                accuracy_bayes_dict[ntopics].append(bayes_res["accuracy"])
                accuracy_bayesnorm_dict[ntopics].append(bayesnorm_res["accuracy"])

                y=self.ann_list
                cell_topic_mat = topic_cell_mat.T

                model_l = LogisticRegression(random_state=i+1)
                model_l.fit(cell_topic_mat,y=y)
                y_p = model_l.predict(cell_topic_mat)
                accuracy = accuracy_score(y, y_p)
                accuracy_lg_dict[ntopics].append(accuracy)

        metrics_dict = {"Topic": ntopics_list,
                        "lg_accuracy":accuracy_lg_dict,
                                        "Bayes_accuracy": accuracy_bayes_dict,
                                        "BayesNorm_accuracy": accuracy_bayesnorm_dict,
                                        "umass_coherence": umass_coherence_dict,
                                        "cv_coherence": cv_coherence_dict,
                                        "nmi": nmi_dict}
        self.benchmark_metrics_dict = metrics_dict
        self.model = lda
