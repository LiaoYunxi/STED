import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
import plotly.figure_factory as ff

from typing import Optional, Union, Tuple,List,Any, Callable

import logging

class MyLogger:
    def __init__(self):
        self.logger = logging.getLogger("BERTopic")

    def configure(self, level):
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]


def select_topic_representation(
    ctfidf_embeddings: Optional[Union[np.ndarray, csr_matrix]] = None,
    embeddings: Optional[Union[np.ndarray, csr_matrix]] = None,
    use_ctfidf: bool = True,
    output_ndarray: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Select the topic representation.
    Arguments:
        ctfidf_embeddings: The c-TF-IDF embedding matrix
        embeddings: The topic embedding matrix
        use_ctfidf: Whether to use the c-TF-IDF representation. If False, topics embedding representation is used, if it exists. Default is True.
        output_ndarray: Whether to convert the selected representation into ndarray
    Raises
        ValueError:
            - If no topic representation was found
            - If c-TF-IDF embeddings are not a numpy array or a scipy.sparse.csr_matrix
    Returns:
        The selected topic representation and a boolean indicating whether it is c-TF-IDF.
    """

    def to_ndarray(array: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if isinstance(array, csr_matrix):
            return array.toarray()
        return array

    logger = MyLogger()

    if use_ctfidf:
        if ctfidf_embeddings is None:
            logger.warning(
                "No c-TF-IDF matrix was found despite it is supposed to be used (`use_ctfidf` is True). "
                "Defaulting to semantic embeddings."
            )
            repr_, ctfidf_used = embeddings, False
        else:
            repr_, ctfidf_used = ctfidf_embeddings, True
    else:
        if embeddings is None:
            logger.warning(
                "No topic embeddings were found despite they are supposed to be used (`use_ctfidf` is False). "
                "Defaulting to c-TF-IDF representation."
            )
            repr_, ctfidf_used = ctfidf_embeddings, True
        else:
            repr_, ctfidf_used = embeddings, False

    return to_ndarray(repr_) if output_ndarray else repr_, ctfidf_used

def validate_distance_matrix(X, n_samples):
    """Validate the distance matrix and convert it to a condensed distance matrix
    if necessary.

    A valid distance matrix is either a square matrix of shape (n_samples, n_samples)
    with zeros on the diagonal and non-negative values or condensed distance matrix
    of shape (n_samples * (n_samples - 1) / 2,) containing the upper triangular of the
    distance matrix.

    Arguments:
        X: Distance matrix to validate.
        n_samples: Number of samples in the dataset.

    Returns:
        X: Validated distance matrix.

    Raises:
        ValueError: If the distance matrix is not valid.
    """
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

def visualize_hierarchy(
    topic_model,
    orientation: str = "left",
    topics: List[int] = None,
    top_n_topics: int = None,
    use_ctfidf: bool = True,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Hierarchical Clustering</b>",
    width: int = 1000,
    height: int = 600,
    hierarchical_topics: pd.DataFrame = None,
    linkage_function: Callable[[csr_matrix], np.ndarray] = None,
    distance_function: Callable[[csr_matrix], csr_matrix] = None,
    color_threshold: int = 1,
) -> go.Figure:

    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, "ward", optimal_ordering=True)

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Select embeddings
    all_topics = topic_model.unique_topics_
    indices = np.array([all_topics.index(topic) for topic in topics])

    # Select topic embeddings
    embeddings = select_topic_representation(topic_model.c_tf_idf_, topic_model.topic_embeddings_, use_ctfidf)[0][
        indices
    ]

    embeddings = np.nan_to_num(embeddings, nan=0.0)
    
    # Annotations
    if hierarchical_topics is not None and len(topics) == len(freq_df.Topic.to_list()):
        annotations = _get_annotations(
            topic_model=topic_model,
            hierarchical_topics=hierarchical_topics,
            embeddings=embeddings,
            distance_function=distance_function,
            linkage_function=linkage_function,
            orientation=orientation,
            custom_labels=custom_labels,
        )
    else:
        annotations = None

    # wrap distance function to validate input and return a condensed distance matrix
    distance_function_viz = lambda x: validate_distance_matrix(distance_function(x), embeddings.shape[0])
    # Create dendogram
    fig = ff.create_dendrogram(
        embeddings,
        orientation=orientation,
        distfun=distance_function_viz,
        linkagefun=linkage_function,
        hovertext=annotations,
        color_threshold=color_threshold,
    )

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis"
    if isinstance(custom_labels, str):
        new_labels = [
            [[str(x), None]] + topic_model.topic_aspects_[custom_labels][x] for x in fig.layout[axis]["ticktext"]
        ]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        new_labels = [
            topic_model.custom_labels_[topics[int(x)] + topic_model._outliers] for x in fig.layout[axis]["ticktext"]
        ]
    else:
        new_labels = [
            [[str(topics[int(x)]), None]] + topic_model.get_topic(topics[int(x)]) for x in fig.layout[axis]["ticktext"]
        ]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor="#ECEFF1",
        template="plotly_white",
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(
            height=200 + (15 * len(topics)),
            width=width,
            yaxis=dict(tickmode="array", ticktext=new_labels,side="left"),
        )

        # Fix empty space on the bottom of the graph
        y_max = max([trace["y"].max() + 5 for trace in fig["data"]])
        y_min = min([trace["y"].min() - 5 for trace in fig["data"]])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(
            width=200 + (15 * len(topics)),
            height=height,
            xaxis=dict(tickmode="array", ticktext=new_labels),
        )

    if hierarchical_topics is not None:
        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    marker_color="black",
                    hovertext=hovertext,
                    hoverinfo="text",
                    mode="markers",
                    showlegend=False,
                )
            )
    return fig

def _get_annotations(
    topic_model,
    hierarchical_topics: pd.DataFrame,
    embeddings: csr_matrix,
    linkage_function: Callable[[csr_matrix], np.ndarray],
    distance_function: Callable[[csr_matrix], csr_matrix],
    orientation: str,
    custom_labels: bool = False,
) -> List[List[str]]:
    """Get annotations by replicating linkage function calculation in scipy.

    Arguments:
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children.
                             NOTE: The hierarchical topic names are only visualized
                             if both `topics` and `top_n_topics` are not set.
        embeddings: The c-TF-IDF matrix on which to model the hierarchy
        linkage_function: The linkage function to use. Default is:
                          `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
                          NOTE: Make sure to use the same `linkage_function` as used
                          in `topic_model.hierarchical_topics`.
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                           `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of
                            shape (n_samples, n_samples) with zeros on the diagonal and
                            non-negative values or condensed distance matrix of shape
                            (n_samples * (n_samples - 1) / 2,) containing the upper
                            triangular of the distance matrix.
                           NOTE: Make sure to use the same `distance_function` as used
                           in `topic_model.hierarchical_topics`.
        orientation: The orientation of the figure.
                     Either 'left' or 'bottom'
        custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       NOTE: Custom labels are only generated for the original
                       un-merged topics.

    Returns:
        text_annotations: Annotations to be used within Plotly's `ff.create_dendogram`
    """
    df = hierarchical_topics.loc[hierarchical_topics.Parent_Name != "Top", :]

    # Calculate distance
    X = distance_function(embeddings)
    X = validate_distance_matrix(X, embeddings.shape[0])

    # Calculate linkage and generate dendrogram
    Z = linkage_function(X)
    P = sch.dendrogram(Z, orientation=orientation, no_plot=True)

    # store topic no.(leaves) corresponding to the x-ticks in dendrogram
    x_ticks = np.arange(5, len(P["leaves"]) * 10 + 5, 10)
    x_topic = dict(zip(P["leaves"], x_ticks))

    topic_vals = dict()
    for key, val in x_topic.items():
        topic_vals[val] = [key]

    parent_topic = dict(zip(df.Parent_ID, df.Topics))

    # loop through every trace (scatter plot) in dendrogram
    text_annotations = []
    for index, trace in enumerate(P["icoord"]):
        fst_topic = topic_vals[trace[0]]
        scnd_topic = topic_vals[trace[2]]

        if len(fst_topic) == 1:
            if isinstance(custom_labels, str):
                fst_name = f"{fst_topic[0]}_" + "_".join(
                    list(zip(*topic_model.topic_aspects_[custom_labels][fst_topic[0]]))[0][:3]
                )
            elif topic_model.custom_labels_ is not None and custom_labels:
                fst_name = topic_model.custom_labels_[fst_topic[0] + topic_model._outliers]
            else:
                fst_name = "_".join([word for word, _ in topic_model.get_topic(fst_topic[0])][:5])
        else:
            for key, value in parent_topic.items():
                if set(value) == set(fst_topic):
                    fst_name = df.loc[df.Parent_ID == key, "Parent_Name"].values[0]

        if len(scnd_topic) == 1:
            if isinstance(custom_labels, str):
                scnd_name = f"{scnd_topic[0]}_" + "_".join(
                    list(zip(*topic_model.topic_aspects_[custom_labels][scnd_topic[0]]))[0][:3]
                )
            elif topic_model.custom_labels_ is not None and custom_labels:
                scnd_name = topic_model.custom_labels_[scnd_topic[0] + topic_model._outliers]
            else:
                scnd_name = "_".join([word for word, _ in topic_model.get_topic(scnd_topic[0])][:5])
        else:
            for key, value in parent_topic.items():
                if set(value) == set(scnd_topic):
                    scnd_name = df.loc[df.Parent_ID == key, "Parent_Name"].values[0]

        text_annotations.append([fst_name, "", "", scnd_name])

        center = (trace[0] + trace[2]) / 2
        topic_vals[center] = fst_topic + scnd_topic

    return text_annotations