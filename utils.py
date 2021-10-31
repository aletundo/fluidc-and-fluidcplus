import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.io import mmread
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score as ars
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi


def _invert_dict(orig_dict):
    return_dict = {}
    for v, k in orig_dict.items():
        try:
            return_dict[k].append(v)
        except KeyError:
            return_dict[k] = [v]
    return return_dict


def nodes_color(in_graph, communities):
    """
    Convert nodes communities to int
    Args:
        in_graph: input graph (networkX graph)
        communities: communities associated to in_graph (lists of nodes)

    Returns:
        list of colors (list)
    """
    color_list = [0] * len(in_graph.nodes)

    for color in range(len(communities)):
        node_idxs = communities[color]
        for j in node_idxs:
            color_list[j] = color

    return color_list


def from_mtx_to_graph(mtx_file, plot=False):
    """
    Read mtx file and create networkX graph
    Args:
        mtx_file: file name (string)
        plot: flag for visualizing graph (bool)

    Returns:
        networkX graph
    """
    mtx = mmread(mtx_file)
    mtx_to_dense = mtx.todense()
    in_graph = nx.from_numpy_matrix(mtx_to_dense)

    if plot:
        cmap = cm.get_cmap('jet')
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_title("Dolphins")
        nx.draw(in_graph, ax=ax1, with_labels=True, font_weight='bold', cmap=cmap,
                pos=nx.spring_layout(in_graph, seed=99))
        plt.show()

    return in_graph


def purity_score(true_labels, pred_labels):
    """
    Compute purity score
    Args:
        true_labels: groundtruth communities (list)
        pred_labels: predicted communities (list)

    Returns:
        purity score (float)
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels, pred_labels)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def graph_loader(name):
    """
    Load real graph corresponding to input name
    Args:
        name: real graph name (string)

    Returns:
        real graph (networkX graph), communities number (int)
    """
    if name == "karate":
        G = nx.karate_club_graph()
        communities_number = 2
    elif name == "football":
        G = nx.read_gml("football.gml")
        G = nx.convert_node_labels_to_integers(G)
        communities_number = 12
    elif name == "dolphins":
        G = nx.read_gml("dolphins.gml")
        G = nx.convert_node_labels_to_integers(G)
        communities_number = 2
    elif name == 'citeseer':
        G = nx.read_gml("citeseer.gml")
        G = nx.convert_node_labels_to_integers(G)
        communities_number = 6
    elif name == 'polbooks':
        G = nx.read_gml("polbooks.gml")
        G = nx.convert_node_labels_to_integers(G)
        communities_number = 3
    else:
        raise ValueError("Improper graph name as input")
    return G, communities_number


def gt_loader(G, name):
    """
    Load groundtruth from real input graph G and convert data to int
    Args:
        G: input graph (networkX graph)
        name: real graph name (string)

    Returns:
        groundtruth (list)
    """
    gt = []
    for i in range(len(G)):
        if name == "karate":
            gt_com = G.nodes[i]["club"]
            if gt_com == "Officer":
                gt.append(0)
            elif gt_com == "Mr. Hi":
                gt.append(1)
            else:
                raise ValueError("Invalid community label")
        elif name == 'dolphins':
            gt_com = G.nodes[i]["group"]
            gt.append(gt_com)
        elif name == "football":
            gt_com = G.nodes[i]["value"]
            gt.append(gt_com)
        elif name == "citeseer":
            gt_com = G.nodes[i]["community"]
            gt.append(gt_com)
        elif name == "polbooks":
            gt_com = G.nodes[i]["value"]
            if gt_com == "n":
                gt.append(0)
            elif gt_com == "c":
                gt.append(1)
            elif gt_com == "l":
                gt.append(2)
            else:
                raise ValueError("Invalid community label")
    return gt


def plot_communities(G, plus_colors, orig_colors, gt_colors):
    """
    Plot graphs and color nodes by considering their community
    Args:
        G: input graph (networkX graph)
        plus_colors: FluidC+ communities (list)
        orig_colors: FluidC communities (list)
        gt_colors: groundtruth communities (list)

    Returns:
        None
    """
    cmap = cm.get_cmap('jet')
    plt.figure()
    plt.title('Fluidc+')
    nx.draw(G, with_labels=True, node_color=plus_colors, cmap=cmap,
            pos=nx.spring_layout(G, seed=99))
    plt.show()

    plt.figure()
    plt.title('Original')
    nx.draw(G, with_labels=True, node_color=orig_colors, cmap=cmap,
            pos=nx.spring_layout(G, seed=99))
    plt.show()

    plt.figure()
    plt.title('GT')
    nx.draw(G, with_labels=True, node_color=gt_colors, cmap=cmap,
            pos=nx.spring_layout(G, seed=99))
    plt.show()


def print_metrics(gt, plus_colors, orig_colors, optional_msg=None):
    """
    Print FluidC and FluidC+ metrics
    Args:
        gt: groundtruth communities (list)
        plus_colors: FluidC+ communities (list)
        orig_colors: FluidC communities (list)
        optional_msg: optional message to print when results are shown (string)

    Returns:
        FludiC+ metrics (list), FluidC metrics (list)
    """

    # FluidC+ scores
    nmi_score_plus = nmi(gt, plus_colors)
    ars_score_plus = ars(gt, plus_colors)
    pur_score_plus = purity_score(gt, plus_colors)

    # FluidC original scores
    nmi_score_orig = nmi(gt, orig_colors)
    ars_score_orig = ars(gt, orig_colors)
    pur_score_orig = purity_score(gt, orig_colors)

    plus_metrics = [nmi_score_plus, ars_score_plus, pur_score_plus]
    orig_metrics = [nmi_score_orig, ars_score_orig, pur_score_orig]

    if optional_msg:
        print(optional_msg)
    print("Normalized Mutual Information (NMI)")
    print(f'FluidC+: {nmi_score_plus},\t Original: {nmi_score_orig}')
    print('')
    print("Adjusted Rand Index (ARI)")
    print(f'FluidC+: {ars_score_plus},\t Original: {ars_score_orig}')
    print('')
    print("Cluster Purity (CP)")
    print(f'FluidC+: {pur_score_plus},\t Original: {pur_score_orig}')
    print('')

    return plus_metrics, orig_metrics
