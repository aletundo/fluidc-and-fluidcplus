import argparse
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from asyn_fluid_communities import asyn_fluidc
from fluidc_plus import fluidc_plus
from utils import nodes_color, print_metrics, graph_loader, gt_loader


def generate_synth_dataset1(folder):
    """
    Generate synthetic graphs using LFR
    Args:
        folder: destination folder (string)

    Returns:
        None
    """
    if not os.path.isdir(folder):
        raise ValueError(f'{folder} does not exists.')

    mu = 0.1
    mu_max = 0.8
    k = 15
    tau1 = 3  # Not specified in the paper
    tau2 = 1.5  # Not specified in the paper

    while mu <= mu_max:
        G = nx.generators.community.LFR_benchmark_graph(n=1000, tau1=tau1, tau2=tau2, mu=mu, average_degree=15,
                                                        max_degree=38, min_community=10, max_community=50)
        nx.set_node_attributes(G, {n: list(G.nodes[n]['community']) for n in G.nodes()}, 'community')
        nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
        # nx.set_node_attributes(G, {n: ','.join(map(str, G.nodes[n]['community'])) for n in G.nodes()}, 'community')
        file_name = folder + '/' + 'mu-' + f"{mu:.1}" + '.gml'
        nx.write_gml(G, file_name)
        print(f'Generated graph {file_name}')
        mu += 0.1


def get_communities_from_synth(G):
    """
    Get communities from input graph G
    Args:
        G: input graph (networkX graph)

    Returns:
        communities (list), number of communities (int)
    """
    nodes = list(G.nodes())
    comm_idx = 0
    communities_color = [0] * len(nodes)
    while nodes:
        current_node = nodes.pop(0)
        new_community = G.nodes[current_node]['community']
        for n in new_community:
            if n in nodes:
                communities_color[n] = comm_idx
                nodes.remove(n)
        communities_color[int(current_node)] = comm_idx
        comm_idx += 1

    return communities_color, comm_idx - 1


def synthetic_test1(folder, seed):
    """
    Test FluidC and FluidC+ on synthetic graphs and plot results
    Args:
        folder: synthetic data path (string)

    Returns:
        None
    """
    mu_list = np.arange(0.1, 0.9, 0.1).tolist()
    dest = './plots/'
    if not os.path.isdir(dest):
        os.mkdir(dest)

    nmi_list_plus = []
    ars_list_plus = []
    purity_list_plus = []

    nmi_list_orig = []
    ars_list_orig = []
    purity_list_orig = []

    for mu in mu_list:
        msg = f'Synthetic LFR graph with mu={mu:.1}'
        file_name = folder + "/mu-" + f"{mu:.1}" + ".gml"

        # Load
        G = nx.read_gml(file_name)
        G = nx.convert_node_labels_to_integers(G)
        gt, comm_num = get_communities_from_synth(G)

        # Run FluidC+ and Fluidc
        random.seed(seed)
        fluidc_com = fluidc_plus(G, comm_num)
        random.seed(seed)
        orig_com = asyn_fluidc(G, comm_num)

        # Communities colors
        plus_colors = nodes_color(G, fluidc_com)
        orig_colors = nodes_color(G, orig_com)

        # Get metrics for plot
        plus_metrics, orig_metrics = print_metrics(gt, plus_colors, orig_colors, msg)

        nmi_list_plus.append(plus_metrics[0])
        ars_list_plus.append(plus_metrics[1])
        purity_list_plus.append(plus_metrics[2])

        nmi_list_orig.append(orig_metrics[0])
        ars_list_orig.append(orig_metrics[1])
        purity_list_orig.append(orig_metrics[2])

    plt.figure()
    plt.title('Normalized Mutual Information (NMI)')
    plt.plot(mu_list, nmi_list_plus, label="FluidC+")
    plt.plot(mu_list, nmi_list_orig, label="FluidC")
    plt.xlabel("mu")
    plt.legend()
    plt.savefig(dest + "NMI-" + str(seed) + ".png", bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.title('Adjusted Rand Index (ARI)')
    plt.plot(mu_list, ars_list_plus, label="FluidC+")
    plt.plot(mu_list, ars_list_orig, label="FluidC")
    plt.xlabel("mu")
    plt.legend()
    plt.savefig(dest + "ARI-" + str(seed) + ".png", bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.title('Cluster Purity (CP)')
    plt.plot(mu_list, purity_list_plus, label="FluidC+")
    plt.plot(mu_list, purity_list_orig, label="FluidC")
    plt.xlabel("mu")
    plt.legend()
    plt.savefig(dest + "CP-" + str(seed) + ".png", bbox_inches='tight')
    plt.show()


def draw_bar_plot(datasets, seed):
    """
    Plot barplots for comparing FluidC and FluidC+ on different real datasets
    Args:
        datasets: list of dataset names (list)

    Returns:
        None
    """

    plus_metrics_list = []
    orig_metrics_list = []

    for name in datasets:
        in_graph, communities_number = graph_loader(name)
        random.seed(seed)
        fluidc_communties = fluidc_plus(in_graph, communities_number)
        random.seed(seed)
        orig_com = asyn_fluidc(in_graph, communities_number)
        plus_colors = nodes_color(in_graph, fluidc_communties)
        orig_colors = nodes_color(in_graph, orig_com)
        gt = gt_loader(in_graph, name)
        plus_metrics, orig_metrics = print_metrics(gt, plus_colors, orig_colors)
        plus_metrics_list.append(plus_metrics)
        orig_metrics_list.append(orig_metrics)

    plus_mat = np.asarray(plus_metrics_list)
    orig_mat = np.asarray(orig_metrics_list)
    X = np.arange(len(datasets)) * 2
    width = 0.5

    plots_dir = './plots/'
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)

    fig, ax = plt.subplots()
    ax.bar(X - 0.5, orig_mat[:, 0], color='tab:red', width=width, label='NMI')
    ax.bar(X + 0.0, orig_mat[:, 1], color='tab:orange', width=width, label='ARI')
    ax.bar(X + 0.5, orig_mat[:, 2], color='tab:blue', width=width, label='CP')
    ax.set_ylabel('Scores')
    ax.set_xlabel('Real Datasets')
    ax.set_title('FluidC')
    ax.set_xticks(X)
    ax.set_ylim([0, 1])
    ax.set_xticklabels(datasets)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plots_dir + 'FluidC_bar_' + str(seed) + '.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(X - 0.5, plus_mat[:, 0], color='tab:red', width=width, label='NMI')
    ax.bar(X + 0.0, plus_mat[:, 1], color='tab:orange', width=width, label='ARI')
    ax.bar(X + 0.5, plus_mat[:, 2], color='tab:blue', width=width, label='CP')
    ax.set_ylabel('Scores')
    ax.set_xlabel('Real Datasets')
    ax.set_title('FluidC+')
    ax.set_xticks(X)
    ax.set_ylim([0, 1])
    ax.set_xticklabels(datasets)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plots_dir + 'FluidC_plus_bar_' + str(seed) + '.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='FluidC+')
    parser.add_argument('--synth1', default=None,
                        help='Synthetic dataset1 folder')
    parser.add_argument('--gen', action='store_true', default=False,
                        help='Generate synthetic data')
    parser.add_argument('--bar', action='store_true', default=False,
                        help='Draw bar plots for all real datsets')
    parser.add_argument('--seed', default=1256, type=int,
                        help='Define seed for random node shuffle')
    args = parser.parse_args()

    folder1 = args.synth1
    datasets = ['karate', 'dolphins', 'football', 'polbooks', 'citeseer']

    if args.gen:
        generate_synth_dataset1(folder1)
        print('Synthetic graphs generation done.')
    if args.synth1:
        synthetic_test1(folder1, args.seed)
    if args.bar:
        draw_bar_plot(datasets, args.seed)


if __name__ == "__main__":
    main()
