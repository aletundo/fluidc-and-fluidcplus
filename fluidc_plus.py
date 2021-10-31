"""FluidC+: A novel community detection algorithm based on fluid propagation.
See: https://doi.org/10.1142/S0129183119500219

Input: G = (V,E) and k;
    set C = C1, C2,... ,Ck
    set seedSet = random select k nodes form V
    seedSet → badSeed
    sort V based on degree
    set backNum = time = 0
    for backNum < 10 and time < 100 do
        for node v in seedSet do
            set d(v)=1.0
            v → Cv
        end for
        for C != C' do
            for node v in V do
                update C'(v) using Eq. (2)
            end for
        end for
        if NMI(C') < min(NMI) then
            seedSet → badSeed
            backNum = backNum+1
        end if
        NMI(C') → NMI
        set C = C'
        set Flag = true
        while Flag do
            set seedSet = NULL
            for Ck in C do
                random select node v from Ck
                v → seedSet
            end for
            if seedSet not in badSeed then
                set Flag = false
            end if
        end while
        time = time+1
    end for
Output: C
"""

import argparse
import random
from collections import Counter
from copy import deepcopy

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

from asyn_fluid_communities import asyn_fluidc
from utils import nodes_color, _invert_dict, graph_loader, gt_loader, plot_communities, print_metrics


def fluidc_modified(G, vertices, communities, density, max_density, com_to_numvertices, max_iter):

    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        # Commented since vertices are ordered wrt their degrees
        # vertices = list(G)
        # random.shuffle(vertices)

        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]: density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbour vertex communities
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]: density[communities[v]]})
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [com for com, freq in com_counter.items()
                                    if (max_freq - freq) < 0.0001]
                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True

                    # Instead of using random selection, community are selected considering the number of nodes
                    # (density)
                    # new_com = random.choice(best_communities)
                    if len(best_communities) > 1:
                        com_densities = [max_density / com_to_numvertices[b] for b in best_communities]
                        new_com = best_communities[np.argmin(com_densities)]
                    else:
                        new_com = best_communities[0]

                    # Update previous community status
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = max_density / com_to_numvertices[communities[vertex]]
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = max_density / com_to_numvertices[communities[vertex]]
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            print('Exiting by max iterations!')
            break
    # Return results by grouping communities as list of vertices
    return list(_invert_dict(communities).values())


def fluidc_plus(G, k, max_iter=100):
    """

    Args:
        G: input graph (networkX graph)
        k: number of communities (int)
        max_iter: maximum number of FluidC+ iterations (int)

    Returns:
        communities found (lists of nodes)
    """

    max_density = 1.0
    min_NMI = 1.

    # Random seed set initialization (bad seed set)
    vertices = list(G)
    random.shuffle(vertices)
    seed_set = {n: i for i, n in enumerate(vertices[:k])}
    bad_seed_set = {} | seed_set

    # Sort by nodes degree
    sorted_vertices = sorted(G.degree, key=lambda x: x[1], reverse=True)
    sorted_vertices = [x[0] for x in sorted_vertices]

    seed_init_num = 0
    iter = 0
    density = {}
    com_to_numvertices = {}
    old_communities = None
    new_communities = None
    nmi_list = []

    while seed_init_num < 10 and iter < max_iter:

        # Initialize communities seed and vertices density
        for v in seed_set:
            com_to_numvertices[seed_set[v]] = 1
            density[seed_set[v]] = max_density

        # Call to the original FluidC algorithm (with some modification)
        in_seed_set = deepcopy(seed_set)
        new_communities = fluidc_modified(G, sorted_vertices, in_seed_set, density, max_density, com_to_numvertices,
                                          max_iter)

        # Compute NMI (Normalized Mutual Information)
        new_communities_list = nodes_color(G, new_communities)
        if iter != 0:
            NMI = nmi(new_communities_list, old_communities)
            nmi_list.append(NMI)
            if nmi(new_communities_list, old_communities) < min_NMI:
                bad_seed_set = bad_seed_set | seed_set
                seed_init_num += 1
                min_NMI = NMI

        old_communities = new_communities_list

        # Discard and create a new seed set
        # Each element belong to a different community
        seed_set = {}
        community_idx = 0
        removed_list = []
        while community_idx < len(new_communities):
            # Extract possible seed element from a community from which bad seeds were removed
            current_community = new_communities[community_idx]
            current_community = list(set(current_community) - set(removed_list))
            if not current_community:
                seed_init_num = 10
                break
            new_v = random.choice(current_community)
            if new_v in list(bad_seed_set.keys()):
                removed_list.append(new_v)
            else:
                seed_set[new_v] = community_idx
                removed_list = []
                community_idx += 1

        iter += 1

    return new_communities


def main():
    parser = argparse.ArgumentParser(description='FluidC+')
    parser.add_argument('--name', default="karate", type=str,
                        help='Input graph name. Possible values are: [karate, football, polbooks, citeseer]')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot input graph communities (NOT recommended when node number is high)')
    parser.add_argument('--seed', default=99, type=int,
                        help='Define seed for random node shuffle')
    args = parser.parse_args()

    # Load graph
    in_graph, communities_number = graph_loader(args.name)

    # Compute FluidC and FluidC+ communities
    print('Executing Fluidc and Fluidc+')
    random.seed(args.seed)
    fluidc_communties = fluidc_plus(in_graph, communities_number)
    random.seed(args.seed)
    orig_com = asyn_fluidc(in_graph, communities_number)
    plus_colors = nodes_color(in_graph, fluidc_communties)
    orig_colors = nodes_color(in_graph, orig_com)

    # Get Groundtruth
    gt = gt_loader(in_graph, args.name)

    if args.plot:
        plot_communities(in_graph, plus_colors, orig_colors, gt)

    # Get metrics: NMI, ARI and PS
    msg = f'Dataset name: {args.name}\n'
    print_metrics(gt, plus_colors, orig_colors, optional_msg=msg)


if __name__ == "__main__":
    main()
