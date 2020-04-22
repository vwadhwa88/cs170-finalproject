import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import itertools
import sys


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    all_connected_subgraphs = []

    # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
    for nb_nodes in range(2, G.number_of_nodes() + 1):
        for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
            if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G, SG.nodes):
                all_connected_subgraphs.append(SG)
    
    minSG = None
    minAPD = float('inf')
    for SG in all_connected_subgraphs:
        apd = average_pairwise_distance(SG)
        if apd < minAPD:
            minSG = SG
            minAPD = apd
    
    return minSG


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'out/test.out')
