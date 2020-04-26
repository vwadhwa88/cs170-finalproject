import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance
import itertools
import copy
import sys
import operator
from collections import OrderedDict,deque
from queue import PriorityQueue
from networkx.algorithms import approximation
import time
import multiprocessing
from multiprocessing import Pool, cpu_count

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
    for nb_nodes in range(1, G.number_of_nodes() + 1):
        for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
            if nx.algorithms.components.connected.is_connected(SG) and nx.algorithms.dominating.is_dominating_set(G, SG.nodes):
                SG = nx.minimum_spanning_tree(SG)
                if SG.number_of_nodes()==1:
                    return SG
                all_connected_subgraphs.append((SG, average_pairwise_distance(SG)))
                #print("adding subgraph")
                #print(SG.nodes)

    print("finding best subgraph")
    minSG = min(all_connected_subgraphs,key=operator.itemgetter(1))[0]

    return minSG


def solveMP(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G,1)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            if SG.number_of_nodes()==1:
                return SG

    # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
    num_cores = multiprocessing.cpu_count()//2
    with Pool(num_cores) as p:
        result = p.starmap(process, zip(itertools.repeat(G, G.number_of_nodes()), range(2, G.number_of_nodes() + 1)))
    s = time.time()
    SG_lst = []
    for i in result:
        SG_lst += i
    minSG = min(SG_lst,key=operator.itemgetter(1))[0]
    print("finding best sg time: " + str(time.time()-s))
    return minSG

def process(G, nb_nodes):
    connected_subgraphs = []
    for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
        if nx.algorithms.components.connected.is_connected(SG) and nx.algorithms.dominating.is_dominating_set(G, SG.nodes):
            SG = nx.minimum_spanning_tree(SG)
            connected_subgraphs.append((SG,average_pairwise_distance(SG)))
            #print("adding subgraph")
            #print(SG.nodes)
                
    return connected_subgraphs

# MST approach, where we cut adjacent vertices after
def solve2(G):
    T = nx.minimum_spanning_tree(G)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    print(remove)
    T.remove_nodes_from(remove)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            print("can remove node!")
            if len(T.nodes)==1:
                break
            T.remove_node(nodeToRemove)
            removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
            setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
            print("new 1-edge nodes: " + str(setDiff))
            for i in setDiff:
                node2 = list(T.edges(i))[0][1]
                q.put((-T[i][node2]['weight'],i))
    return T


def solve3(G):
    # Set all node weights to be the min weight of adjacent edges

    for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G,1)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            if SG.number_of_nodes()==1:
                return SG

    for u in range(G.number_of_nodes()):
        adj_edges = [G.edges[u, v]['weight'] for v in list(G[u])]
        #G.nodes[u]['weight'] = sum(adj_edges)/len(adj_edges)
        G.nodes[u]['weight'] = min(adj_edges)
    # Obtain dominating set that minimizes node weights
    ds = nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, 'weight')
    # Connect the set with steiner tree approx
    T = nx.algorithms.approximation.steinertree.steiner_tree(G, ds)
    return T

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    print(path)
    G = read_input_file(path)


    if G.number_of_nodes() == 24:
        print("multiprocessing brute force")
        all_connected_subgraphs = []
        start = time.time()
        T = solve(G)
        end = time.time()
        assert is_valid_network(G, T)
        current_T = read_output_file('outputs/' + path.split('.')[0].split('/')[1] + '.out', G)
        if len(T)==1 or average_pairwise_distance(T) < average_pairwise_distance(current_T):
            write_output_file(T, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
            print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
        else:
            print("not better pairwise dist")
        if len(T)==1:
            print("one vertex")
        else:
            print("New pairwise distance: {}".format(average_pairwise_distance(T)))
            print("total time: " + str(end - start))
    else:
        print("MST + cut")
        T = solve2(G)
        T2 = solve3(G)
        assert is_valid_network(G, T)
        assert is_valid_network(G, T2)
        if len(T)==1:
            write_output_file(T, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
            print("one vertex")
        elif len(T2)==1:
            write_output_file(T2, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
            print("one vertex")
        else:
            if average_pairwise_distance(T) <= average_pairwise_distance(T2):
                current_T = read_output_file('outputs/' + path.split('.')[0].split('/')[1] + '.out',G)
                if average_pairwise_distance(T) < average_pairwise_distance(current_T):
                    write_output_file(T, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
                    print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
                else:
                    print("not better pairwise dist")
                print("New MST pairwise distance: {}".format(average_pairwise_distance(T)))
            else:
                current_T = read_output_file('outputs/' + path.split('.')[0].split('/')[1] + '.out',G)
                if average_pairwise_distance(T2) < average_pairwise_distance(current_T):
                    write_output_file(T2, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
                    print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
                else:
                    print("not better pairwise dist")
                print("New Steiner pairwise distance: {}".format(average_pairwise_distance(T2)))



