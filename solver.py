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
import random


"""
To run properly, make sure all the .out files first have valid graphs (by running any of the solve algorithms below)

Then run: for i in {1..1000}; do for file in inputs/*;do for j in {5,10,20,30,50,100}; do pypy3 solver.py $file $j;done;done;done;

j refers to the range of random values to add to the edge weights for solve_random_MST_cut_sometimes and solve_random_MST_cut_all
"""

def solve_vertex_combinations(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    dominating_trees = []
    for n in range(1, G.number_of_nodes() + 1):
        for SG in (G.subgraph(nodes) for nodes in itertools.combinations(G, n)):
            if nx.algorithms.components.connected.is_connected(SG) and nx.algorithms.dominating.is_dominating_set(G, SG.nodes):
                dominating_tree = nx.minimum_spanning_tree(SG)
                if dominating_tree.number_of_nodes()==1:
                    return dominating_tree
                dominating_trees.append((dominating_tree, average_pairwise_distance(dominating_tree)))
                # print("adding dominating tree")
                # print(dominating_tree.nodes)

    min_dominating_tree = min(dominating_trees,key=operator.itemgetter(1))[0]

    return min_dominating_tree


def solve_vertex_combinations_MP(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    for SG in (G.subgraph(nodes) for nodes in itertools.combinations(G, 1)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G, SG.nodes):
            if SG.number_of_nodes() == 1:
                return SG

    num_cores = multiprocessing.cpu_count() // 2 - 1
    with Pool(num_cores) as p:
        result = p.starmap(process_vertices,zip(itertools.repeat(G, G.number_of_nodes()), range(2, G.number_of_nodes() + 1)))
    dominating_trees_lst = []
    for i in result:
        dominating_trees_lst += i
    min_dominating_tree = min(dominating_trees_lst, key=operator.itemgetter(1))[0]
    return min_dominating_tree


def process_vertices(G, n):
    dominating_trees = []
    for SG in (G.subgraph(nodes) for nodes in itertools.combinations(G, n)):
        if nx.algorithms.components.connected.is_connected(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            dominating_tree = nx.minimum_spanning_tree(SG)
            connected_subgraphs.append((dominating_tree, average_pairwise_distance(dominating_tree)))
            # print("adding dominating tree")
            # print(dominating_tree.nodes)

    return dominating_trees

def solve_edge_combinations_brute_force(G):
    dominating_trees = []
    for SG in (G.subgraph(nodes) for nodes in itertools.combinations(G,1)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            if SG.number_of_nodes()==1:
                return SG

    for e in range(1, G.number_of_edges() + 1):
        for SG in (G.edge_subgraph(edges) for edges in itertools.combinations(G.edges(), e)):
            if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
                dominating_trees.append((SG, average_pairwise_distance(SG)))
                # print("adding dominating tree")
                # print(SG.nodes)


    min_dominating_tree = min(dominating_trees, key=operator.itemgetter(1))[0]
    return min_dominating_tree

def solve_edge_combinations_brute_force_MP(G):
    for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G,1)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            if SG.number_of_nodes()==1:
                return SG

    num_cores = multiprocessing.cpu_count()//2-1
    with Pool(num_cores) as p:
        result = p.starmap(process_edges, zip(itertools.repeat(G, G.number_of_edges()), range(1, G.number_of_edges() + 1)))
    dominating_trees_lst = []
    for i in result:
        dominating_trees_lst += i
    min_dominating_tree = min(dominating_trees_lst,key=operator.itemgetter(1))[0]
    return min_dominating_tree

def process_edges(G, e):
    dominating_trees = []
    for SG in (G.edge_subgraph(edges) for edges in itertools.combinations(G.edges(), e)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            dominating_trees.append((SG, average_pairwise_distance(SG)))
            print("adding dominating tree")
            print(SG.nodes)
    return dominating_trees


def solve_MST_cut_all(G):
    T = nx.minimum_spanning_tree(G)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    T.remove_nodes_from(remove)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes)==1:
                break
            T.remove_node(nodeToRemove)
            removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
            setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
            # print("new 1-edge nodes: " + str(setDiff))
            for i in setDiff:
                node2 = list(T.edges(i))[0][1]
                q.put((-T[i][node2]['weight'],i))
    return T


def solve_MST_cut_sometimes(G):
    T = nx.minimum_spanning_tree(G)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    for i in remove:
        T_temporary = copy.deepcopy(T)
        T_temporary.remove_node(i)
        if average_pairwise_distance(T_temporary) <= average_pairwise_distance(T):
            T.remove_node(i)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes)==1:
                break
            T_temp = copy.deepcopy(T)
            T_temp.remove_node(nodeToRemove)
            if average_pairwise_distance(T_temp) <= average_pairwise_distance(T):
                T.remove_node(nodeToRemove)
                removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
                setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
                # print("new 1-edge nodes: " + str(setDiff))
                for i in setDiff:
                    node2 = list(T.edges(i))[0][1]
                    q.put((-T[i][node2]['weight'],i))
    return T


def solve_max_MST_cut_sometimes(G):
    G2 = copy.deepcopy(G)
    path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G2))
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += max(sum(path_lengths[n1].values()), sum(path_lengths[n2].values()))
    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    for i in remove:
        T_temporary = copy.deepcopy(T)
        T_temporary.remove_node(i)
        if average_pairwise_distance(T_temporary) <= average_pairwise_distance(T):
            T.remove_node(i)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes)==1:
                break
            T_temp = copy.deepcopy(T)
            T_temp.remove_node(nodeToRemove)
            if average_pairwise_distance(T_temp) <= average_pairwise_distance(T):
                T.remove_node(nodeToRemove)
                removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
                setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
                # print("new 1-edge nodes: " + str(setDiff))
                for i in setDiff:
                    node2 = list(T.edges(i))[0][1]
                    q.put((-T[i][node2]['weight'],i))
    return T



def solve_max_MST_cut_all(G):
    G2 = copy.deepcopy(G)
    path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G2))
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += max(sum(path_lengths[n1].values()), sum(path_lengths[n2].values()))
        #print(str(n1) + " and " + str(n2) + " w weight: " + str(G2[n1][n2]['weight']))

    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    T.remove_nodes_from(remove)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'], i))
    while q.qsize() > 0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes) == 1:
                break
            T.remove_node(nodeToRemove)
            removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
            setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
            # print("new 1-edge nodes: " + str(setDiff))
            for i in setDiff:
                node2 = list(T.edges(i))[0][1]
                q.put((-T[i][node2]['weight'], i))
    return T


def solve_sum_MST_cut_sometimes(G):
    G2 = copy.deepcopy(G)
    path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G2))
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += sum(path_lengths[n1].values())+ sum(path_lengths[n2].values())
    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    for i in remove:
        T_temporary = copy.deepcopy(T)
        T_temporary.remove_node(i)
        if average_pairwise_distance(T_temporary) <= average_pairwise_distance(T):
            T.remove_node(i)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes)==1:
                break
            T_temp = copy.deepcopy(T)
            T_temp.remove_node(nodeToRemove)
            if average_pairwise_distance(T_temp) <= average_pairwise_distance(T):
                T.remove_node(nodeToRemove)
                removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
                setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
                # print("new 1-edge nodes: " + str(setDiff))
                for i in setDiff:
                    node2 = list(T.edges(i))[0][1]
                    q.put((-T[i][node2]['weight'],i))
    return T

def solve_sum_MST_cut_all(G):
    G2 = copy.deepcopy(G)
    path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G2))
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += sum(path_lengths[n1].values())+ sum(path_lengths[n2].values())
        #print(str(n1) + " and " + str(n2) + " w weight: " + str(G2[n1][n2]['weight']))

    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    T.remove_nodes_from(remove)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'], i))
    while q.qsize() > 0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes) == 1:
                break
            T.remove_node(nodeToRemove)
            removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
            setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
            # print("new 1-edge nodes: " + str(setDiff))
            for i in setDiff:
                node2 = list(T.edges(i))[0][1]
                q.put((-T[i][node2]['weight'], i))
    return T

def solve_min_MST_cut_sometimes(G):
    G2 = copy.deepcopy(G)
    path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G2))
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += min(sum(path_lengths[n1].values()), sum(path_lengths[n2].values()))
    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    for i in remove:
        T_temporary = copy.deepcopy(T)
        T_temporary.remove_node(i)
        if average_pairwise_distance(T_temporary) <= average_pairwise_distance(T):
            T.remove_node(i)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes)==1:
                break
            T_temp = copy.deepcopy(T)
            T_temp.remove_node(nodeToRemove)
            if average_pairwise_distance(T_temp) <= average_pairwise_distance(T):
                T.remove_node(nodeToRemove)
                removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
                setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
                # print("new 1-edge nodes: " + str(setDiff))
                for i in setDiff:
                    node2 = list(T.edges(i))[0][1]
                    q.put((-T[i][node2]['weight'],i))
    return T

def solve_min_MST_cut_all(G):
    G2 = copy.deepcopy(G)
    path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G2))
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += min(sum(path_lengths[n1].values()), sum(path_lengths[n2].values()))
        #print(str(n1) + " and " + str(n2) + " w weight: " + str(G2[n1][n2]['weight']))

    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    T.remove_nodes_from(remove)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'], i))
    while q.qsize() > 0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes) == 1:
                break
            T.remove_node(nodeToRemove)
            removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
            setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
            # print("new 1-edge nodes: " + str(setDiff))
            for i in setDiff:
                node2 = list(T.edges(i))[0][1]
                q.put((-T[i][node2]['weight'], i))
    return T

def solve_random_MST_cut_sometimes(G, top):
    G2 = copy.deepcopy(G)
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += random.randrange(1, int(top), 1)
    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    for i in remove:
        T_temporary = copy.deepcopy(T)
        T_temporary.remove_node(i)
        if average_pairwise_distance(T_temporary) <= average_pairwise_distance(T):
            T.remove_node(i)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'],i))
    while q.qsize()>0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes)==1:
                break
            T_temp = copy.deepcopy(T)
            T_temp.remove_node(nodeToRemove)
            if average_pairwise_distance(T_temp) <= average_pairwise_distance(T):
                T.remove_node(nodeToRemove)
                removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
                setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
                # print("new 1-edge nodes: " + str(setDiff))
                for i in setDiff:
                    node2 = list(T.edges(i))[0][1]
                    q.put((-T[i][node2]['weight'],i))
    return T

def solve_random_MST_cut_all(G,top):
    G2 = copy.deepcopy(G)
    for e in G2.edges():
        n1 = e[0]
        n2 = e[1]
        G2[n1][n2]['weight'] += random.randrange(1, int(top), 1)
        #print(str(n1) + " and " + str(n2) + " w weight: " + str(G2[n1][n2]['weight']))

    T = nx.minimum_spanning_tree(G2)
    T = nx.Graph(G.edge_subgraph(T.edges()))
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    T.remove_nodes_from(remove)
    remove = [node for node, degree in dict(T.degree()).items() if degree == 1]
    # print(remove)
    q = PriorityQueue()
    for i in remove:
        n2 = list(T.edges(i))[0][1]
        q.put((-T[i][n2]['weight'], i))
    while q.qsize() > 0:
        nodeToRemove = q.get()[1]
        # print("removing node: " + str(nodeToRemove))
        G2 = copy.deepcopy(G)
        G2.remove_node(nodeToRemove)
        nodesInGraphNotTree = [node for node in G2.nodes if not node in T.nodes]
        canRemove = True
        for n in nodesInGraphNotTree:
            if len(set(G2.neighbors(n)) & set(T.nodes)) == 0:
                canRemove = False
                break
        if canRemove:
            # print("can remove node!")
            if len(T.nodes) == 1:
                break
            T.remove_node(nodeToRemove)
            removeMore = [node for node, degree in dict(T.degree()).items() if degree == 1]
            setDiff = list(set(removeMore).difference(set([x[1] for x in q.queue])))
            # print("new 1-edge nodes: " + str(setDiff))
            for i in setDiff:
                node2 = list(T.edges(i))[0][1]
                q.put((-T[i][node2]['weight'], i))
    return T


def solve_steiner(G):
    # Set all node weights to be the min weight of adjacent edges

    for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G,1)):
        if nx.algorithms.tree.recognition.is_tree(SG) and nx.algorithms.dominating.is_dominating_set(G,SG.nodes):
            if SG.number_of_nodes()==1:
                return SG
    # path_lengths = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G))
    for u in range(G.number_of_nodes()):
        adj_edges = [G.edges[u, v]['weight'] for v in list(G[u])]
        #G.nodes[u]['weight'] = sum(adj_edges)/len(adj_edges)
        # sum_paths = sum((path_lengths[u]).values())
        # G.nodes[u]['weight'] = sum_paths
        G.nodes[u]['weight'] = min(adj_edges)
    # Obtain dominating set that minimizes node weights
    ds = nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, 'weight')
    # Connect the set with steiner tree approx
    T = nx.algorithms.approximation.steinertree.steiner_tree(G, ds)
    return T



# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 3
    path = sys.argv[1]
    rang = sys.argv[2]
    print(path)
    G = read_input_file(path)
    start = time.time()
    try:
        current_T = read_output_file('outputs/' + path.split('.')[0].split('/')[1] + '.out', G)

        if current_T.number_of_nodes()>1:

            if G.number_of_edges() < 24:
                # DON'T USE MP, MUCH SLOWER SINCE DATA HAS TO BE COPIED TO PROCESSORS!!!
                # T = solve_edge_combinations_brute_force_MP(G)
                T = solve_edge_combinations_brute_force(G)
                assert is_valid_network(G, T)
                if len(T) == 1 or average_pairwise_distance(T) < average_pairwise_distance(current_T):
                    write_output_file(T, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
                    print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
                else:
                    print("not better pairwise dist: " + str(average_pairwise_distance(current_T)))
                if len(T)==1:
                    print("MP_edge_brute_force new pairwise distance: 0")
                else:
                    print("MP_edge_brute_force new pairwise distance: {}".format(average_pairwise_distance(T)))
            else:
        #     not using this because it takes too long and isn't an actual brute force (the other approximations work better)
        #     if G.number_of_nodes() < 24:
        #         T = solve_vertex_combinations(G)
        #         #T = solve_vertex_combinations_MP(G)
        #         end = time.time()
        #         assert is_valid_network(G, T)
        #         if len(T)==1 or average_pairwise_distance(T) < average_pairwise_distance(current_T):
        #             write_output_file(T, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
        #             print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
        #         else:
        #             print("not better pairwise dist" + str(average_pairwise_distance(current_T)))
        #         if len(T)==1:
        #             print("MP_vertex_combos new pairwise distance: 0")
        #         else:
        #             print("MP_vertex_combos new pairwise distance: {}".format(average_pairwise_distance(T)))
                T = solve_steiner(G)
                T2 = solve_MST_cut_all(G)
                T3 = solve_max_MST_cut_all(G)
                T4 = solve_sum_MST_cut_all(G)
                T5 = solve_min_MST_cut_all(G)
                T6 = solve_random_MST_cut_all(G,rang)
                T7 = solve_MST_cut_sometimes(G)
                T8 = solve_max_MST_cut_sometimes(G)
                T9 = solve_sum_MST_cut_sometimes(G)
                T10 = solve_min_MST_cut_sometimes(G)
                T11 = solve_random_MST_cut_sometimes(G,rang)

                assert is_valid_network(G, T)
                assert is_valid_network(G, T2)
                assert is_valid_network(G, T3)
                assert is_valid_network(G, T4)
                assert is_valid_network(G, T5)
                assert is_valid_network(G, T6)
                assert  is_valid_network(G,T7)
                assert is_valid_network(G, T8)
                assert is_valid_network(G, T9)
                assert is_valid_network(G, T10)
                assert is_valid_network(G, T11)
                if len(T2)==1:
                    write_output_file(T2, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
                    print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
                    print("MST_cut_all new pairwise distance: 0")
                else:
                    trees = [(T, average_pairwise_distance(T), "Steiner"), (T2, average_pairwise_distance(T2), "MST_cut_all"),
                             (T3, average_pairwise_distance(T3), "max_MST_cut_all"), (T4, average_pairwise_distance(T4), "sum_MST_cut_all"),
                             (T5, average_pairwise_distance(T5), "min_MST_cut_all"), (T6, average_pairwise_distance(T6),"random_MST_cut_all"),
                             (T7, average_pairwise_distance(T7),"MST_cut_sometimes"),(T8, average_pairwise_distance(T8),"max_MST_cut_sometimes"),
                             (T9, average_pairwise_distance(T9),"sum_MST_cut_sometimes"),(T10, average_pairwise_distance(T10),"min_MST_cut_sometimes"),
                             (T11, average_pairwise_distance(T11),"random_MST_cut_sometimes")]
                    #trees = [(T11, average_pairwise_distance(T11),"Random-Special"),(T6, average_pairwise_distance(T6),"Random")]
                    best_T = min(trees,key=operator.itemgetter(1))
                    if best_T[1] < average_pairwise_distance(current_T):
                        write_output_file(best_T[0], 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
                        print("Old pairwise distance: {}".format(average_pairwise_distance(current_T)))
                    else:
                        print("not better pairwise dist: " + str(average_pairwise_distance(current_T)))
                    print(best_T[2]+" new pairwise distance: {}".format(best_T[1]))
    except:
        T = solve_MST_cut_all(G)
        assert is_valid_network(G, T)
        print("no valid out file detected")
        write_output_file(T, 'outputs/' + path.split('.')[0].split('/')[1] + '.out')
        print("MST_cut_all pairwise distance: {}".format(average_pairwise_distance(T)))


    print("total time: " + str(time.time() - start))
