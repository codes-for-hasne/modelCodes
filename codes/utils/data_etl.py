import math
import sys
import os
import networkx as nx
import numpy as np

from codes.hierarchical_node.node import Node


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, '..'))

total_levels = 1
all_leaves = set()

def dfs(u, tree, level):
    global total_levels,all_leaves
    tree[u].level = level
    tree[u].path.append(u)
    if len(tree[u].direct_children) == 0:   # leaf nodes
        if level > total_levels:
            total_levels = level
        all_leaves.add(u)
        tree[u].all_children = []
        tree[u].leaves = [u]
        return
    for v in tree[u].direct_children:
        tree[v].path.extend(tree[u].path)
        dfs(v, tree, level + 1)
        tree[u].all_children = list(set(tree[u].all_children) | set(tree[v].all_children))
        tree[u].leaves = list(set(tree[u].leaves) | set(tree[v].leaves))

def get_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    return [i[1] for i in lst]

def prepare_graph(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            G.add_edge(int(items[0]), int(items[1]))
    return G






def prepare_tree(file_path):
    """
    prepare our data
    :param file_path:
    :return:
    """
    g, n, m = build_hierarchical_tree(os.path.join(file_path))

    # n:num of total nodes; m:num of leaf nodes;
    tree = [None] * n
    for each in g:
        # init each node
        tree[each] = Node(each, list(g[each].keys()), [], [], 1, [])
    level = 0
    dfs(n-1,tree,level + 1)
    ret = (tree,total_levels,list(all_leaves))
    return ret

def build_hierarchical_tree(file_path):
    G = nx.DiGraph()
    n, m = None, None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            if n is None:
                n, m = int(items[0]), int(items[1])
            else:
                G.add_edge(int(items[0]), int(items[1]))
    return G,n,m


def get_leaves_similarity(graph):
    g_mat = transfer_to_matrix(graph)
    return local_katz_sim(g_mat,3,0.01)


def transfer_to_matrix(graph):
    n = graph.number_of_nodes()
    mat = np.zeros([n, n])

    for e in graph.edges():
        mat[e[0]][e[1]] = 1
        mat[e[1]][e[0]] = 1
    return mat


def common_neighbor_sim(adj_mat_):
    """
    common neighbor similarity
    :param adj_mat_:
    :return:
    """
    n = len(adj_mat_)
    # init diagonal = 1
    adj_mat = np.zeros([n,n]) + adj_mat_
    for i in range(n):
        adj_mat[i][i] = 1
    adj_mat[np.where(adj_mat > 0)] = 1
    sim_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i,n):
            if i == j:
                sim_mat[i][j] = 1
            else:
                degree = math.sqrt(np.sum(adj_mat[i])*np.sum(adj_mat[j]))
                comNeighbor = np.sum(adj_mat[i]*adj_mat[j])
                sim_mat[i][j] = sim_mat[j][i] = comNeighbor*1.0/degree

    return sim_mat

def local_katz_sim(adj_mat_,n,alpha):
    n = len(adj_mat_)
    adj_mat = np.zeros([n,n]) + adj_mat_
    for i in range(n):
        adj_mat[i][i] = 1
    adj_mat[np.where(adj_mat > 0)] = 1

    # calc the katz in paper.
    a2 = np.dot(adj_mat,adj_mat)
    a3 = np.dot(adj_mat,a2)
    a4 = np.dot(adj_mat,a3)

    res = a2 + alpha*a3 + alpha*alpha*a4

    sim = res
    for i in range(n):

        for j in range(i,n):
            if i == j:
                sim[i][j] = 1
            else:
                degree = math.sqrt(np.sum(adj_mat[i])*np.sum(adj_mat[j]))
                sim[i][j] = sim[j][i] = res[i][j] / degree

    return sim





def get_network(self, fa_id, tree):
    n = self.n
    childst = list(tree[fa_id].childst)
    n_ch = len(childst)
    node_in_tree = []
    for i in range(n_ch):
        node_in_tree.append(childst[i])

    # return matrix
    sim_mat = np.zeros([n_ch, n_ch])
    var_mat = np.zeros([n_ch, n_ch])

    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == j:
                sim_mat[i][j] = 1
                var_mat[i][j] = 0
            else:
                coverst_i = tree[childst[i]].coverst    # all the children of i
                len_i = len(coverst_i)
                coverst_j = tree[childst[j]].coverst    # all the children of j
                len_j = len(coverst_j)

                mat = np.zeros([len_i, len_j])
                ci = 0
                cj = 0
                for p in coverst_i:
                    cj = 0
                    for q in coverst_j:
                        mat[ci][cj] = self.sim_mat_n[p][q]
                        cj = cj+1
                    ci = ci+1
                i2j = np.mean(mat, axis=1)
                sim_mat[i][j] = sim_mat[j][i] = np.mean(i2j)
                var_mat[i][j] = np.std(i2j)
                j2i = np.mean(mat, axis=0)
                var_mat[j][i] = np.std(j2i)

    return node_in_tree, sim_mat, var_mat


def get_branch_common_similarity_matrix(parent, tree, leavesMartrix):
    """
    calc the similarity.
    :param parent:
    :param tree:
    :param leavesMartrix:
    :return:
    """

    leavesNum = len(leavesMartrix)
    children = list(tree[parent].direct_children)
    childrenNum = len(children)

    # return matrix
    simMat = np.zeros([childrenNum, childrenNum])
    varMat = np.zeros([childrenNum, childrenNum])

    for i in range(childrenNum):
        for j in range(i, childrenNum):
            if i == j:
                simMat[i][j] = 1
                varMat[i][j] = 0
            else:
                iLeaves = tree[children[i]].leaves    # all the children of i
                iLen = len(iLeaves)
                jLeaves = tree[children[j]].leaves    # all the children of j
                jlen = len(jLeaves)

                mat = np.zeros([iLen, jlen])
                ci = 0
                cj = 0
                for p in iLeaves:
                    cj = 0
                    for q in jLeaves:
                        mat[ci][cj] = leavesMartrix[p][q]
                        cj = cj+1
                    ci = ci+1

                simMat[i][j] = simMat[j][i] = np.sum(mat)

    return children, simMat

def get_nodes_sim_based_on_matrix(parent, tree, leavesMartrix, scalingFactor):
    # Number of leaves.
    leavesNum = len(leavesMartrix)
    # Children of current node.
    children = list(tree[parent].direct_children)
    childrenNum = len(children)

    # return matrix
    simMat = np.zeros([childrenNum, childrenNum])

    for i in range(childrenNum):
        for j in range(i, childrenNum):
            if i == j:
                simMat[i][j] = 1
            else:
                iLeaves = tree[children[i]].leaves  # all the children of i
                iLen = len(iLeaves)
                jLeaves = tree[children[j]].leaves  # all the children of j
                jLen = len(jLeaves)

                mat = np.zeros([iLen, jLen])
                ci = 0
                cj = 0
                for p in list(iLeaves):
                    cj = 0
                    for q in list(jLeaves):
                        mat[ci][cj] = leavesMartrix[p][q]
                        cj = cj + 1
                    ci = ci + 1

                numerator = np.mean(mat)
                diff = abs(iLen - jLen)
                tune = round( diff / scalingFactor, 4 )
                denominator = math.exp( tune )
                # denominator = 1
                simMat[i][j] = simMat[j][i] = numerator / denominator
                # simMat[i][j] = simMat[j][i] = numerator
    return children, simMat

def normalize_adj_matrix(mat):
    normMatrix = mat / np.sum(mat, axis=1, keepdims=True)
    return normMatrix

def normalizeMatrix(mat):
    """
    Normalize the target matrix.
    :param mat: target
    :return:
    """
    return mat / np.sum(mat, axis=1, keepdims=True)

def getNodesSimBasedOnLeavesSim(parent, tree, leavesMatrix, scalingFactor):
    """
    Calc the similarity of all the nodes in a layer.
    :param leavesMatrix:    the leaf nodes similarity matrix
    :param childrenList:    all the children in this layer
    :param tree:    the tree
    :param scalingFactor:   scaling factor to tune the influence that the number of children
    :return:    the similarity matrix
    """
    # Total nodes number of the child layer.
    leavesNum = len(leavesMatrix)
    children = list(tree[parent].direct_children)
    childrenNum = len(children)
    # Initialize the result matrix.
    simMat = np.zeros([childrenNum, childrenNum])

    for i in range(childrenNum):
        for j in range(i, childrenNum):
            if i == j:
                simMat[i][j] = 1
            else:
                iLeaves = tree[children[i]].leaves  # all the leaf children of i
                iLen = len(iLeaves)
                jLeaves = tree[children[j]].leaves  # all the leaf children of j
                jLen = len(jLeaves)

                mat = np.zeros([iLen, jLen])
                ci = 0
                cj = 0
                for p in list(iLeaves):
                    cj = 0
                    for q in list(jLeaves):
                        mat[ci][cj] = leavesMatrix[p][q]
                        cj = cj + 1
                    ci = ci + 1

                numerator = np.mean(mat)
                diff = abs(iLen - jLen)
                tune = round(diff / scalingFactor, 4)
                denominator = math.exp(tune)
                simMat[i][j] = simMat[j][i] = numerator / denominator
    return children, simMat