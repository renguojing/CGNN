import networkx as nx
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
import models.node2vec
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score

def split_edges(g, ratio, seed=None):
    test_num = np.floor(len(g.edges()) * (1 - ratio))
    edges = np.array(list(g.edges()))
    np.random.seed(seed)
    np.random.shuffle(edges)
    edges = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    test_edges = []
    count = 0
    for edge in edges:
        if g.degree(edge[0]) > 1 and g.degree(edge[1]) > 1:
            g.remove_edge(edge[0], edge[1])
            test_edges.append(edge)
            count += 1
        if count >= test_num:
            break
    train_edges = list(set(g.edges()) - set(test_edges))
    return train_edges, test_edges

def mask_test_edges(g, test_frac=.1, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print('preprocessing...')

    orig_num_cc = nx.number_connected_components(g)

    edges = np.array(list(g.edges()))
    num_node = len(g.nodes())

    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    #num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    #val_edges = set()

    if verbose == True:
        print('generating test sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on很费时间
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test:
            break

    if (len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, num_node)
        idx_j = np.random.randint(0, num_node)
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, num_node)
        idx_j = np.random.randint(0, num_node)
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.to_numpy_matrix(g, nodelist=sorted(g.nodes()))

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, test_edges, test_edges_false

def get_edgeindex(edgelist):
    edge = edgelist.T
    edge_index = torch.LongTensor(edge)
    edge_index_u = torch.vstack((edge_index[1],edge_index[0]))
    edge_index = torch.hstack((edge_index,edge_index_u))
    return edge_index

def get_gt_matrix(anchors,shape):
    gt = np.zeros(shape)
    for line in anchors:
        gt[int(line[0]),int(line[1])] = 1
    return gt

def get_rsvd(mat, d = 0, seed=None):
    U, sigma, Vt = randomized_svd(mat, n_components=d, random_state=seed)
    # print(sigma,U.shape,Vt.shape)
    emb = np.matmul(mat, Vt.T)
    return emb

def get_svd(A, d=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.FloatTensor(A)
    A = A.to(device)
    U, sigma, Vt = torch.linalg.svd(A, full_matrices=False)
    emb = A @ Vt.T[:,:d]
    emb = emb.detach().cpu()
    return emb

def get_n2v(adj):
    embedding = models.node2vec.get_embedding(adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
    WORKERS=12, ITER=5, verbose=0)
    return embedding

def expand_graph(g1, g2, seeds):
    '''
    g1, g2: edgelist, min node index is 0;
    seeds: for seed in seeds, seed[0] is in g1, seed[1] is in g2 
    '''
    n1 = np.max(g1) + 1
    n2 = np.max(g2) + 1
    g2 = g2 + n1
    seeds[:, 1] = seeds[:, 1] + n1
    edges = np.vstack((g1, g2))
    edges = np.vstack((edges, seeds))
    adj = np.zeros((n1 + n2, n1 + n2))
    for edge in edges:
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1
    # edges_t = np.vstack((edges.T[1], edges.T[0]))
    # edge_index = np.hstack((edges.T, edges_t))
    # edge_index = torch.LongTensor(edge_index)
    return adj

def expand_graph1(g1, g2, seeds):
    '''
    g1, g2: edgelist, min node index is 0;
    seeds: for seed in seeds, seed[0] is in g1, seed[1] is in g2 
    '''
    n1 = np.max(g1) + 1
    # n2 = np.max(g2) + 1
    g2 = g2 + n1
    seeds[:, 1] = seeds[:, 1] + n1
    edges = np.vstack((g1, g2))
    edges = np.vstack((edges, seeds))
    # adj = np.zeros((n1 + n2, n1 + n2))
    # for edge in edges:
    #     adj[edge[0], edge[1]] = 1
    #     adj[edge[1], edge[0]] = 1
    edges_t = np.vstack((edges.T[1], edges.T[0]))
    edge_index = np.hstack((edges.T, edges_t))
    edge_index = torch.LongTensor(edge_index)
    return edge_index

def expand_edges(g_s, g_t, seeds, s_edge, t_edge):
    seed_list1 = seeds.T[0]
    seed_list2 = seeds.T[1]
    for i in range(len(seed_list1) - 1):
        for j in range(i + 1, len(seed_list1)):
            if not g_s.has_edge(seed_list1[i], seed_list1[j]) and g_t.has_edge(seed_list2[i], seed_list2[j]):
                g_s.add_edge(seed_list1[i], seed_list1[j])
                s_edge = torch.hstack((s_edge, torch.LongTensor([[seed_list1[i], seed_list1[j]], [seed_list1[j], seed_list1[i]]])))
            if g_s.has_edge(seed_list1[i], seed_list1[j]) and not g_t.has_edge(seed_list2[i], seed_list2[j]):
                g_t.add_edge(seed_list2[i], seed_list2[j])
                t_edge = torch.hstack((t_edge, torch.LongTensor([[seed_list2[i], seed_list2[j]], [seed_list2[j], seed_list2[i]]])))
    return g_s, g_t, s_edge, t_edge

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input: positive test edges, negative test edges, edge score matrix
# Output: ROC AUC score,  AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])

    # Store negative edge predictions, actual values
    preds_neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])  # 按水平方向拼接数组
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, ap_score
    return roc_score, ap_score