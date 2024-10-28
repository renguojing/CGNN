from __future__ import division
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import models.node2vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import spectral_embedding
import scipy
import torch
from utils import *
import models.GAE
import models.GAT
import models.GraphSAGE

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

# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split

    test_edges_list = test_edges.tolist()  # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list]  # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)

# Input: NetworkX training graph, train_test_split (from preprocessing)
# Output: dictionary with ROC AUC, AP, Runtime
def common_neighbors_scores(train_test_split):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack input
    g_train = nx.Graph(adj_train)

    # start_time = time.time()
    # cn_scores = {}

    # Calculate scores
    cn_matrix = np.zeros(adj_train.shape)
    ebunch =get_ebunch(train_test_split)
    for (u,v) in ebunch:
        p=len(list(nx.common_neighbors(g_train, u,v)))
        cn_matrix[u][v] = p
        cn_matrix[v][u] = p
    return cn_matrix
    # cn_matrix = cn_matrix / cn_matrix.max()  # Normalize matrix

    # runtime = time.time() - start_time
    # cn_roc, cn_ap = get_roc_score(test_edges, test_edges_false, cn_matrix)

    # cn_scores['cn_test_auc'] = cn_roc
    # cn_scores['cn_test_ap'] = cn_ap
    # cn_scores['cn_runtime'] = runtime
    # return cn_scores

# Input: NetworkX training graph, train_test_split (from preprocessing)
# Output: dictionary with ROC AUC, AP, Runtime
def adamic_adar_scores(train_test_split):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack input
    g_train = nx.Graph(adj_train)

    # start_time = time.time()
    # aa_scores = {}

    # Calculate scores
    aa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.adamic_adar_index(g_train, ebunch=get_ebunch(
            train_test_split)):  # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p  # make sure it's symmetric
    return aa_matrix
    # aa_matrix = aa_matrix / aa_matrix.max()  # Normalize matrix

    # runtime = time.time() - start_time
    # aa_roc, aa_ap = get_roc_score(test_edges, test_edges_false, aa_matrix)

    # aa_scores['aa_test_auc'] = aa_roc
    # aa_scores['aa_test_ap'] = aa_ap
    # aa_scores['aa_runtime'] = runtime
    # return aa_scores

# Input: NetworkX training graph, train_test_split (from preprocessing)
# Output: dictionary with ROC AUC, AP, Runtime
def jaccard_coefficient_scores(train_test_split):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split # Unpack input
    g_train = nx.Graph(adj_train)

    # start_time = time.time()
    # jc_scores = {}

    # Calculate scores
    jc_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.jaccard_coefficient(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Jaccard coefficient
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p # make sure it's symmetric
    return jc_matrix
    # jc_matrix = jc_matrix / jc_matrix.max() # Normalize matrix

    # runtime = time.time() - start_time
    # jc_roc, jc_ap = get_roc_score(test_edges, test_edges_false, jc_matrix)

    # jc_scores['jc_test_auc'] = jc_roc
    # jc_scores['jc_test_ap'] = jc_ap
    # jc_scores['jc_runtime'] = runtime
    # return jc_scores

def preferential_attachment_scores(train_test_split):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split # Unpack input
    g_train = nx.Graph(adj_train)

    # start_time = time.time()
    # pa_scores = {}

    # 计算得分
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train, ebunch=get_ebunch(train_test_split)): #  (u, v) = 节点索引, p = PA 指数
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # 确保它是对称的
    return pa_matrix
    # pa_matrix = pa_matrix / pa_matrix.max() # 归一化矩阵

    # runtime = time.time() - start_time
    # pa_roc, pa_ap = get_roc_score(test_edges, test_edges_false, pa_matrix)

    # pa_scores['pa_test_roc'] = pa_roc
    # pa_scores['pa_test_ap'] = pa_ap
    # pa_scores['pa_runtime'] = runtime
    # return pa_scores

def resource_allocation_index_scores(train_test_split):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack input
    g_train = nx.Graph(adj_train)

    # start_time = time.time()
    # ra_scores = {}

    # Calculate scores
    ra_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.resource_allocation_index(g_train, ebunch=get_ebunch(
            train_test_split)):  # (u, v) = node indices, p = RA index
        ra_matrix[u][v] = p
        ra_matrix[v][u] = p  # make sure it's symmetric
    return ra_matrix
    # ra_matrix = ra_matrix / ra_matrix.max()  # Normalize matrix

    # runtime = time.time() - start_time
    # ra_roc, ra_ap = get_roc_score(test_edges, test_edges_false, ra_matrix)

    # ra_scores['ra_test_auc'] = ra_roc
    # ra_scores['ra_test_ap'] = ra_ap
    # ra_scores['ra_runtime'] = runtime
    # return ra_scores

def common_neighbor_centrality_scores(train_test_split, alpha=0.8):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split # Unpack input
    g_train = nx.Graph(adj_train)

    # start_time = time.time()
    # ccpa_scores = {}

    # Calculate scores
    ccpa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.common_neighbor_centrality(g_train, ebunch=get_ebunch(train_test_split), alpha=alpha): # (u, v) = node indices, p = Jaccard coefficient
        ccpa_matrix[u][v] = p
        ccpa_matrix[v][u] = p # make sure it's symmetric
    return ccpa_matrix
    # ccpa_matrix = ccpa_matrix / ccpa_matrix.max() # Normalize matrix

    # runtime = time.time() - start_time
    # ccpa_roc, ccpa_ap = get_roc_score(test_edges, test_edges_false, ccpa_matrix)

    # ccpa_scores['ccpa_test_auc'] = ccpa_roc
    # ccpa_scores['ccpa_test_ap'] = ccpa_ap
    # ccpa_scores['ccpa_runtime'] = runtime
    # return ccpa_scores

def spectral_clustering_scores(train_test_split, d=16, random_state=0):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split # Unpack input

    # start_time = time.time()
    # sc_scores = {}

    # 进行谱聚类链接预测
    adj_train = scipy.sparse.csr_matrix(adj_train) 
    spectral_emb = spectral_embedding(adj_train, n_components=d, random_state=random_state)
    # sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)
    sc_score_matrix = cosine_similarity(spectral_emb)
    return sc_score_matrix

    # runtime = time.time() - start_time
    # sc_test_roc, sc_test_ap = get_roc_score(test_edges, test_edges_false, sc_score_matrix)

    # # 记录得分
    # sc_scores['sc_test_roc'] = sc_test_roc
    # sc_scores['sc_test_ap'] = sc_test_ap
    # sc_scores['sc_runtime'] = runtime
    # return sc_scores

def svd_scores(train_test_split, d=32):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split # Unpack input

    # start_time = time.time()
    # svd_scores = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_train = torch.FloatTensor(adj_train)
    adj_train = adj_train.to(device)
    U, sigma, Vt = torch.linalg.svd(adj_train, full_matrices=False)
    emb = adj_train @ Vt.T[:,:d]
    emb = emb.detach().cpu()
    svd_score_matrix = cosine_similarity(emb)
    return svd_score_matrix

    # runtime = time.time() - start_time
    # svd_test_roc, svd_test_ap = get_roc_score(test_edges, test_edges_false, svd_score_matrix)

    # svd_scores['svd_test_roc'] = svd_test_roc
    # svd_scores['svd_test_ap'] = svd_test_ap
    # svd_scores['svd_runtime'] = runtime
    # return svd_scores

# Input: NetworkX training graph, train_test_split (from preprocessing), n2v hyperparameters
# Output: dictionary with ROC AUC, AP, Runtime
def node2vec_scores(
        train_test_split,
        P=1,  # Return hyperparameter
        Q=1,  # In-out hyperparameter
        WINDOW_SIZE=10,  # Context size for optimization
        NUM_WALKS=10,  # Number of walks per source
        WALK_LENGTH=80,  # Length of walk per source
        DIMENSIONS=128,  # Embedding dimension
        DIRECTED=False,  # Graph directed/undirected
        WORKERS=4,  # Num. parallel workers
        ITER=1,  # SGD epochs
        edge_score_mode="edge-emb",  # Whether to use bootstrapped edge embeddings + LogReg (like in node2vec paper),
        # or simple dot-product (like in GAE paper) for edge scoring,
        #or cosine in MODEL paper
        verbose=1
):

    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

    g_train = nx.Graph(adj_train)
    
    if g_train.is_directed():
        DIRECTED = True    

    # start_time = time.time()

    # Preprocessing, generate walks
    if verbose >= 1:
        print('Preprocessing grpah for node2vec...','p=',P,' q=',Q)
    g_n2v = models.node2vec.Graph(g_train, DIRECTED, P, Q)  # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [list(map(str, walk)) for walk in walks]

    # Train skip-gram model
    # model = Word2Vec(walks, vector_size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, epochs=ITER)
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    # Store embeddings mapping
    emb_mappings = model.wv

    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)
    score_matrix = cosine_similarity(emb_matrix)
    return score_matrix

    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    # if edge_score_mode == "edge-emb":

    #     def get_edge_embeddings(edge_list):
    #         embs = []
    #         for edge in edge_list:
    #             node1 = edge[0]
    #             node2 = edge[1]
    #             emb1 = emb_matrix[node1]
    #             emb2 = emb_matrix[node2]
    #             edge_emb = np.multiply(emb1, emb2)#内积，哈达玛积
    #             embs.append(edge_emb)
    #         embs = np.array(embs)
    #         return embs

    #     # Train-set edge embeddings
    #     pos_train_edge_embs = get_edge_embeddings(train_edges)
    #     neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    #     train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    #     # Create train-set edge labels: 1 = real edge, 0 = false edge
    #     train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    #     # Test-set edge embeddings, labels
    #     pos_test_edge_embs = get_edge_embeddings(test_edges)
    #     neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    #     test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    #     # Create val-set edge labels: 1 = real edge, 0 = false edge
    #     test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    #     # Train logistic regression classifier on train-set edge embeddings
    #     edge_classifier = LogisticRegression(random_state=0)
    #     edge_classifier.fit(train_edge_embs, train_edge_labels)

    #     # Predicted edge scores: probability of being of class "1" (real edge)
    #     test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    #     runtime = time.time() - start_time

    #     # Calculate scores
    #     n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    #     # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    #     n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    # elif edge_score_mode == "dot-product":
    #     score_matrix = np.dot(emb_matrix, emb_matrix.T)
    #     runtime = time.time() - start_time

    #     # Test set scores
    #     n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    # elif edge_score_mode == "cos":
    #     score_matrix = cosine_similarity(emb_matrix)
    #     runtime = time.time() - start_time
    #     # Test set scores
    #     n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix)

    # else:
    #     print("Invalid edge_score_mode! Either use edge-emb, dot-product or cos.")

    # # Record scores
    # n2v_scores = {}

    # n2v_scores['n2v_test_auc'] = n2v_test_roc
    # n2v_scores['n2v_test_ap'] = n2v_test_ap
    # n2v_scores['n2v_runtime'] = runtime

    # return n2v_scores

def gae_scores(train_test_split, dim=16, lr=0.01, epochs=200, verbose=0):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

    g = nx.Graph(adj_train)
    edge_index = get_edgeindex(np.array(g.edges()))
    x = get_svd(adj_train, d=32)
    # x = torch.FloatTensor(adj_train)
    embedding = models.GAE.get_embedding(x, edge_index, dim=dim, lr=lr, epochs=epochs, verbose=verbose, \
            test_edges=test_edges, test_edges_false=test_edges_false)
    score_matrix = cosine_similarity(embedding)
    return score_matrix

def gat_scores(train_test_split, dim=16, hidden=32, heads=1, lr=0.01, epochs=200, verbose=0):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

    g = nx.Graph(adj_train)
    edge_index = get_edgeindex(np.array(g.edges()))
    x = get_svd(adj_train, d=32)
    # x = torch.FloatTensor(adj_train)
    embedding = models.GAT.get_embedding(x, edge_index, dim=dim, hidden=hidden, heads=heads, lr=lr, epochs=epochs, verbose=verbose, \
            test_edges=test_edges, test_edges_false=test_edges_false)
    score_matrix = cosine_similarity(embedding)
    return score_matrix

def sage_scores(train_test_split, dim=16, lr=0.01, epochs=200, verbose=0):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

    g = nx.Graph(adj_train)
    edge_index = get_edgeindex(np.array(g.edges()))
    x = get_svd(adj_train, d=32)
    # x = torch.FloatTensor(adj_train)
    embedding = models.GraphSAGE.get_embedding(x, edge_index, dim=dim, lr=lr, epochs=epochs, verbose=verbose, \
            test_edges=test_edges, test_edges_false=test_edges_false)
    score_matrix = cosine_similarity(embedding)
    return score_matrix