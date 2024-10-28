import pickle as pkl
from scores import *
import argparse
import networkx as nx
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction")
    parser.add_argument('--anchor', default='data/fb-tt/anchor0.9.pkl')
    parser.add_argument('--split_dir', default='data/fb-tt/split0.9.pkl', help='Path to split dir')
    parser.add_argument('--method', default='GAE', help='LP method')
    return parser.parse_args()

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
    return adj, n1

print('Load data...')
args = parse_args()

with open(args.split_dir, 'rb') as f:
    train_test_split0 = pkl.load(f, encoding='iso-8859-1')

train_test_split_s, train_test_split_t = train_test_split0

adj_train_s, train_edges_s, train_edges_false_s, test_edges_s, test_edges_false_s = train_test_split_s
adj_train_t, train_edges_t, train_edges_false_t, test_edges_t, test_edges_false_t = train_test_split_t

g_s = nx.Graph(adj_train_s)
g_t = nx.Graph(adj_train_t)
train_anchor, test_anchor = pkl.load(file=open(args.anchor, 'rb'), encoding='iso-8859-1')

print("Expanding graph...")
s_edges = np.array(g_s.edges())
t_edges = np.array(g_t.edges())
adj_train, s_num = expand_graph(s_edges, t_edges, train_anchor)

train_edges_t = train_edges_t + s_num
train_edges_false_t = train_edges_false_t + s_num
test_edges_t = test_edges_t + s_num
test_edges_false_t = test_edges_false_t + s_num
train_test_split = adj_train, train_edges_t, train_edges_false_t, test_edges_t, test_edges_false_t

method = args.method
if method == 'node2vec':
    scores = node2vec_scores(train_test_split, P=1, Q=1,WINDOW_SIZE=10,NUM_WALKS=10,WALK_LENGTH=80,DIMENSIONS=128,DIRECTED=False,\
        WORKERS=12,ITER=5,edge_score_mode="cos",verbose=0)

elif method == 'SC':
    scores = spectral_clustering_scores(train_test_split, d=16)

elif method == 'SVD':
    scores = svd_scores(train_test_split, d=32)

elif method == 'GAE':
    scores = gae_scores(train_test_split, dim=16, lr=0.01, epochs=200, verbose=0)

elif method == 'GAT':
    scores = gat_scores(train_test_split, dim=16, hidden=8, heads=4, lr=0.01, epochs=200, verbose=0)

elif method == 'GraphSAGE':
    scores = sage_scores(train_test_split, dim=16, lr=0.01, epochs=200, verbose=0)

auc_s, ap_s = get_roc_score(test_edges_s, test_edges_false_s, scores)
auc_t, ap_t = get_roc_score(test_edges_t, test_edges_false_t, scores)
print(args)
print('Source network | AUC: {:.4f}, AP: {:.4f}'.format(auc_s, ap_s))
print('Target network | AUC: {:.4f}, AP: {:.4f}'.format(auc_t, ap_t))