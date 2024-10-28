import pickle as pkl
from scores import *
import argparse
import networkx as nx
import numpy as np
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction")
    # parser.add_argument('--s_graph', default='data/douban/graph1.pkl')
    parser.add_argument('--anchor', default='data/fb-tt/anchor0.9.pkl')
    parser.add_argument('--split_dir', default='data/fb-tt/split0.9.pkl', help='Path to split dir')
    parser.add_argument('--method', default='GAE', help='LP method')
    parser.add_argument('--normalize', default=False, type=bool)
    parser.add_argument('--verbose', default=0, type=int)
    return parser.parse_args()

def expand_edge(g_s, g_t, seeds):
    seed_list1 = seeds.T[0]
    seed_list2 = seeds.T[1]
    add1 = 0
    add2 = 0
    for i in range(len(seed_list1) - 1):
        for j in range(i + 1, len(seed_list1)):
            if not g_s.has_edge(seed_list1[i], seed_list1[j]) and g_t.has_edge(seed_list2[i], seed_list2[j]):
                g_s.add_edge(seed_list1[i], seed_list1[j])
                add1 += 1
            if g_s.has_edge(seed_list1[i], seed_list1[j]) and not g_t.has_edge(seed_list2[i], seed_list2[j]):
                g_t.add_edge(seed_list2[i], seed_list2[j])
                add2 += 1
    print("Add {:d} edges in source network and {:d} edges in target network".format(add1, add2))
    return g_s, g_t

args = parse_args()
print('Load data...')

with open(args.split_dir, 'rb') as f:
    train_test_split0 = pkl.load(f, encoding='iso-8859-1')

train_test_split_s, train_test_split_t = train_test_split0

adj_train_s, train_edges_s, train_edges_false_s, test_edges_s, test_edges_false_s = train_test_split_s
adj_train_t, train_edges_t, train_edges_false_t, test_edges_t, test_edges_false_t = train_test_split_t

g_s = nx.Graph(adj_train_s)
g_t = nx.Graph(adj_train_t)
train_anchor, test_anchor = pkl.load(file=open(args.anchor, 'rb'), encoding='iso-8859-1')

start_time = time()

print("Expanding edges...")
g_s, g_t = expand_edge(g_s, g_t, train_anchor)

adj_train_s = nx.to_numpy_matrix(g_s, nodelist=sorted(g_s.nodes()))
adj_train_t = nx.to_numpy_matrix(g_t, nodelist=sorted(g_t.nodes()))
train_test_split_s = adj_train_s, train_edges_s, train_edges_false_s, test_edges_s, test_edges_false_s
train_test_split_t = adj_train_t, train_edges_t, train_edges_false_t, test_edges_t, test_edges_false_t

method = args.method
print(method, "for link prediction...")

if method == 'CN':
    scores_s = common_neighbors_scores(train_test_split_s)
    scores_t = common_neighbors_scores(train_test_split_t)

elif method == 'AA':
    scores_s = adamic_adar_scores(train_test_split_s)
    scores_t = adamic_adar_scores(train_test_split_t)

elif method == 'JC':
    scores_s = jaccard_coefficient_scores(train_test_split_s)
    scores_t = jaccard_coefficient_scores(train_test_split_t)

elif method == 'PA':
    scores_s = preferential_attachment_scores(train_test_split_s)
    scores_t = preferential_attachment_scores(train_test_split_t)

elif method == 'RA':
    scores_s = resource_allocation_index_scores(train_test_split_s)
    scores_t = resource_allocation_index_scores(train_test_split_t)

elif method == 'CCPA':
    scores_s = common_neighbor_centrality_scores(train_test_split_s)
    scores_t = common_neighbor_centrality_scores(train_test_split_t)

elif method == 'SC':
    scores_s = spectral_clustering_scores(train_test_split_s, d=16)
    scores_t = spectral_clustering_scores(train_test_split_t, d=16)

elif method == 'SVD':
    scores_s = svd_scores(train_test_split_s, d=32)
    scores_t = svd_scores(train_test_split_t, d=32)

elif method == 'node2vec':
    scores_s = node2vec_scores(train_test_split_s,P=1,Q=1,WINDOW_SIZE=10,NUM_WALKS=10,WALK_LENGTH=80,DIMENSIONS=128,DIRECTED=False,\
        WORKERS=12,ITER=5,edge_score_mode="cos",verbose=args.verbose)
    scores_t = node2vec_scores(train_test_split_t,P=1,Q=1,WINDOW_SIZE=10,NUM_WALKS=10,WALK_LENGTH=80,DIMENSIONS=128,DIRECTED=False,\
        WORKERS=12,ITER=5,edge_score_mode="cos",verbose=args.verbose)

elif method == 'GAE':
    # scores_s = gae_scores(train_test_split_s, dim=16, lr=0.01, epochs=200, verbose=args.verbose)
    # scores_t = gae_scores(train_test_split_t, dim=16, lr=0.01, epochs=200, verbose=args.verbose)
    scores_s = gae_scores(train_test_split_s, dim=128, lr=0.001, epochs=2000, verbose=args.verbose)
    scores_t = gae_scores(train_test_split_t, dim=128, lr=0.001, epochs=2000, verbose=args.verbose)


elif method == 'GAT':
    scores_s = gat_scores(train_test_split_s, dim=128, hidden=32, heads=8, lr=0.001, epochs=2000, verbose=args.verbose)
    scores_t = gat_scores(train_test_split_t, dim=128, hidden=32, heads=8, lr=0.001, epochs=2000, verbose=args.verbose)

elif method == 'GraphSAGE':
    scores_s = sage_scores(train_test_split_s, dim=128, lr=0.001, epochs=2000, verbose=args.verbose)
    scores_t = sage_scores(train_test_split_t, dim=128, lr=0.001, epochs=2000, verbose=args.verbose)

print("Runtime: {:.4f}".format(time() - start_time))

if args.normalize:
    scores_s = scores_s / scores_s.max()
    scores_t = scores_t / scores_t.max()

auc_s, ap_s = get_roc_score(test_edges_s, test_edges_false_s, scores_s)
auc_t, ap_t = get_roc_score(test_edges_t, test_edges_false_t, scores_t)
print(args)
print('Source network | AUC: {:.4f}, AP: {:.4f}'.format(auc_s, ap_s))
print('Target network | AUC: {:.4f}, AP: {:.4f}'.format(auc_t, ap_t))