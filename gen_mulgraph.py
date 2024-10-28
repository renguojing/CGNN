import networkx as nx
import numpy as np
import pickle
from scores import *
import argparse
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction on a single network")
    parser.add_argument('--split_dir', default='data/fb-tt/split0.9.pkl', help='Path to split dir')
    parser.add_argument('--method', default='CCPA', help='LP method')
    parser.add_argument('--normalize', default=True, type=bool)
    parser.add_argument('--layer', default=1, type=int)
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
print(args)
print('Load data...')

with open(args.split_dir, 'rb') as f:
    train_test_split0 = pickle.load(f, encoding='iso-8859-1')

train_test_split_s, train_test_split_t = train_test_split0
if args.layer == 1:
    train_test_split = train_test_split_s
elif args.layer == 2:
    train_test_split = train_test_split_t
adj, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split

g = nx.Graph(adj)

degree_sq = [d for v, d in g.degree()]
mean_k = sum(degree_sq) / len(degree_sq)
# print(degree_sq)
# g1 = nx.barabasi_albert_graph(n=len(degree_sq), m=int(mean_k / 2), seed=None, initial_graph=None)
# g1 = nx.configuration_model(deg_sequence=degree_sq, create_using=nx.MultiGraph)
g1 = nx.expected_degree_graph(degree_sq, seed=None, selfloops=False)
g1 = nx.Graph(g1)
g1.remove_edges_from(nx.selfloop_edges(g1))
print(len(g1.nodes()),len(g1.edges()))

anchor = np.arange(len(g.nodes()), dtype=int)
anchor = np.vstack((anchor, anchor)).T
# print(anchor)

print("Expanding edges...")
g, g1 = expand_edge(g, g1, anchor)

adj_train = nx.to_numpy_matrix(g, nodelist=sorted(g.nodes()))
train_test_split = adj_train, train_edges, train_edges_false, test_edges, test_edges_false

method = args.method
print(method, "for link prediction...")
start_time = time()

if method == 'CN':
    result = common_neighbors_scores(train_test_split)
elif method == 'AA':
    result = adamic_adar_scores(train_test_split)
elif method == 'JC':
    result = jaccard_coefficient_scores(train_test_split)
elif method == 'PA':
    result = preferential_attachment_scores(train_test_split)
elif method == 'RA':
    result = resource_allocation_index_scores(train_test_split)
elif method == 'CCPA':
    result = common_neighbor_centrality_scores(train_test_split)
elif method == 'SC':
    result = spectral_clustering_scores(train_test_split, d=16)
elif method == 'SVD':
    result = svd_scores(train_test_split, d=32)
elif method == 'node2vec':
    result = node2vec_scores(train_test_split,P=1,Q=1,WINDOW_SIZE=10,NUM_WALKS=10,WALK_LENGTH=80,DIMENSIONS=128,DIRECTED=False,\
        WORKERS=12,ITER=5,edge_score_mode="cos",verbose=args.verbose)
elif method == 'GAE':
    result = gae_scores(train_test_split, dim=16, lr=0.01, epochs=200, verbose=args.verbose)
elif method == 'GAT':
    result = gat_scores(train_test_split, dim=16, hidden=8, heads=4, lr=0.01, epochs=200, verbose=args.verbose)
elif method == 'GraphSAGE':
    result = sage_scores(train_test_split, dim=16, lr=0.01, epochs=200, verbose=args.verbose)

print("Runtime: {:.4f}".format(time() - start_time))

if args.normalize:
    result = result / result.max()

auc, ap = get_roc_score(test_edges, test_edges_false, result)
print('AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))