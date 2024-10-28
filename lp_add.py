import pickle as pkl
from scores import *
import argparse
import networkx as nx
import numpy as np
from lp_ee import expand_edge

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction")
    parser.add_argument('--anchor', default='data/fb-tt/anchor0.9.pkl')
    parser.add_argument('--split_dir', default='data/fb-tt/split0.9.pkl', help='Path to split dir')
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--method', default='SC', help='LP method')
    parser.add_argument('--normalize', default=True, type=bool)
    return parser.parse_args()

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

anchor_st = {x[0]: x[1] for x in train_anchor}
test_edges_all_s = np.vstack((test_edges_s, test_edges_false_s)).tolist()
# test_edges_all_s = [tuple(node_pair) for node_pair in test_edges_all_s]

anchor_ts = {x[1]: x[0] for x in train_anchor}
test_edges_all_t = np.vstack((test_edges_t, test_edges_false_t)).tolist()
# test_edges_all_t = [tuple(node_pair) for node_pair in test_edges_all_t]

# print("Expanding edges...")
# g_s, g_t = expand_edge(g_s, g_t, train_anchor)

adj_train_s = nx.to_numpy_matrix(g_s, nodelist=sorted(g_s.nodes()))
adj_train_t = nx.to_numpy_matrix(g_t, nodelist=sorted(g_t.nodes()))
train_test_split_s = adj_train_s, train_edges_s, train_edges_false_s, test_edges_s, test_edges_false_s
train_test_split_t = adj_train_t, train_edges_t, train_edges_false_t, test_edges_t, test_edges_false_t

a = args.alpha
method = args.method
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
        WORKERS=12,ITER=5,edge_score_mode="cos",verbose=0)
    scores_t = node2vec_scores(train_test_split_t,P=1,Q=1,WINDOW_SIZE=10,NUM_WALKS=10,WALK_LENGTH=80,DIMENSIONS=128,DIRECTED=False,\
        WORKERS=12,ITER=5,edge_score_mode="cos",verbose=0)

elif method == 'GAE':
    scores_s = gae_scores(train_test_split_s, dim=16, lr=0.01, epochs=200, verbose=2)
    scores_t = gae_scores(train_test_split_t, dim=16, lr=0.01, epochs=200, verbose=2)

elif method == 'GAT':
    scores_s = gat_scores(train_test_split_s, dim=16, heads=1, lr=0.01, epochs=200, verbose=2)
    scores_t = gat_scores(train_test_split_t, dim=16, heads=1, lr=0.01, epochs=200, verbose=2)

elif method == 'GraphSAGE':
    scores_s = sage_scores(train_test_split_s, dim=16, lr=0.01, epochs=200, verbose=2)
    scores_t = sage_scores(train_test_split_t, dim=16, lr=0.01, epochs=200, verbose=2)

print("Results aggregation...")
for edge in test_edges_all_s:
    if edge[0] in anchor_st.keys() and edge[1] in anchor_st.keys():
        scores_s[edge[0]][edge[1]] = a *  scores_s[edge[0]][edge[1]] + (1 - a) * scores_t[anchor_st[edge[0]]][anchor_st[edge[1]]]

for edge in test_edges_all_t:
    if edge[0] in anchor_ts.keys() and edge[1] in anchor_ts.keys():
        scores_t[edge[0]][edge[1]] = a *  scores_t[edge[0]][edge[1]] + (1 - a) * scores_s[anchor_ts[edge[0]]][anchor_ts[edge[1]]]

if args.normalize:
    scores_s = scores_s / scores_s.max()
    scores_t = scores_t / scores_t.max()

auc_s, ap_s = get_roc_score(test_edges_s, test_edges_false_s, scores_s)
auc_t, ap_t = get_roc_score(test_edges_t, test_edges_false_t, scores_t)
print(args)
print('Source network | AUC: {:.4f}, AP: {:.4f}'.format(auc_s, ap_s))
print('Target network | AUC: {:.4f}, AP: {:.4f}'.format(auc_t, ap_t))