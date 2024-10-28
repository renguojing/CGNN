import pickle
from scores import *
import argparse
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction on a single network")
    parser.add_argument('--split_dir', default='data/fb-tt/split0.9.pkl', help='Path to split dir')
    parser.add_argument('--method', default='GAE', help='LP method')
    parser.add_argument('--normalize', default=False, type=bool)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    return parser.parse_args()

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
adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split

# print("Training edges (positive):", len(train_edges))
# print("Training edges (negative):", len(train_edges_false))
# print("Test edges (positive):", len(test_edges))
# print("Test edges (negative):", len(test_edges_false))

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