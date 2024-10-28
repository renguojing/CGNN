import numpy as np
import argparse
import networkx as nx
import pickle as pkl
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="split edge and anchor")
    parser.add_argument('--s_edge', default='data/douban-weibo/graph1.pkl')
    parser.add_argument('--t_edge', default='data/douban-weibo/graph2.pkl')
    # parser.add_argument('--anchor', default='data/fb-tt/groundtruth')
    # parser.add_argument('--test_anchor', default='data/dblp1/ind.dblp01.anchors.test')
    parser.add_argument('--out_dir', default='data/douban-weibo')
    parser.add_argument('--edge_split', default=0.8, type=float)
    # parser.add_argument('--anchor_split', default=0.1, type=float)
    return parser.parse_args()

args = parse_args()
print(args)
print('Load data...')
# g_s = nx.read_edgelist(args.s_edge, nodetype=int, delimiter='\t')
# g_t = nx.read_edgelist(args.t_edge, nodetype=int, delimiter='\t')
# pkl.dump(g_s, file=open(args.out_dir + '/graph1.pkl', 'wb'))
# pkl.dump(g_t, file=open(args.out_dir + '/graph2.pkl', 'wb'))
g_s = pkl.load(file=open(args.s_edge, 'rb'), encoding='iso-8859-1')
g_t = pkl.load(file=open(args.t_edge, 'rb'), encoding='iso-8859-1')
# print('source nodes: %d\tsource edges: %d'%(len(g_s.nodes()), len(g_s.edges())))
# print('target nodes: %d\ttarget edges: %d'%(len(g_t.nodes()), len(g_t.edges())))

if args.edge_split is not None:
    print('split edges with train ratio %.2f'%args.edge_split)
    train_test_split_s = mask_test_edges(g_s, test_frac=1-args.edge_split)
    train_test_split_t = mask_test_edges(g_t, test_frac=1-args.edge_split)
    train_test_split = train_test_split_s, train_test_split_t
    pkl.dump(train_test_split, file=open(args.out_dir + '/split%.1f.pkl'%args.edge_split, 'wb'))

# anchor = np.loadtxt(args.anchor, dtype=int, delimiter='\t')
# test_anchor = np.loadtxt(args.test_anchor, dtype=int, delimiter='\t')
# anchor = np.vstack((anchor, test_anchor))
# np.savetxt(args.out_dir + '/groundtruth', anchor, fmt='%d', delimiter=' ')
# anchor = np.loadtxt(args.anchor, dtype=int, delimiter=' ')
# np.random.shuffle(anchor)
# train_num = int(len(anchor) * args.anchor_split)
# train_anchor = anchor[:train_num]
# test_anchor = anchor[train_num:]
# pkl.dump((train_anchor, test_anchor), file=open(args.out_dir + '/anchor%.1f.pkl'%args.anchor_split, 'wb'))
# print('train anchors: %d\ttest anchors: %d'%(len(train_anchor), len(test_anchor)))