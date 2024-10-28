import numpy as np
import networkx as nx
from train_test_splits import mask_test_edges
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Split edgelist to train and test set.")
    parser.add_argument('--input', default='./data/douban/online/raw/edgelist', help='Path to edgelist')
    parser.add_argument('--split_dir', default='./data/douban/online/processed/split0.1.pkl', help='Path to split dir')
    parser.add_argument('--train_dir', default='./data/douban/online/raw/train0.1_edgelist', help='Path to train dir')
    parser.add_argument('--split', type=float, default=0.1, help='Train/test split')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    #打开文件生成邻接矩阵
    input = args.input
    f = open(input,'r')
    g = nx.read_edgelist(f, nodetype=int)
    print(nx.number_connected_components(g))
    # wcc=max(nx.connected_components(g))#最大连通子图
    # adj = nx.adjacency_matrix(g.subgraph(wcc))
    adj = nx.adjacency_matrix(g)
    print(adj.shape)

    #隐藏部分边，生成训练图、训练正例/负例、验证正例/负例、测试正例/负例
    split = args.split

    split_dir = args.split_dir
    train_test_split = mask_test_edges(adj, test_frac=split,verbose=True)
    with open(split_dir, 'wb') as f:
        pickle.dump(train_test_split, f, protocol=2)

    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split
    print(adj_train.shape, train_edges.shape[0], test_edges.shape[0])

    train_edges_dir = args.train_dir
    np.savetxt(train_edges_dir, train_edges, fmt='%d')

