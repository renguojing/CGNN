import torch
import numpy as np
from models.CGNN import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import argparse
import networkx as nx
from utils import *
# from torch_sparse import SparseTensor
# import random
# import os
import pickle as pkl
# from math import sqrt, log
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="CGNN")
    # parser.add_argument('--s_graph', default='data/fb-tt/graph1.pkl')
    # parser.add_argument('--t_graph', default='data/fb-tt/graph2.pkl')
    parser.add_argument('--anchor', default='data/fb-tt/anchor0.9.pkl')
    parser.add_argument('--split_path', default='data/fb-tt/split0.9.pkl')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--pre_epochs', default=20, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--margin', default=0.9, type=float)
    parser.add_argument('--neg', default=1, type=int)
    parser.add_argument('--verbose', default=2, type=int)
    parser.add_argument('--expand_edges', default=False, type=bool)
    parser.add_argument('--edge_type', default='gcn', type=str)
    parser.add_argument('--nodeattr', default='svd', type=str)
    return parser.parse_args()

def LR_scores(emb_matrix, train_test_split):
    def get_edge_embeddings(edge_list, g):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)# Hadamard product 
                # print(edge_emb.shape)
                # pa_preds = nx.preferential_attachment(g, ebunch=[(node1, node2)])
                # for u,v,p in pa_preds:
                #     pa = p
                # aa_preds = nx.adamic_adar_index(g, ebunch=[(node1, node2)])
                # for u,v,p in aa_preds:
                #     aa = p
                edge_emb = np.concatenate((emb1, edge_emb, emb2),axis=-1)
                # print(edge_emb.shape)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split
    g = nx.Graph(adj_train)
    # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(train_edges, g)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false, g)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges, g)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false, g)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(solver='liblinear', random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    # Calculate scores
    test_auc = roc_auc_score(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)
    return test_auc, test_ap

if __name__ == "__main__":
    # lp_results = dict.fromkeys(('AUC', 'AP', 'runtime'), 0) # save lp results
    AUC_list_s = []
    AP_list_s = []
    Runtime_list = []
    AUC_list_t = []
    AP_list_t = []

    AUC_list_s1 = []
    AP_list_s1 = []
    AUC_list_t1 = []
    AP_list_t1 = []
    
    N = 10 # repeat times for average, default: 1
    for i in range(N):
        args = parse_args()

        # def set_seed(seed):
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     os.environ['PYTHONHASHSEED'] = str(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        # seed = 42
        # set_seed(seed)

        print('Load data...')
        with open(args.split_path, 'rb') as f:
            train_test_split0 = pkl.load(f, encoding='iso-8859-1')

        train_test_split_s, train_test_split_t = train_test_split0

        adj_train_s, train_edges_s, train_edges_false_s, test_edges_s, test_edges_false_s = train_test_split_s
        adj_train_t, train_edges_t, train_edges_false_t, test_edges_t, test_edges_false_t = train_test_split_t

        g_s = nx.Graph(adj_train_s)
        g_t = nx.Graph(adj_train_t)
  
        print('source nodes: %d\tsource edges: %d'%(len(g_s.nodes()), len(g_s.edges())))
        print('target nodes: %d\ttarget edges: %d'%(len(g_t.nodes()), len(g_t.edges())))

        # load train and test anchor links
        train_anchor, test_anchor = pkl.load(file=open(args.anchor, 'rb'), encoding='iso-8859-1')
        print('train anchors: %d\ttest anchors: %d'%(len(train_anchor), len(test_anchor)))
        train_anchor = torch.LongTensor(train_anchor)
        # generate test anchor matrix for evaluation
        s_num = len(g_s.nodes())
        t_num = len(g_t.nodes())
        groundtruth_matrix = get_gt_matrix(test_anchor, (s_num, t_num))

        start_time = time()

        # generate edge_index(pyG version)
        s_e = get_edgeindex(np.array(g_s.edges()))
        t_e = get_edgeindex(np.array(g_t.edges()))

        if args.edge_type == 'gcn':
            s_w = None
            t_w = None

        elif args.edge_type == 'pa1':
            degree_s = np.array([d for (u,d) in g_s.degree()])
            degree_t = np.array([d for (u,d) in g_t.degree()])
            s_w = np.sqrt(degree_s[s_e[0]] * degree_s[s_e[1]])
            # s_k = len(s_w) / s_num
            s_w = torch.FloatTensor(s_w)
            t_w = []
            t_w = np.sqrt(degree_t[t_e[0]] * degree_t[t_e[1]])
            # t_k = len(t_w) / t_num
            t_w = torch.FloatTensor(t_w)

        elif args.edge_type == 'pa2':
            degree_s = np.array([d for (u,d) in g_s.degree()])
            degree_t = np.array([d for (u,d) in g_t.degree()])
            s_w = []
            s_w = np.log(1 + degree_s[s_e[0]] * degree_s[s_e[1]])
            # s_k = len(s_w) / s_num
            s_w = torch.FloatTensor(s_w)
            t_w = []
            t_w = np.log(1 + degree_t[t_e[0]] * degree_t[t_e[1]])
            # t_k = len(t_w) / t_num
            t_w = torch.FloatTensor(t_w)

        if args.expand_edges:
            print('Expand edges...')
            g_s, g_t, s_e, t_e = expand_edges(g_s, g_t, train_anchor, s_e, t_e)

        s_adj = nx.to_numpy_matrix(g_s, nodelist=sorted(g_s.nodes()))
        t_adj = nx.to_numpy_matrix(g_t, nodelist=sorted(g_t.nodes()))
        # feature vectors
        if args.nodeattr == 'svd':
            print('SVD...')
            s_x = get_svd(s_adj, d=32)
            t_x = get_svd(t_adj, d=32)
            s_x = torch.FloatTensor(s_x)
            t_x = torch.FloatTensor(t_x)
        elif args.nodeattr == 'adj':
            print('Adjacency...')
            s_x = torch.FloatTensor(s_adj)
            t_x = torch.FloatTensor(t_adj)
        # s_x = torch.arange(s_num, dtype=torch.int32)
        # t_x = torch.arange(t_num, dtype=torch.int32)

        print('Generate embeddings...')
        s_embedding, t_embedding= get_embedding(s_x, t_x, s_e, t_e, s_w, t_w, g_s, g_t, train_anchor, test_edges_s, test_edges_false_s,\
            test_edges_t, test_edges_false_t, dim=args.dim,\
            lr=args.lr, pre_epochs=args.pre_epochs, epochs=args.epochs, alpha=args.alpha, margin=args.margin, neg=args.neg, verbose=args.verbose)

        t = time() - start_time
        print('Finished in %.4f s!'%(t))

        # LP
        print('Evaluating link prediction...')
        print("cos")
        s_score_matrix = cosine_similarity(s_embedding, s_embedding)
        t_score_matrix = cosine_similarity(t_embedding, t_embedding)
        
        s_test_roc, s_test_ap = get_roc_score(test_edges_s, test_edges_false_s, s_score_matrix)
        t_test_roc, t_test_ap = get_roc_score(test_edges_t, test_edges_false_t, t_score_matrix)

        print('Source network | AUC: {:.4f}, AP: {:.4f}'.format(s_test_roc, s_test_ap))
        print('Target network | AUC: {:.4f}, AP: {:.4f}'.format(t_test_roc, t_test_ap))

        AUC_list_s.append(s_test_roc)
        AP_list_s.append(s_test_ap)
        AUC_list_t.append(t_test_roc)
        AP_list_t.append(t_test_ap)
        Runtime_list.append(t)

        print("LR")
        s_embedding = F.normalize(s_embedding, p=2., dim=-1)
        t_embedding = F.normalize(t_embedding, p=2., dim=-1)
        s_embedding = s_embedding.numpy()
        t_embedding = t_embedding.numpy()
        s_test_roc, s_test_ap = LR_scores(s_embedding, train_test_split_s)
        t_test_roc, t_test_ap = LR_scores(t_embedding, train_test_split_t)

        print('Source network | AUC: {:.4f}, AP: {:.4f}'.format(s_test_roc, s_test_ap))
        print('Target network | AUC: {:.4f}, AP: {:.4f}'.format(t_test_roc, t_test_ap))

        AUC_list_s1.append(s_test_roc)
        AP_list_s1.append(s_test_ap)
        AUC_list_t1.append(t_test_roc)
        AP_list_t1.append(t_test_ap)


    AUC_list_s = np.array(AUC_list_s) * 100
    AP_list_s = np.array(AP_list_s) * 100
    AUC_list_t = np.array(AUC_list_t) * 100
    AP_list_t = np.array(AP_list_t) * 100
    AUC_list_s1 = np.array(AUC_list_s1) * 100
    AP_list_s1 = np.array(AP_list_s1) * 100
    AUC_list_t1 = np.array(AUC_list_t1) * 100
    AP_list_t1 = np.array(AP_list_t1)  * 100   
    Runtime_list = np.array(Runtime_list)

    print('\nCGCN')
    print(args)
    print("cos")
    print("Source network:")
    print('List of AUC: ', AUC_list_s)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AUC_list_s), np.std(AUC_list_s), np.max(AUC_list_s), np.min(AUC_list_s)))
    print('List of AP: ', AP_list_s)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AP_list_s), np.std(AP_list_s), np.max(AP_list_s), np.min(AP_list_s)))

    print("Target network:")
    print('List of AUC: ', AUC_list_t)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AUC_list_t), np.std(AUC_list_t), np.max(AUC_list_t), np.min(AUC_list_t)))
    print('List of AP: ', AP_list_t)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AP_list_t), np.std(AP_list_t), np.max(AP_list_t), np.min(AP_list_t)))

    print("LR")
    print("Source network:")
    print('List of AUC: ', AUC_list_s1)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AUC_list_s1), np.std(AUC_list_s1), np.max(AUC_list_s1), np.min(AUC_list_s1)))
    print('List of AP: ', AP_list_s1)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AP_list_s1), np.std(AP_list_s1), np.max(AP_list_s1), np.min(AP_list_s1)))

    print("Target network:")
    print('List of AUC: ', AUC_list_t1)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AUC_list_t1), np.std(AUC_list_t1), np.max(AUC_list_t1), np.min(AUC_list_t1)))
    print('List of AP: ', AP_list_t1)
    print('Mean: {:.2f}\tStd: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(np.mean(AP_list_t1), np.std(AP_list_t1), np.max(AP_list_t1), np.min(AP_list_t1)))

    print('List of runtime: ', Runtime_list)
    print('Mean: {:.4f}\tMax: {:.4f}\tMin: {:.4f}'.format(np.mean(Runtime_list), np.max(Runtime_list), np.min(Runtime_list)))