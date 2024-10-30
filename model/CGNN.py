import torch
import numpy as np
import torch.nn as nn
# import torch_geometric
# from torch_geometric.nn import GCNConv
from models.mulconv import MulConv as GCNConv
from sklearn.metrics.pairwise import cosine_similarity
# from models.metrics import top_k, compute_precision_k
import networkx as nx
from utils import *
from sklearn.metrics import roc_auc_score

# from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

EPS = 1e-15  # avoid zero when calculating logarithm

class CrossModel(torch.nn.Module):
    def __init__(self, s_input, t_input, output):
        super().__init__()
        self.conv1 = GCNConv(s_input, 2 * output)
        self.conv2 = GCNConv(t_input, 2 * output)
        self.conv3 = GCNConv(2 * output, output)
        # self.emb1 = nn.Embedding(s_input,2 * output)
        # self.emb2 = nn.Embedding(t_input,2 * output)
        self.activation = nn.ReLU()
        
    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2, seeds):
        '''
        embeddings of source network g_s(V_s, E_s):
        x is node feature vectors of g_s, with shape [V_s, n_feats], V_s is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_s], E_s is the number of edges
        '''
        # x1_seed, x2_seed = self.inter_propagate(x1, x2, seeds)
        # x1 = self.emb1(x1)
        # x2 = self.emb2(x2)
        x1 = self.conv1(x1, edge_index1, edge_weight=edge_weight1)
        x2 = self.conv2(x2, edge_index2, edge_weight=edge_weight2)
        x1 = self.activation(x1)
        x2 = self.activation(x2)
        x1_seed, x2_seed = self.inter_propagate(x1, x2, seeds)
        x1 = self.conv3(x1, edge_index1, x2_seed, edge_weight=edge_weight1)
        x2 = self.conv3(x2, edge_index2, x1_seed, edge_weight=edge_weight2)
        return x1, x2

    def pre_forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        '''
        embeddings of source network g_s(V_s, E_s):
        x is node feature vectors of g_s, with shape [V_s, n_feats], V_s is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_s], E_s is the number of edges
        '''
        x1 = self.conv1(x1, edge_index1, edge_weight=edge_weight1)
        x2 = self.conv2(x2, edge_index2, edge_weight=edge_weight2)
        x1 = self.activation(x1)
        x2 = self.activation(x2)
        x1 = self.conv3(x1, edge_index1, edge_weight=edge_weight1)
        x2 = self.conv3(x2, edge_index2, edge_weight=edge_weight2)
        return x1, x2

    def inter_propagate(self, x1, x2, seeds):
        x1_mask = torch.zeros_like(x2)
        x2_mask = torch.zeros_like(x1)
        x1_mask[seeds[1]] = x1[seeds[0]]
        x2_mask[seeds[0]] = x2[seeds[1]]
        return x1_mask, x2_mask

class RWLoss(torch.nn.Module):
    def __init__(self, edge_index, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False):
        super().__init__()

        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        self.N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(self.N, self.N))
        self.adj = self.adj.to('cpu')

        assert walk_length >= context_size

        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
    
    def loader(self, batch_size):
        # return DataLoader(range(self.adj.sparse_size(0)),
        #                   collate_fn=self.sample, **kwargs)
        batch = np.random.choice(self.N, size=batch_size, replace=False, p=None).astype(np.int64)
        return self.sample(batch)

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.adj.sparse_size(0),
                           (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def forward(self, pos_rw, neg_rw, embedding, dim):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = embedding[start].view(pos_rw.size(0), 1, dim)
        h_rest = embedding[rest.view(-1)].view(pos_rw.size(0), -1, dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = embedding[start].view(neg_rw.size(0), 1, dim)
        h_rest = embedding[rest.view(-1)].view(neg_rw.size(0), -1, dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss


def get_embedding(s_x, t_x, s_e, t_e, s_w, t_w, g_s, g_t, anchor, test_edges_s, test_edges_false_s, test_edges_t, test_edges_false_t, \
        dim=128, lr=0.001, alpha=0.5, margin=0.8, neg=1, pre_epochs=20, epochs=1000, verbose=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s_x = s_x.to(device)
    t_x = t_x.to(device)
    s_e = s_e.to(device)
    t_e = t_e.to(device)
    if s_w is not None:
        s_w = s_w.to(device)
        t_w = t_w.to(device)
    seeds = anchor.T.to(device)
    s_input = s_x.shape[-1]
    t_input = t_x.shape[-1]
    model = CrossModel(s_input, t_input, dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    cosine_loss=nn.CosineEmbeddingLoss(margin=margin)
    in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg) # hard negative sampling

    rw_loss1 = RWLoss(edge_index=s_e, walk_length=20, context_size=10, walks_per_node=10, p=1, q=1, num_nodes=s_x.shape[0]).to(device)
    rw_loss2 = RWLoss(edge_index=t_e, walk_length=20, context_size=10, walks_per_node=10, p=1, q=1, num_nodes=t_x.shape[0]).to(device)
    
    for epoch in range(pre_epochs):
        model.train()
        optimizer.zero_grad()
        pw1, nw1 = rw_loss1.loader(batch_size=512)
        pw2, nw2 = rw_loss2.loader(batch_size=512)
    
        zs, zt = model.pre_forward(s_x, s_e, s_w, t_x, t_e, t_w)
        loss = rw_loss1.forward(pw1, nw1, zs, dim) + rw_loss2.forward(pw2, nw2, zt, dim)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if verbose > 0 and epoch % 1 == 0:
            if verbose == 1:
                print('Epoch: {:03d}, loss_train: {:.8f}'.format(epoch, loss))
            elif verbose == 2:
                auc_s = evaluate(zs, test_edges_s, test_edges_false_s)
                auc_t = evaluate(zt, test_edges_t, test_edges_false_t)
                print('Epoch: {:03d}, loss_train: {:.8f}, auc_s: {:.8f},auc_t: {:.8f},'.format(epoch, loss, auc_s, auc_t))

    for epoch in range(epochs - pre_epochs):
        model.train()
        optimizer.zero_grad()
        pw1, nw1 = rw_loss1.loader(batch_size=512)
        pw2, nw2 = rw_loss2.loader(batch_size=512)
    
        zs, zt = model.forward(s_x, s_e, s_w, t_x, t_e, t_w, seeds)
        intra_loss = rw_loss1.forward(pw1, nw1, zs, dim) + rw_loss2.forward(pw2, nw2, zt, dim)
        anchor_label = anchor_label.view(-1).to(device)
        inter_loss = cosine_loss(zs[in_a], zt[in_b], anchor_label)
        loss = alpha * intra_loss + (1 - alpha) * inter_loss
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if verbose > 0 and epoch % 1 == 0:
            if verbose == 1:
                print('Epoch: {:03d}, intra_loss: {:.8f}, inter_loss: {:.8f}, loss_train: {:.8f}'.format(epoch + pre_epochs,\
                intra_loss, inter_loss, loss))
            elif verbose == 2:
                auc_s = evaluate(zs, test_edges_s, test_edges_false_s)
                auc_t = evaluate(zt, test_edges_t, test_edges_false_t)
                print('Epoch: {:03d}, intra_loss: {:.8f}, inter_loss: {:.8f},loss_train: {:.8f}, auc_s: {:.8f},auc_t: {:.8f},'.format(epoch + pre_epochs,\
                    intra_loss, inter_loss, loss, auc_s, auc_t))
    model.eval()
    zs, zt = model.forward(s_x, s_e, s_w, t_x, t_e, t_w, seeds)
    s_embedding = zs.detach().cpu()
    t_embedding = zt.detach().cpu()
    return s_embedding, t_embedding

@torch.no_grad()
def evaluate(zt, test_edges, test_edges_false):
    '''
    calculate AUC of target network for evaluation
    '''
    zt = zt.detach().cpu()
    t_score_matrix = cosine_similarity(zt, zt)
    t_test_roc = get_roc_score(test_edges, test_edges_false, t_score_matrix)
    return t_test_roc

def get_roc_score(edges_pos, edges_neg, score_matrix):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return None

    # Store positive edge predictions, actual values
    preds_pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]])

    # Store negative edge predictions, actual values
    preds_neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]])

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    
    # return roc_score
    return roc_score

def sample(anchor_train, gs, gt, neg=1, seed=None):
    '''
    sample non-anchors for each anchor
    '''
    triplet_neg = neg  # number of non-anchors for each anchor, when neg=1, there are two negtives for each anchor
    anchor_flag = 1
    anchor_train_len = anchor_train.shape[0]
    anchor_train_a_list = np.array(anchor_train.T[0])
    anchor_train_b_list = np.array(anchor_train.T[1])
    input_a = []
    input_b = []
    classifier_target = torch.empty(0)
    index = 0
    while index < anchor_train_len:
        a = anchor_train_a_list[index]
        b = anchor_train_b_list[index]
        input_a.append(a)
        input_b.append(b)
        an_target = torch.ones(anchor_flag)
        classifier_target = torch.cat((classifier_target, an_target), dim=0)
        # an_negs_index = list(set(node_t) - {b}) # all nodes except anchor node
        an_negs_index = list(gt.neighbors(b)) # neighbors of each anchor node
        np.random.seed(seed)
        an_negs_index_sampled = list(np.random.choice(an_negs_index, triplet_neg, replace=True)) # randomly sample negatives
        an_as = triplet_neg * [a]
        input_a += an_as
        input_b += an_negs_index_sampled

        # an_negs_index1 = list(set(node_f) - {a})
        an_negs_index1 = list(gs.neighbors(a))
        np.random.seed(seed)
        an_negs_index_sampled1 = list(np.random.choice(an_negs_index1, triplet_neg, replace=True))
        an_as1 = triplet_neg * [b]
        input_b += an_as1
        input_a += an_negs_index_sampled1

        un_an_target = torch.zeros(triplet_neg * 2)
        classifier_target = torch.cat((classifier_target, un_an_target), dim=0)
        index += 1

    cosine_target = torch.unsqueeze(2 * classifier_target - 1, dim=1)  # labels are [1,-1,-1]
    # classifier_target = torch.unsqueeze(classifier_target, dim=1)  # labels are [1,0,0]

    # [ina, inb] is all anchors and sampled non-anchors, cosine_target is their labels
    ina = torch.LongTensor(input_a)
    inb = torch.LongTensor(input_b)

    return ina, inb, cosine_target