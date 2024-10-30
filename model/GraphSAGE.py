import torch
import torch_geometric
from torch_geometric.nn import SAGEConv
# from mulconv import MulConv as  GCNConv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score

class SAGEEncoder(torch.nn.Module):
    """GCN组成的编码器"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=2 * out_channels)
        self.conv2 = SAGEConv(in_channels=2 * out_channels, out_channels=out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class InnerProductDecoder(torch.nn.Module):
    """解码器，用向量内积表示重建的图结构"""

    def forward(self, z, edge_index, sigmoid=True):
        """
        参数说明：
        z: 节点表示
        edge_index: 边索引，也就是节点对
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class GAE(torch.nn.Module):
    """图自编码器。
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder()

    def encode(self, *args, **kwargs):
        """编码功能"""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """解码功能"""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """计算正边和负边的二值交叉熵

        参数说明
        ----
        z: 编码器的输出
        pos_edge_index: 正边的边索引
        neg_edge_index: 负边的边索引
        """
        EPS = 1e-15  # EPS是一个很小的值，防止取对数的时候出现0值

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()  # 正样本的损失函数

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0))  # 负采样
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean()  # 负样本的损失函数

        return pos_loss + neg_loss

def get_embedding(x, edge_index, dim=128, lr=0.001, epochs=1000, verbose=1, test_edges=None, test_edges_false=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_index = edge_index.to(device)
    input = x.shape[1]
    model = GAE(SAGEEncoder(input, dim))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 and verbose > 0:
            if verbose == 1:
                print('Epoch: {:03d}, loss_train: {:.8f}'.format(epoch, loss))
            else:
                auc = evaluate(z, test_edges, test_edges_false)
                print('Epoch: {:03d}, loss_train: {:.8f}, auc: {:.8f}'.format(epoch, loss, auc))
    model.eval()
    embedding=model.encode(x, edge_index)
    embedding=embedding.detach().cpu()
    return embedding

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
    preds_all = np.hstack([preds_pos, preds_neg])  # 按水平方向拼接数组
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    
    # return roc_score
    return roc_score