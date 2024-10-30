import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def greedy_match(S):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:, :k]
    #top = np.argsort()[:, ::-1]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
    return result

def get_gt_matrix(path,shape):
    gt = np.zeros(shape)
    with open(path) as file:
        for line in file:
            s, t = line.strip().split()
            gt[int(s),int(t)] = 1
    return gt

def get_statistics(alignment_matrix, groundtruth_matrix):
    results = {}
    pred = greedy_match(alignment_matrix)
    greedy_match_acc = compute_accuracy(pred, groundtruth_matrix)
    results['Acc'] = greedy_match_acc
    # MAP = compute_MAP(alignment_matrix, groundtruth_matrix)
    # print("MAP: %.4f" % MAP)

    MAP, AUC, Hit = compute_MAP_AUC_Hit(alignment_matrix, groundtruth_matrix)
    results['MRR'] = MAP
    results['AUC'] = AUC
    results['Hit'] = Hit

    pred_top_1 = top_k(alignment_matrix, 1)
    precision_1 = compute_precision_k(pred_top_1, groundtruth_matrix)

    pred_top_5 = top_k(alignment_matrix, 5)
    precision_5 = compute_precision_k(pred_top_5, groundtruth_matrix)

    pred_top_10 = top_k(alignment_matrix, 10)
    precision_10 = compute_precision_k(pred_top_10, groundtruth_matrix)

    pred_top_15 = top_k(alignment_matrix, 15)
    precision_15 = compute_precision_k(pred_top_15, groundtruth_matrix)

    pred_top_20 = top_k(alignment_matrix, 20)
    precision_20 = compute_precision_k(pred_top_20, groundtruth_matrix)

    pred_top_25 = top_k(alignment_matrix, 25)
    precision_25 = compute_precision_k(pred_top_25, groundtruth_matrix)
    
    pred_top_30 = top_k(alignment_matrix, 30)
    precision_30 = compute_precision_k(pred_top_30, groundtruth_matrix)

    results['Precision@1'] = precision_1
    results['Precision@5'] = precision_5
    results['Precision@10'] = precision_10
    results['Precision@15'] = precision_15
    results['Precision@20'] = precision_20
    results['Precision@25'] = precision_25
    results['Precision@30'] = precision_30
    return results

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0
    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_accuracy(greedy_matched, gt):
    # print(gt)
    n_matched = 0
    for i in range(greedy_matched.shape[0]):
        if greedy_matched[i].sum() > 0 and np.array_equal(greedy_matched[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    print("True matched nodes: " + str(n_matched))
    print("Total test nodes: " + str(n_nodes))
    return n_matched/n_nodes

def compute_MAP(alignment_matrix, gt):
    S_argsort = np.argsort(-alignment_matrix)#对齐矩阵S降序排列的下标
    gt_candidates = np.argmax(gt, axis=1)#真实锚连接下标
    MAP = 0
    for i in range(len(S_argsort)):
        predicted_source_to_target = S_argsort[i]
        if gt[i][gt_candidates[i]] == 1:
            for k in range(len(predicted_source_to_target)):
                if predicted_source_to_target[k] == gt_candidates[i]:
                    ra = k + 1
                    MAP += 1/ra
                    break
    n_nodes = (gt==1).sum()
    MAP /= n_nodes
    return MAP

def compute_MAP_AUC_Hit(alignment_matrix, gt):
    S_argsort = alignment_matrix.argsort(axis=1)[:, ::-1]
    m = gt.shape[1] - 1
    MAP = 0
    AUC = 0
    Hit = 0
    for i in range(len(S_argsort)):
        predicted_source_to_target = S_argsort[i]
        # true_source_to_target = gt[i]
        for j in range(gt.shape[1]):
            if gt[i, j] == 1:
                for k in range(len(predicted_source_to_target)):
                    if predicted_source_to_target[k] == j:
                        ra = k + 1
                        MAP += 1/ra
                        AUC += (m+1-ra)/m
                        Hit += (m+2-ra)/(m+1)
                        break
                break
    n_nodes = (gt==1).sum()
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, AUC, Hit