import pandas as pd
import numpy as np
from scipy import sparse
import torch
import bottleneck as bn
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader():
    def __init__(self, pro_dir='./ml-20m/processed_data/'):
        self.pro_dir = pro_dir
        self.unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                self.unique_sid.append(line.strip())

        self.n_items = len(self.unique_sid)

    def load_train_data(self, csv_file):
        print("** Load training data...")
        tp = pd.read_csv(os.path.join(self.pro_dir, csv_file))
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, self.n_items))

        # data = data.toarray()
        # zero_tokens = np.zeros([data.shape[0], 1])
        # data = np.concatenate((zero_tokens, data), 1)
        # data_sparse = torch.from_numpy(data).to_sparse().to('cpu')
        # del data

        print("** Done!")
        return data


    def load_tr_te_data(self, csv_file_tr, csv_file_te):
        print("** Load evaluation data...")
        tp_tr = pd.read_csv(os.path.join(self.pro_dir,csv_file_tr))
        tp_te = pd.read_csv(os.path.join(self.pro_dir,csv_file_te))

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))

        # data_tr = data_tr.toarray()
        # zero_tokens = np.zeros([data_tr.shape[0], 1])
        # data_tr = np.concatenate((zero_tokens, data_tr), 1)
        # data_tr_sparse = torch.from_numpy(data_tr).to_sparse().to('cpu')

        # data_te = data_te.toarray()
        # zero_tokens = np.zeros([data_te.shape[0], 1])
        # data_te = np.concatenate((zero_tokens, data_te), 1)
        # data_te_sparse = torch.from_numpy(data_te).to_sparse().to('cpu')

        # del data_tr; del data_te
        print("** Done!")
        return data_tr, data_te



def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return list(DCG / IDCG)


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return list(recall)


def NDCG_score(X_pred, X_true_sparse, k=100):    
    nusers = X_pred.shape[0]
    X_true_orig = X_true_sparse.toarray()
    
    zero_pad = np.zeros([nusers,1])
    X_true = np.concatenate((zero_pad, X_true_orig), axis=-1)

    ndcgs = []
    for u in range(nusers):
        X_pred_top_k_idx = (-X_pred[u]).argsort()[:k]
        X_true_top_k_idx = (-X_true[u]).argsort()[:k]
        
        discount = 1. / np.log2(np.arange(2, k + 2))

        DCG = (X_true[u][X_pred_top_k_idx] * discount).sum()
        IDCG = (X_true[u][X_true_top_k_idx] * discount).sum()
        
        ndcgs.append(DCG/IDCG)

    del X_true_orig, X_true

    return ndcgs


def Recall_score(X_pred, X_true_sparse, k=100):
    nusers = X_pred.shape[0]
    
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(nusers)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary_orig = (X_true_sparse > 0).toarray()
    # X_true_binary = (X_true > 0)
    false_pad = (np.zeros([nusers,1]) > 0)
    X_true_binary = np.concatenate((false_pad, X_true_binary_orig), axis=-1)

    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recalls = tmp / np.minimum(k, X_true_binary.sum(axis=1))

    del X_true_binary_orig, X_true_binary, X_pred_binary, tmp

    return recalls


