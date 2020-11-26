# import torchtext
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformer import TransformerCF
import time
import math
import os
from tqdm import tqdm

USE_WANDB = True

if USE_WANDB:
    import wandb
    wandb.init(project="transformer-cf")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from util import DataLoader, NDCG_score, Recall_score

def get_train_batch(data, user_idx, seq_len=100, item_sample_rate=0.5):
    sources = []
    targets = []
    
    batch_size = len(user_idx)
    zero_tokens = torch.zeros([batch_size, 1]).to(device)
    batches = torch.Tensor(data[user_idx].toarray()).to(device)     # [batch_size, ntokens-1]
    batches = torch.cat((zero_tokens, batches), 1)                  # [batch_size, ntokens]
    # batches = data[user_idx]
    # masked_batches = torch.clone(batches)
    # choose movie_seen and movie_to_be_seen (padding if input_size > num_movie_seen)

    for i in range(batch_size):        
        items = batches[i].nonzero().flatten()
        item_num = len(items)
        sample_num = int(item_sample_rate * len(items))
        
        idxs = np.random.permutation(item_num)
        source_idx = idxs[:sample_num]        
        
        source = items[source_idx][:seq_len]
        # masked_batches[i][source] = 0

        if sample_num < seq_len:
            source = F.pad(source, pad=[0, seq_len - sample_num])

        sources.append(source)    
    
    targets = batches                                          # [batch_size, ntokens]
    # masked_targets = masked_batches
    sources = torch.stack(sources).permute((1,0))              # [seq_len, batch_size]
    
    return sources, targets #, masked_targets

def get_eval_batch(data_tr, user_idx, seq_len=100):
    sources = []
    targets = []
    nonzeroidxs = []
    
    batch_size = len(user_idx)
    zero_tokens = torch.zeros([batch_size, 1]).to(device)
    batches_tr = torch.Tensor(data_tr[user_idx].toarray()).to(device)     # [batch_size, ntokens-1]
    batches_tr = torch.cat((zero_tokens, batches_tr), 1)                  # [batch_size, ntokens]
    # batches_tr = data_tr[user_idx]
    # batches_te = torch.Tensor(data_te[user_idx].toarray()).to(device)     # [batch_size, ntokens-1]
    # batches_te = torch.cat((zero_tokens, batches_te), 1)                  # [batch_size, ntokens]

    for i in range(batch_size):        
        source = batches_tr[i].nonzero().flatten()
        item_num = len(source)
        # sample_num = int(item_sample_rate * len(items))        
        # idxs = np.random.permutation(item_num)
        # source_idx = idxs[:sample_num]
        # target_idx = idxs[sample_num:]

        if item_num > seq_len:            
            idxs = np.random.permutation(item_num)
            source_idx = idxs[:seq_len]
            source = source[source_idx]
        
        nonzeroidxs.append(source)

        if item_num < seq_len:
            source = F.pad(source, pad=[0, seq_len - item_num])

        sources.append(source)
        
    targets = batches_tr                                          # [batch_size, ntokens]
    sources = torch.stack(sources).permute((1,0))                 # [seq_len, batch_size]
    
    return sources, targets, nonzeroidxs


def train(model, n_train, batch_size, nseq, train_data, epoch, criterion, optimizer, scheduler):
    model.train() 
    total_loss = 0.
    start_time = time.time()
    idxlist = np.arange(n_train)
    np.random.shuffle(idxlist)
    # ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, n_train, batch_size)):
        useridx = idxlist[i:(i+batch_size)]
        # chunk_size = len(useridx)

        data, targets = get_train_batch(train_data, useridx, seq_len=nseq)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | ppl {:8.2f}'.format(
                    epoch, batch, train_data.shape[0] // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()

        del data, targets
    del idxlist


def evaluate(model, num_data, eval_batch_size, nseq, eval_data_tr, eval_data_te, criterion, mode='valid'):
    # model.train() 
    total_loss = 0.

    idxlist = np.arange(num_data)
    np.random.shuffle(idxlist)

    ndcgs100 = []
    ndcgs50 = []
    recalls100 = []
    recalls50 = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_data, eval_batch_size), desc='Validating..'):
            useridx = idxlist[i : (i+eval_batch_size)]            

            data, targets, nonzeroidxs = get_eval_batch(eval_data_tr, useridx, seq_len=nseq)
            output = model(data)            
            
            total_loss += len(data) * criterion(output, targets).item()
            pred = (output > 0.5).float().cpu()
            
            for i in range(len(pred)):
                pred[i][nonzeroidxs[i]] = 0.

            ndcg100 = NDCG_score(pred, eval_data_te[useridx], k=100)
            ndcg50 = NDCG_score(pred, eval_data_te[useridx], k=50)
            # recall100 = Recall_score(pred, eval_data_te[useridx], k=100)
            # recall50 = Recall_score(pred, eval_data_te[useridx], k=50)

            ndcgs100.extend(ndcg100)
            ndcgs50.extend(ndcg50)
            # recalls100.extend(recall100)
            # recalls50.extend(recall50)
        
            del data, targets, pred, nonzeroidxs
        del idxlist

    ndcg100 = np.nanmean(ndcgs100)
    ndcg50 = np.nanmean(ndcgs50)
    recall100 = np.nanmean(recalls100)
    recall50 = np.nanmean(recalls50)

    del ndcgs100, ndcgs50 #, recalls100, recalls50

    return total_loss / num_data, ndcg100, ndcg50, recall100, recall50


def main():
    epochs = 200
    best_model = None
    best_val_loss = float("inf")

    dataloader = DataLoader('ml-20m/processed_data/')
    # unique_sid = dataloader.unique_sid
    # n_items = dataloader.n_items

    train_data = dataloader.load_train_data('train.csv')

    valid_data_tr, valid_data_te =  dataloader.load_tr_te_data('validation_tr.csv', 
                                                            'validation_te.csv')
                                                            
    test_data_tr, test_data_te =  dataloader.load_tr_te_data('test_tr.csv', 
                                                            'test_te.csv')

    n_train = train_data.shape[0]
    n_valid = valid_data_tr.shape[0]
    n_test = test_data_tr.shape[0]

    ntokens = train_data.shape[1] + 1 # Add padding token

    print("-- # of train users : ", n_train)
    print("-- # of valid users : ", n_valid)
    print("-- # of test users : ", n_test)
    print("-- # of items : ", ntokens)
    
    batch_size = 32
    eval_batch_size = 32
    emsize = 256
    nhid = 256
    nlayers = 1
    nhead = 2
    nseq = 120
    dropout = 0.2
    save_dir = 'checkpoints'

    model = TransformerCF(ntokens, emsize, nhead, nhid, nlayers, nseq, dropout, use_posenc=True).to(device)

    criterion = torch.nn.BCELoss()
    lr = 3. # Defualt : 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.1, gamma=0.95)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()        
        
        train(model, n_train, batch_size, nseq, train_data, epoch, criterion, optimizer, scheduler)
        val_loss, ndcg100, ndcg50, recall100, recall50 = evaluate(model, n_valid, eval_batch_size, nseq, valid_data_tr, valid_data_te, criterion, mode='valid')

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('| ndcg100: {:8.2f} | ndcg50: {:8.2f} | recall100: {:8.2f} | recall50: {:8.2f} |'.format(ndcg100, ndcg50, recall100, recall50))
        print('-' * 89)
        
        if USE_WANDB:
            wandb.log({"Loss (val)": val_loss}, step=epoch)
            wandb.log({"NDCG100 (val)": ndcg100}, step=epoch)
            wandb.log({"NDCG50 (val)": ndcg50}, step=epoch)
            # wandb.log({"Recall100 (val)": ndcg100}, step=epoch)
            # wandb.log({"Recall50 (val)": ndcg50}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), os.path.join(save_dir, 'model-epoch%d' % epoch))

        del ndcg100, ndcg50, recall100, recall50 

        scheduler.step()

    # Test Evaluation
    _, ndcg100, ndcg50, recall100, recall50 = evaluate(best_model, n_test, eval_batch_size, nseq, test_data_tr, test_data_te, criterion, mode='test')

    print('-' * 89)
    print('| Test Results of the best model (w.r.t valid loss) |')    
    print('| ndcg100: {:8.4f} | ndcg50: {:8.4f} | recall100: {:8.4f} | recall50: {:8.4f} |'.format(ndcg100, ndcg50, recall100, recall50))
    print('-' * 89)

if __name__ == '__main__':
    main()