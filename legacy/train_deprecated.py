import torchtext
import torch
import numpy as np
import pandas as pd
from transformer import TransformerModel
import time
import math
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dict = np.load('save_dict.npy').item()
sparse_matrix = load_dict['sparse_matrix']
padded_list = load_dict['padded_list']
movie_ids_to_idx = load_dict['movie_ids_to_idx']

nusers = load_dict['num_users']
ntokens = load_dict['num_movies']
batch_size = 20
eval_batch_size = 10
emsize = 256
nhid = 256
nlayers = 1
nhead = 2
nseq = 35
dropout = 0.

train_data = padded_list
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, nseq, dropout, use_posenc=True).to(device)

def pad_list(input_list, padded_len=128):
    padding = []
    input_list = list(input_list)
    if len(input_list) < padded_len:
        padding = [0] * (padded_len - len(input_list))
    output_list = input_list + padding
    output_list = output_list[:padded_len]
    return output_list

def generate_masked_binary_vector(input_list, mask_list):    
    bin_vector = np.zeros(ntokens, dtype=float)
    bin_vector[np.array(input_list, dtype=int)] = 1.
    bin_vector[mask_list] = 0.
    return bin_vector

def onehot(vec):
    A = np.eye(ntokens) #, dtype=torch.float
    return A[vec]

def get_batch(train_data, batch_size=4, seq_size=35):
    # choose num_users
    user_idx = np.random.choice(np.arange(nusers), batch_size)
    np.random.shuffle(user_idx)
    sources = []
    targets = []
    unmasked_targets = []
    
    # choose movie_seen and movie_to_be_seen (padding if input_size > num_movie_seen)
    for uid in user_idx:
        # original_bivec = generate_masked_binary_vector(user_movie_dict[uid], [])
        source_mask_idx = np.random.choice(np.arange(ntokens), ntokens//2, replace=False)
        target_mask_idx = np.arange(ntokens)[~np.isin(np.arange(ntokens), source_mask_idx)]
        
        # source_bivec = copy.deepcopy(sparse_matrix[uid])
        # source_bivec[source_mask_idx] = 0.

        target_bivec = copy.deepcopy(sparse_matrix[uid])
        target_bivec[target_mask_idx] = 0.

        # source_bivec = generate_masked_binary_vector(train_data[uid], source_mask_idx)
        # target_bivec = generate_masked_binary_vector(train_data[uid], target_mask_idx)
        # unmasked_target_bivec = generate_masked_binary_vector(train_data[uid], [])
        
        # source = source_bivec.nonzero()[0]
        # source_onehot = onehot(source)
        # source = source_bivec.nonzero()[0]
        source = padded_list[uid][:seq_size]
        # pad_list(list(source), padded_len=seq_size)

        # source = pad_list(list(source), padded_len=input_size)
        # source = np.array(source)
        # source = torch.Tensor(source).int().to(device)
        # source_onehot = onehot(source)
        
        sources.append(source)
        targets.append(target_bivec)
        # unmasked_targets.append(unmasked_target_bivec)

    sources = torch.Tensor(sources).to(device).permute((1,0)).long()    # [num_seq, batch_size]
    targets = torch.Tensor(targets).to(device)                   # [batch_size, num_seq, num_tokens]
    # unmasked_targets = torch.Tensor(unmasked_targets).to(device)

    return sources, targets  #, unmasked_targets

# train_data = batchify(train_txt, batch_size)
# val_data = batchify(val_txt, eval_batch_size)
# test_data = batchify(test_txt, eval_batch_size)

criterion = torch.nn.BCELoss()
lr = 5.0 # 학습률
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train():
    model.train() # 학습 모드를 시작합니다.
    total_loss = 0.
    start_time = time.time()
    # ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, nusers, batch_size)):
        data, targets = get_batch(train_data, batch_size)
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
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()

# def evaluate(eval_model, data_source):
#     eval_model.eval() # 평가 모드를 시작합니다.
#     total_loss = 0.
#     ntokens = len(TEXT.vocab.stoi)
#     with torch.no_grad():
#         for i in range(0, data_source.size(0) - 1, bptt):
#             data, targets = get_batch(data_source, i)
#             output = eval_model(data)
#             output_flat = output.view(-1, ntokens)
#             total_loss += len(data) * criterion(output_flat, targets).item()
#     return total_loss / (len(data_source) - 1)

# import math

# best_val_loss = float("inf")
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    # val_loss = evaluate(model, val_data)
    # print('-' * 89)
    # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
    #       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
    #                                  val_loss, math.exp(val_loss)))
    # print('-' * 89)

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     best_model = model

    scheduler.step()
