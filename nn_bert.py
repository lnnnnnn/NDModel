import pandas as pd
import numpy as np
import torch
import re
import transformers
import os
import random
import time
from multiprocessing import Process,cpu_count,Manager,Pool
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader

# from transformers import *
from transformers import (BertModel, BertTokenizer)

MODELS = [(BertModel,  BertTokenizer)]
for model_class, tokenizer_class in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained('../model/scibert_scivocab_uncased')
    model = model_class.from_pretrained('../model/scibert_scivocab_uncased')

class BertTrans(nn.Module):
    def __init__(self,model):
        super(BertTrans, self).__init__()

        self.bert = model
        self.linear_origin = nn.Linear(768, 1)

    def forward(self, input_ids_1=None):

        origin_data = self.bert(input_ids_1)[0]
        output = self.linear_origin(origin_data[:,0,:])
        logits = torch.sigmoid(output)
        return logits

max_seq_length = 300

train = pd.read_csv('../data/bert_corpus.csv')

def convert_data(data, max_seq_length_a=200, max_seq_length_b=500, tokenizer=None):
    all_tokens = []
    longer = 0
    for row in data.itertuples():
        paper_abs = getattr(row, "paper_abs")

        # print(paper_abs)
        tokens_a = tokenizer.tokenize(getattr(row, "paper_abs"))
        tokens_b = tokenizer.tokenize(getattr(row, "author_corpus"))
        if len(tokens_a)>max_seq_length_a:
            tokens_a = tokens_a[:max_seq_length_a]
            longer += 1
        if len(tokens_a)<max_seq_length_a:
            tokens_a = tokens_a+[0] * (max_seq_length_a - len(tokens_a))
        if len(tokens_b)>max_seq_length_b:
            tokens_b = tokens_b[:max_seq_length_b]
            longer += 1
        if len(tokens_b)<max_seq_length_b:
            tokens_b = tokens_b+[0] * (max_seq_length_b - len(tokens_b))
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"]+tokens_b+["[SEP]"])
        all_tokens.append(one_token)
    data['bert_token'] = all_tokens
    return data

def pool_extract(data, f ,chunk_size, max_seq_length,tokenizer, worker=4):
    cpu_worker = cpu_count()
    print('cpu core:{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker
    print('use cpu:{}'.format(worker))
    t1 = time.time()
    len_data = len(data)
    start = 0
    end = 0
    p = Pool(worker)
    res = []  # 保存的每个进程的返回值
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        rslt = p.apply_async(f, (data[start:end],100,200,tokenizer))
        start = end
        res.append(rslt)
    p.close()
    p.join()
    for tmp in [i.get() for i in res]:
        print (tmp.shape) 
    t2 = time.time()
    print((t2 - t1)/60)
    results = pd.concat([i.get() for i in res], axis=0, ignore_index=True)
    return results

train = pool_extract(train,convert_data,50000,max_seq_length,tokenizer,15)

import pickle
pickle.dump(train[['bert_token','label']],open('../data/bert/bert_data.pkl','wb'))

x_torch = torch.tensor(train['bert_token'].values.tolist(), dtype=torch.long)#.cuda()
y_train_torch = torch.tensor(train['label'][:, np.newaxis],
                             dtype=torch.float32)#.cuda()

class MyDataset(Dataset):
    def __init__(self, data1,labels):
        self.data1= data1
        self.labels = labels

    def __getitem__(self, index):    
        d1,target = self.data1[index], self.labels[index]
        return d1,target

    def __len__(self):
        return len(self.data1)

batch_size = 16#96
n_epochs=4
loss_fn = torch.nn.BCELoss()

train_dataset = MyDataset(x_torch, y_train_torch)

model = BertTrans(model)

n_gpu=4

# ## bert训练采用的优化方法就是adamw，对除了layernorm，bias项之外的模型参数做weight decay
# bias 的更新跟权重衰减无关
param_optimizer = list(model.named_parameters())
print('model.named_parameters():\n',model.named_parameters())
no_decay = [ 'LayerNorm.bias', 'LayerNorm.weight','bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=0.001, correct_bias=True)

###########################################################################

total_steps = (x_torch.shape[0]/768*n_epochs/batch_size )
#学习率预热，训练时先从小的学习率开始训练
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                      num_training_steps=total_steps)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


for epoch in range(n_epochs):
    start_time = time.time()

    scheduler.step()

    model.train()
    avg_loss = 0.
    optimizer.zero_grad()
    count = 0
    for data in tqdm(train_loader, disable=False):
        x_batch = data[:-1][0]#.cuda()
        y_batch = data[-1]#.cuda()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()

        optimizer.step()
        model.zero_grad()
        avg_loss += loss.item() / len(train_loader)
        #each_loss = loss.item()/((count+1)*batch_size)
        count = count+1
#         model.eval()
        #print('loss={:.4f}'.format( each_loss),flush=True)
    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
          epoch + 1, n_epochs, avg_loss, elapsed_time))
    torch.save(model, '../model/model_bert.pkl_'+str(epoch))
