from triplet_model import TripletModel
from utils import load_json, load_pickle,  TextToVec
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader




NEW_DATA_DIR = '../data/varible_test'          # original info, for test
NEW_DATA_V2_DIR = '../data/varible_train'
NEW_DATA_V3_DIR = '../data/varible'
WHOLE_AUTHOR_PROFILE_PATH = '../data/whole_author_profile.json'
WHOLE_AUTHOR_PROFILE_PUB_PATH = '../data/whole_author_profile_pub.json'



BATCH_SIZE = 512
LR = 0.01
EPOCHS = 20


class ReadData(Dataset):
    def __init__(self, posi_pair_path, neg_pair_path, whole_profile_pub, aid2cate, cate='title'):#aid2titlevec
        super().__init__()
        self.posi_pair = load_pickle(posi_pair_path)
        self.neg_pair = load_pickle(neg_pair_path)
        self.whole_profile_pub = whole_profile_pub
        self.posi_pid2aid = {}
        for pair in self.posi_pair:
            self.posi_pid2aid[pair[1]] = pair[0]
        self.neg_pid2aid = {}
        for pair in self.neg_pair:
            self.neg_pid2aid[pair[1]] = pair[0]
        self.cate = cate
        self.texttovec = TextToVec()
        self.aid2cate = aid2cate #(vector)

        posi_pids = set(self.posi_pid2aid.keys())
        neg_pids = set(self.neg_pid2aid.keys())
        self.innter_pid_set = list(posi_pids & neg_pids)

    def __len__(self):
        return len(self.innter_pid_set)

    def __getitem__(self, index):
        #当前论文文本信息；
        pid_with_index = self.innter_pid_set[index]
        pid, _ = pid_with_index.split('-')
        info = self.whole_profile_pub[pid].get(self.cate) #'title'
        if info is None:
            anchor_data = np.zeros(300)
        else:
            anchor_data = self.texttovec.get_vec(info)
        posi_data = self.aid2cate[self.posi_pid2aid[pid_with_index]]
        neg_data = self.aid2cate[self.neg_pid2aid[pid_with_index]]

        anchor_data = torch.from_numpy(np.expand_dims(anchor_data, axis=0)).to(torch.float)
        posi_data = torch.from_numpy(np.expand_dims(posi_data, axis=0)).to(torch.float)
        neg_data = torch.from_numpy(np.expand_dims(neg_data, axis=0)).to(torch.float)
        return anchor_data, posi_data, neg_data


def AccuracyDis(anchor_emb, posi_emb, neg_emb):
    pos_distance = torch.sqrt(torch.sum(torch.pow((anchor_emb - posi_emb), 2), dim=1))
    neg_distance = torch.sqrt(torch.sum(torch.pow((anchor_emb - neg_emb), 2), dim=1))
    acc = torch.mean((pos_distance < neg_distance).to(torch.float))
    return acc


if __name__ == "__main__":
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    train_posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'train-posi-pair-list.pkl')
    train_neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'train-neg-pair-list.pkl')
    test_posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'test-posi-pair-list.pkl')
    test_neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'test-neg-pair-list.pkl')

    # all_posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend1.pkl')
    # all_neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend1.pkl')

    #某个作者所有论文摘要组成的向量
    # aid2abstractvec = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2abstractvec.pkl'))
    aid2titlevec = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2titlevec.pkl'))

    keyarg = {
        'aid2cate': aid2titlevec,
        'cate': 'title',
        # 'aid2cate': aid2abstractvec,
        # 'cate': 'abstract'
    }
    print(keyarg['cate'])
    train_dataset = ReadData(train_posi_pair_path, train_neg_pair_path, whole_author_profile_pub, **keyarg)
    test_dataset = ReadData(test_posi_pair_path, test_neg_pair_path, whole_author_profile_pub, **keyarg)

    # all_dataset = ReadData(all_posi_pair_path, all_neg_pair_path, whole_author_profile_pub, **keyarg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    # loader = DataLoader(train_loader, batch_size=BATCH_SIZE, num_workers=2)
    triplet_model = TripletModel().to(device)
    criterion = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(triplet_model.parameters(), lr=LR)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10) #基于epoch训练次数进行学习率调整,#每过30个epoch训练，学习率就乘factor


    ind = 0
    for epoch in range(EPOCHS):
        triplet_model.train()
        train_loss = []
        for anchor, posi, neg in train_loader:
            anchor, posi, neg = anchor.to(device), posi.to(device), neg.to(device)
            if ind ==0:
                print("anchor:",anchor.size(),'\n',anchor) #torch.Size([512, 1, 300])
            optimizer.zero_grad()
            embs = triplet_model(anchor, posi, neg)
            loss = criterion(*embs)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            ind += 1

        triplet_model.eval()
        test_loss = []
        accuracy = []
        with torch.no_grad():
            for test_anchor, test_posi, test_neg in test_loader:
                test_anchor, test_posi, test_neg = test_anchor.to(device), test_posi.to(device), test_neg.to(device)
                test_embs = triplet_model(test_anchor, test_posi, test_neg)
                loss = criterion(*test_embs)
                acc = AccuracyDis(*test_embs)
                accuracy.append(acc.item())
                test_loss.append(loss.item())
        lr_schedule.step()
        # print('Epoch: [%d/%d], train loss: %f, test loss %f, acc: %f' % (epoch + 1, EPOCHS, np.mean(train_loss), np.mean(test_loss), np.mean(accuracy)))
        # print('Epoch: [%d/%d], train loss: %f\n' % (epoch + 1, EPOCHS, np.mean(train_loss)))
    torch.save(triplet_model.state_dict(), '../model/tripletmodel.%s.checkpoint' % keyarg['cate'])
