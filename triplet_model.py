import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 2) #in_channels（词向量的维度）, out_channels（卷积核的个数）, kernel_size
        # self.maxpool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 1)
        # self.maxpool2 = nn.MaxPool1d(2)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        # x = self.maxpool2(x)
        x = self.drop(x)
        x = self.fc(x)
        return torch.squeeze(x)


class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddingnet = EmbeddingNet()

    def forward(self, anchor, posi, neg):
        anchor_emb = self.embeddingnet(anchor)
        posi_emb = self.embeddingnet(posi)
        neg_emb = self.embeddingnet(neg)
        return anchor_emb, posi_emb, neg_emb

    def get_emb(self, x):
        return self.embeddingnet(x)


