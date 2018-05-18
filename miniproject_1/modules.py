import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

MLP = nn.Sequential(
        nn.Linear(1400, 140),
        nn.Tanh(),
        nn.Linear(140, 28),
        nn.Tanh(),
        nn.Linear(28, 2)
    )


class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.fc1 = nn.Linear(128, 28)
        self.fc2 = nn.Linear(28, 2)

    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.tanh(F.max_pool2d(self.conv2(x), kernel_size=(3,2), stride=(3,2)))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.fc1(x.view(x.size(0),-1)))
        x = self.fc2(x)
        return x


class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, kernel_size=3)
        self.conv2 = nn.Conv1d(28, 14, kernel_size=3)
        self.conv3 = nn.Conv1d(14, 1, kernel_size=3)
        self.fc1 = nn.Linear(9, 36)
        self.fc2 = nn.Linear(36, 2)

    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.conv1(x), kernel_size=2, stride=2))
        x = F.tanh(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.tanh(self.conv3(x))
        x = self.fc1(x.squeeze(1))
        x = self.fc2(F.tanh(x))
        return x


class CNN_dropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNN_dropout, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, kernel_size=3)
        self.conv2 = nn.Conv1d(28, 14, kernel_size=3)
        self.conv3 = nn.Conv1d(14, 1, kernel_size=3)
        self.fc1 = nn.Linear(9, 30)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.conv1(x), kernel_size=2, stride=2))
        x = F.tanh(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.tanh(self.conv3(x))
        x = self.fc1(x.squeeze(1))
        x = self.dropout(x)
        x = self.fc2(F.tanh(x))
        return x


class CNN_batchnorm(nn.Module):
    def __init__(self):
        super(CNN_batchnorm, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(28)
        self.conv2 = nn.Conv1d(28, 14, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(14)
        self.conv3 = nn.Conv1d(14, 1, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(9, 30)
        self.bn4 = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.bn1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.tanh(F.max_pool1d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = self.fc1(x.squeeze(1))
        x = self.fc2(F.tanh(self.bn4(x)))
        return x


class CNN_both(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNN_both, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(28)
        self.conv2 = nn.Conv1d(28, 14, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(14)
        self.conv3 = nn.Conv1d(14, 1, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(9, 30)
        self.bn4 = nn.BatchNorm1d(30)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.bn1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.tanh(F.max_pool1d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = self.fc1(x.squeeze(1))
        x = self.dropout(self.bn4(x))
        x = self.fc2(F.tanh(x))
        return x