from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, num_input):
        super(Linear, self).__init__()
        self.dim = num_input
        self.linear = nn.Linear(self.dim, 2)

    def forward(self, inputs):
        h = self.linear(inputs)
        return h


class LinearSVMClassifer():
    def __init__(self, num_input, c=1.0, cuda=False):
        self.c = c
        self.dim = num_input
        self.cuda = cuda
        self.model = Linear(self.dim)

    def fit(self, train_x, train_y, n_epoch=50, batch_size=100):
        X = torch.FloatTensor(train_x)
        Y = torch.FloatTensor(train_y)
        N = len(Y)
        if self.cuda:
            self.model.cuda()

        optimizer = optim.Adam(self.model.parameters())
        self.model.train()
        total_loss = []
        for epoch in range(n_epoch):
            perm = torch.randperm(N)
            sum_loss = 0
            for i in range(0, N, batch_size):
                x = X[perm[i:i + batch_size]]
                y = Y[perm[i:i + batch_size]]
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                x = Variable(x)
                y = Variable(y)
                optimizer.zero_grad()
                output = self.model(x)
                loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
                loss += self.c * torch.mean(self.model.linear.weight ** 2)  # l2 penalty
                loss.backward()
                optimizer.step()
                sum_loss += loss.data[0]

            total_loss.append(sum_loss/N)

    def predict(self, test_x):
        self.model.eval()
        x = Variable(torch.FloatTensor(test_x))
        output = self.model(x)
        return F.log_softmax(output).data.max(1, keepdim=True)[1]





