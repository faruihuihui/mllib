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
        self.linear = nn.Linear(self.dim, 1)

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
        return output


def visualize(X, model):
    import numpy as np
    import matplotlib.pyplot as plt
    W = model.weight[0].data.cpu().numpy()
    b = model.bias[0].data.cpu().numpy()

    delta = 0.01
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = map(np.ravel, [x, y])

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.)] = 4
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    z[np.where(z <= -1.)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.xticks([])
    plt.yticks([])
    plt.contourf(x, y, z, alpha=0.8)
    plt.scatter(x=X[:, 0], y=X[:, 1], c='black', s=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs

    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    Y = Y * 2 - 1
    svmc = LinearSVMClassifer(2, 0.1)
    svmc.fit(X, Y)
    visualize(X, svmc.model.linear)





