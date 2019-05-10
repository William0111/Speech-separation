#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:35:00 2018

@author: William
"""

import torch as t
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


#new created data
n_data = t.ones(100, 2)
x0 = t.normal(2*n_data, 1)
y0 = t.zeros(100)
x1 = t.normal(-2*n_data, 1)
y1 = t.ones(100)

x = t.cat((x0, x1), 0).type(t.FloatTensor)  # FloatTensor = 32-bit floating
y = t.cat((y0, y1), ).type(t.LongTensor)    # LongTensor = 64-bit integer

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)
print(net)

params = list(net.parameters())
print("The number of parameters:", len(params))

params = list(net.parameters())
print(len(params))

optimizer = t.optim.SGD(net.parameters(), lr=0.0005)
loss_func = nn.CrossEntropyLoss()

plt.ion()
plt.show()

for T in range(1000):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if T % 50 == 0:
        plt.cla()

        prediction = t.max(F.softmax(out),1)[1]
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

