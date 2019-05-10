#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:20:00 2018

@author: William
"""

import torch as t
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

x = t.unsqueeze(t.linspace(-1,1,100),dim=1)
y = x.pow(3) + 0.2*t.rand(x.size())

plt.scatter(x.numpy(),y.numpy())
plt.show()

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
net = Net(1,9,1)
print(net)

params = list(net.parameters())
print("The number of parameters:",len(params))

params = list(net.parameters())
print(len(params))

optimizer = t.optim.SGD(net.parameters(),lr=0.5)
loss_func = nn.MSELoss()

plt.ion()
plt.show()

for T in range(100):
    prediction = net(x)
    
    loss = loss_func(prediction,y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if T%5 == 0:
        plt.cla()
        plt.scatter(x.numpy(),y.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data[0],fontdict={'size':20,'color':"red"})
        plt.pause(0.1)
        
plt.ioff()
plt.show()
