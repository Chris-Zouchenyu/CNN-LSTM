#导包
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential,Module,Conv2d,MaxPool2d,Linear,LSTM,ReLU,Dropout2d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#模型搭建
'''
train_X大小为(493,7,5)可以视为493张7*5的图像,因而可以用CNN(卷积神经网络)处理
'''
window_size = 7
fea_num = 5
batch_size = 32
class CNN_LSTM(nn.Module):
    '''
    This is a new model
    window_size:
    fea_num:

    '''
    def __init__(self,window_size,fea_num):
        super().__init__()
        self.window_size = window_size
        self.fea_num = fea_num
        self.model1 = Sequential(
            Conv2d(in_channels = 1,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            ReLU(),
            MaxPool2d(kernel_size = 3,stride = 1,padding = 1),
            Dropout2d(0.3),
        )
        self.lstm1 = LSTM(input_size = 64*fea_num, hidden_size = 128, num_layers = 1)
        self.lstm2 = LSTM(input_size = 128, hidden_size = 64, num_layers = 1)
        self.linear1 = Linear(in_features = 64, out_features = 32)
        self.relu = ReLU()
        self.linear2 = Linear(in_features = 32, out_features = 1)
    def forward(self,x):
        x = x.reshape([x.shape[0],1,self.window_size,self.fea_num]) 
        x = self.model1(x)

        x = x.reshape([x.shape[0], self.window_size, -1])
        x,(h,c) = self.lstm1(x)
        x,(h,c) = self.lstm2(x)
        x = x[:,-1,:]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
mymodel = CNN_LSTM(window_size,fea_num)
# test1 = torch.ones((99,7,5))
# print(test1.shape)
# print((mymodel(test1.float())).shape)