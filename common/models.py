#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 21:14:12
LastEditor: John
LastEditTime: 2022-08-29 14:24:44
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ConvNet(nn.Module):
    def __init__(self,input_dim, n_states,n_max_space,n_actions,hidden_dim=128):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(223488, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class ConvNet(nn.Module):
#     def __init__(self,input_dim, n_states,n_max_space,n_actions,hidden_dim=128):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * int(input_dim[0]/4) * int(input_dim[1]/4), 256)
#         self.fc2 = nn.Linear(256, n_actions)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class MLP3(nn.Module):
    def __init__(self,input_dim, n_states,n_max_space,n_actions,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态维度
            n_actions: 输出的动作维度
        """
        super(MLP3, self).__init__()
        # self.lstm = nn.LSTM(n_max_space, n_max_space//4)
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_actions) # 输出层
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 各层对应的激活函数
        # x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLP2(nn.Module):
    def __init__(self, n_states,n_max_space,n_actions,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态维度
            n_actions: 输出的动作维度
        """
        super(MLP2, self).__init__()
        # self.lstm = nn.LSTM(n_max_space, n_max_space//4)
        self.fc1 = nn.Linear(n_states*n_max_space, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim) # 隐藏层
        self.fc4 = nn.Linear(hidden_dim, n_actions) # 输出层
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 各层对应的激活函数
        # x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的特征数即环境的状态维度
            output_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorSoftmax, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self,state):
        dist = F.relu(self.fc1(state))
        dist = F.softmax(self.fc2(dist),dim=1)
        return dist
class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=256):
        super(Critic,self).__init__()
        assert output_dim == 1 # critic must output a single value
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self,state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        return value

class ActorCriticSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, actor_hidden_dim=256,critic_hidden_dim=256):
        super(ActorCriticSoftmax, self).__init__()

        self.critic_fc1 = nn.Linear(input_dim, critic_hidden_dim)
        self.critic_fc2 = nn.Linear(critic_hidden_dim, 1)

        self.actor_fc1 = nn.Linear(input_dim, actor_hidden_dim)
        self.actor_fc2 = nn.Linear(actor_hidden_dim, output_dim)
    
    def forward(self, state):
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        policy_dist = F.relu(self.actor_fc1(state))
        policy_dist = F.softmax(self.actor_fc2(policy_dist), dim=1)

        return value, policy_dist

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value