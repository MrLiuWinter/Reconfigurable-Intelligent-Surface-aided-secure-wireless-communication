#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2022-06-09 19:04:44
@Discription: 
@Environment: python 3.7.7
'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # 如果经验回放没满的话，直接append，否则替代掉position时刻的经验回放
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 随机采出一个batch的transition
        state, action, reward, next_state, done = zip(*batch)
        # 将这个batch里面的state, action, reward, next_state, done分别拼起来
        # 每一个是一个tuple
        return state, action, reward, next_state, done

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
        # [batch_size,3]——>[batch_size,1]

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        # [batch_size,3],[batch_size,1]——>[batch_size,1]


class DDPG:
    def __init__(self, n_states, n_actions, cfg):
        self.device = cfg.device
        self.critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        # critic——输入state和actor的输出(action)，得到一个scalar
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        # actor——输入state,输出离散的action
        self.target_critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        # actor,critic以及对应的目标函数

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        # 初始化的时候，复制参数到目标网络

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        # actor和critic的优化器

        self.memory = ReplayBuffer(cfg.memory_capacity)
        # 经验回放，一个数组
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        # 软更新参数
        self.gamma = cfg.gamma
        # 折扣系数

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 一维Tensor (shape为[3])变成二维Tensor（shape为[1,3]）

        action = self.actor(state)
        # [1,3]——>[1,1]
        a = action.detach().cpu().numpy()[0]
        b = action.detach().cpu().numpy()[0, 0]
        return action.detach().cpu().numpy()[0, 0]
        # 返回action对应的float

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # 当经验回放中transition的数量不满一个batch时，不更新策略

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        # [batch_size,3]
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        # [batch_size,3]
        action = torch.FloatTensor(np.array(action)).to(self.device)
        # [batch_size,1]
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        # [batch_size,1]
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # [batch_size,1]

        ########################计算Actor的loss ############################

        policy_loss = self.critic(state, self.actor(state))
        # 当前时刻的critic预测值
        # [batch_size,1]
        policy_loss = -policy_loss.mean()
        # 由于policy network是梯度上升，所以这里需要加一个负号
        ####################################################################

        ########################计算 Critic的TD loss########################

        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        # next action是target network的结果，所以不用梯度下降（不用更新参数），这里需要detach掉

        expected_value = reward + (1.0 - done) * self.gamma * target_value
        # 如果这个episode还没有结束,那么就加上后面的target value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        # 这两步是计算TD target

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        # TD loss

        ##################################################################

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        # 更新actor
        '''
        可以发现这里用pytorch实现的时候 并没有按照DDPG公式那样计算两个内容的偏导，而是直接对policy_loss求导
        因为actor_optimizer在初始化的时候,存进去的是self.actor.parameters()
        所以进行zero_grad和step的时候，会自动计算这些系数（也就是θμ）的梯度，不用按照算法中实际公式那样地计算
        '''

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 更新critic

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        # 每一次training 软更新target_critic

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        # 每一次training 软更新target_actor

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')
        # 由于决策的时候只需要actor,所以我们保存parameter的时候，只需要保存actor的参数即可

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))
