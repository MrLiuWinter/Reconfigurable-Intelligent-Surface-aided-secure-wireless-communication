#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/09/27 
@email:2218330483@qq.com
@function： 
"""
import datetime
import os

import matplotlib.pyplot as plt
# 导入环境和学习方法
import numpy as np
import pandas as pd
import torch

from ddpg import DDPG
from sec_env import channel_env

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 获取当前时间
curr_path = os.path.dirname(os.path.abspath(__file__))


# 当前文件所在绝对路径


class Config:
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DDPG'
        # 算法名称
        self.env_name = 'Pendulum-v1'
        # 环境名称，gym新版本（约0.21.0之后）中Pendulum-v0改为Pendulum-v1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 检测是否有GPU
        self.train_eps = 200
        self.train_eps_step = 1000
        # 训练的回合数和步数（多少个episode)
        self.test_eps = 20
        # 测试的回合数（多少个episode)
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.99
        # 折扣因子
        self.critic_lr = 1e-9
        # critic网络的学习率
        self.actor_lr = 1e-9
        # actor网络的学习率
        self.memory_capacity = 100000
        # 经验回放的容量
        self.batch_size = 32
        # mini-batch SGD中的批量大小（每一次从经验回放中提取多少的样本出来）
        self.hidden_dim_1 = 300
        self.hidden_dim_2 = 200
        # 网络隐藏层维度
        self.soft_tau = 1e-4
        # 软更新参数
        ################################################################################

        ################################# 保存结果相关参数###############################
        self.result_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'
        # 保存结果的路径
        self.model_path = curr_path + "/outputs/models/"
        # 保存模型的路径
        self.save = True
        # 是否保存图片
        ################################################################################


def Train_test(Pmax, N, P_I, sigma_I):
    cfg = Config()
    print('开始训练！')
    print(f'环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}')
    M = 4
    sigma = np.power(10, -110 / 10)  # 80dBm AWGN
    env = channel_env(M, N, Pmax, P_I, sigma, sigma_I)
    s_dim = env.state_dim
    a_dim = env.action_dim
    agent = DDPG(s_dim, a_dim, cfg)

    rewards = []
    avg_rewards = []
    data_eps = []
    max_rate = 0
    for episode in range(cfg.train_eps):
        state, done = env.reset(), False
        episode_reward = 0
        data_step = []
        for step in range(cfg.train_eps_step):
            action = agent.choose_action(state)
            noise = np.random.normal(loc=0, scale=0.1, size=N)
            action = action + noise
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            episode_reward += reward
            max_rate = max(max_rate, reward)
            agent.update()
            state = next_state
            data_step.append(reward)
        print(f"Epside:{episode + 1}/{cfg.train_eps}, Reward:{episode_reward:.1f}")
        data_eps.append(data_step)
        rewards.append(episode_reward / cfg.train_eps_step)
        avg_rewards.append(np.mean(rewards[-50:]))  # Find the mean value in a group of last 10 items
    data_eps.append(rewards)
    data_eps.append(avg_rewards)
    df = pd.DataFrame(data_eps)
    df.to_csv("results.csv", index=False)
    return rewards, avg_rewards, max_rate


P_inter = [0, 10, 20, 30]
sigmas_I = np.power(10, -110 / 10)  # 80dBm active RIS thermal noise
rwd, avg_reward, maxrate = Train_test(-20, 16, 0, 0)
x = np.linspace(0, len(rwd) - 1, len(rwd))
print(maxrate)
plt.plot(x, rwd, "r-", label="eposide first")
plt.plot(x, avg_reward, "b-", label="eposide first")
plt.xlabel("steps")
plt.ylabel("Rewards")
plt.title("channel")
plt.legend(loc='lower right')
plt.grid()
plt.show()
# P = [-20, -15, -10, -5, 0, 5, 10]
# rate = []
# for i in P:
#     rewards, avg_rewards, maxrate = Train_test(i, 16, 0, 0)  # Pmax, Ny, P_I, sigma_I
#     rate.append([avg_rewards[-1], maxrate])
#     x = np.linspace(0, len(rewards) - 1, len(rewards))
#     plt.plot(x, rewards, "r-", label=f"reward, P={i}dBm")
#     plt.plot(x, avg_rewards, "b-", label=f"Average reward, P={i}dBm")
#     plt.xlabel("steps")
#     plt.ylabel("Rewards")
#     plt.title("channel")
#     plt.legend(loc='lower right')
#     plt.grid()
#     plt.show()
# print(rate)
