#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/10/27
@email:2218330483@qq.com
@function： final version jointly optimal the beamforming and phase shifts
"""
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
# 导入环境和学习方法
import pandas as pd
from sec_env import channel_env
from ddpg import *


class Config:
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DDPG'
        # 算法名称
        self.env_name = 'Pendulum-v1'
        # 环境名称，gym新版本（约0.21.0之后）中Pendulum-v0改为Pendulum-v1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 检测是否有GPU
        self.eps = 100
        self.eps_step = 10000
        # 训练的回合数和步数（多少个episode)
        self.test_eps = 20
        # 测试的回合数（多少个episode)
        self.memory_capacity = 100000
        # 经验回放的容量
        self.batch_size = 16
        # mini-batch SGD中的批量大小（每一次从经验回放中提取多少的样本出来）


"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3,
                 min_sigma=0.3, decay_period=100000):
        self.mu = mu
        # OU噪声的参数（均值）
        self.theta = theta
        # OU噪声的参数（均值项的系数）
        self.sigma = max_sigma
        # OU噪声的参数（布朗运动项的系数）
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.n_actions = action_space
        self.low = 1
        # 2
        self.high = -1
        # -2
        self.reset()

    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        # 注：这里的OU noise中dt为1（Atari游戏的dt），所以看起来少了一项
        # 标准的OU noise中的dx，第一项要乘一个dt，第二项要乘一个sqrt(dt)

        self.obs = x + dx
        # 更新OU noise (加上dx的部分)

        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        # 加了noise的action

        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) \
                     * min(1.0, t / self.decay_period)
        # sigma会逐渐衰减，直到衰减到min_sigma
        # 但这里默认max_sigma和min_sigma是一样大的，所以sigma这里是不会变化的

        return np.clip(action + ou_obs, self.low, self.high)
        # 动作加上噪声后进行剪切（在action合理的区间内）


def Train_test(Pmax, N, P_I, sigma_I):
    M = 4
    sigma = np.power(10, -80 / 10)  # 80dBm AWGN
    # 设置环境
    env = channel_env(M, N, Pmax, P_I, sigma, sigma_I)
    s_dim = env.state_dim
    a_dim = env.a_dim
    OU_noise = OUNoise(N)
    # writer = SummaryWriter('logs')
    cfg = Config()
    print('开始训练！')
    print(f'环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}')

    # Initialize the algorithm
    agent = DDPG(s_dim, a_dim, cfg)
    # rewards = []
    # avg_rewards = []
    data_eps = []
    max_rate = 0
    for episode in range(cfg.eps):
        noise = 1
        state, done = env.reset(), False
        OU_noise.reset()
        episode_reward = 0
        data_step = []
        for step in range(cfg.eps_step):
            action = agent.select_action(state)
            action = OUNoise.get_action(action, episode)
            # action = action + np.random.normal(0, noise, a_dim)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            max_rate = max(max_rate, reward)
            agent.update_parameters(cfg.batch_size)
            noise *= 0.9995
            data_step.append(reward)
            # writer.add_scalar('loss_c', critic_loss, step)
            # writer.add_scalar('loss_a', actor_loss, step)
        print(f"Epside:{episode + 1}/{cfg.eps}, Reward:{episode_reward:.1f},Max.Reward: {max_rate:.3f}\n")
        data_eps.append(data_step)
        # rewards.append(episode_reward / cfg.eps_step)
        # avg_rewards.append(np.mean(rewards[-10:]))
    df = pd.DataFrame(data_eps)
    # df.to_csv(f"{str(Pmax)}-{str(N)}-results.csv", index=False)
    df.to_csv(f"results.csv", index=False)
    # writer.close()
    return max_rate


sigma_I = np.power(10, -80 / 10)  # 80dBm active RIS thermal noise
maxrate = Train_test(10, 16, 0, 0)
print(maxrate)
# P = [10, 20, 40]
# rate = []
# for i in P:
#     _, avg_rwd, maxrate = Train_test(i, 16, 0, 0)  # Pmax, Ny, P_I, sigma_I
#     rate.append([avg_rwd[-1], maxrate])
# print(rate)
# N = [32, 40, 64]
# rate = []
# for n in N:
#     _, avg_rwd, maxrate = Train_test(10, n, 0, 0)  # Pmax, Ny, P_I, sigma_I
#     rate.append([avg_rwd[-1], maxrate])
# print(rate)
