#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:28:30
@LastEditor: John
LastEditTime: 2021-09-16 00:52:30
@Discription: 
@Environment: python 3.7.7
'''
import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        # 翻译一下，这边做的事情就是把action的数值乘个2，然后clip到action合理的数值内
        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, \
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
        self.n_actions = action_space.shape[0]
        self.low = action_space.low
        # 2
        self.high = action_space.high
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
