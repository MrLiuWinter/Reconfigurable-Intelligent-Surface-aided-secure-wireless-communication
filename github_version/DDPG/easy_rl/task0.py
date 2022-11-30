#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2022-07-21 21:51:34
@Discription: 
@Environment: python 3.7.7
'''
import datetime
import os
import pandas as pd
import gym
import torch

from ddpg import DDPG
from env import NormalizedActions, OUNoise
from utils import plot_rewards
from utils import save_results, make_dir

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
        self.seed = 10
        # 随机种子，置0则不设置随机种子
        self.train_eps = 300
        # 训练的回合数（多少个episode)
        self.test_eps = 20
        # 测试的回合数（多少个episode)
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.99
        # 折扣因子
        self.critic_lr = 1e-3
        # critic网络的学习率
        self.actor_lr = 1e-4
        # actor网络的学习率
        self.memory_capacity = 8000
        # 经验回放的容量
        self.batch_size = 128
        # mini-batch SGD中的批量大小（每一次从经验回放中提取多少的样本出来）
        self.hidden_dim = 256
        # 网络隐藏层维度
        self.soft_tau = 1e-2
        # 软更新参数
        ################################################################################

        ################################# 保存结果相关参数###############################
        self.result_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'
        # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/models/'
        # 保存模型的路径
        self.save = True
        # 是否保存图片
        ################################################################################


def env_agent_config(cfg, seed=1):
    env0 = gym.make(cfg.env_name)
    '''
    print(env0.observation_space)
    print(env0.action_space)
    Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
    Box([-2.], [2.], (1,), float32)
    '''

    env = NormalizedActions(env0)
    '''
    print(env.observation_space)
    print(env.action_space)
    Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
    Box([-2.], [2.], (1,), float32)
    尚未调用action函数，所以封装前后目前是一样的
    '''

    # env.seed(seed)  # 随机种子
    n_states = env.observation_space.shape[0]  # 3
    n_actions = env.action_space.shape[0]  # 1

    agent = DDPG(n_states, n_actions, cfg)
    return env, agent


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}')

    ou_noise = OUNoise(env.action_space)
    # 动作噪声（OU噪声，相邻时间片的噪声满足AR(1)）

    rewards = []
    # 记录所有回合的奖励
    ma_rewards = []
    # 记录所有回合的滑动平均奖励
    data_eps = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()[0]
        # 即observation
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        data_step = []
        while not done:  # 200step 自动结束
            i_step += 1
            action = agent.choose_action(state)
            # 根据actor网络计算action
            # 注意：此时action的取值范围是[-1,1]，因为tanh是最后一层的激活函数

            action = ou_noise.get_action(action, i_step)
            # 添加了OU noise之后的action（OU noise 可以看成是一个ar(1)的noise）
            # 注意：此时action的取值范围虽然是[-2,2]，但主体（去噪之后的信号）还是[-1,1]
            # ——>和action的实际取值范围还是有一定的出入

            next_state, reward, _, done, _ = env.step(action)
            # 由于之前算出来的action是[-1,1]（再往外伸出一点点）
            # 但实际的action范围是[-2,2]，所以需要ActionWrapper来进行封装，使得action整体乘个2
            # 然后拿乘了2的action和环境做交互

            ep_reward += reward
            # 这一个episode的reward

            agent.memory.push(state, action, reward, next_state, done)
            # 将这一时刻的transition(st,at,rt,s_{t+1})存入经验回放中

            agent.update()
            # 更新actor和critic的参数,同时对相应的目标网络进行软更新
            state = next_state
            data_step.append(reward)

        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}，奖励：{:.2f}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        # 每10个episode 输出一次结果，这一个episode的累计奖励
        data_eps.append(data_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        # 滑动平均奖励
    print('完成训练！')
    df = pd.DataFrame(data_eps)
    df.to_csv("90-results.csv", index=False)
    return rewards, ma_rewards


def test(cfg, env, agent):
    # 注意：测试的时候，就不用OU noise了，因为加噪声的目的只是为了让结果更robost
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')

    rewards = []
    # 记录所有回合的奖励
    ma_rewards = []

    # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset()[0]
        # 即observation
        done = False
        ep_reward = 0
        i_step = 0

        while not done:
            i_step += 1
            action = agent.choose_action(state)
            # 根据actor网络计算action
            # 注意：此时action的取值范围是[-1,1]，因为tanh是最后一层的激活函数
            next_state, reward, _, done, _ = env.step(action)
            # 由于之前算出来的action是[-1,1]
            # 但实际的action范围是[-2,2]，所以需要ActionWrapper来进行封装，使得action整体乘个2
            # 然后拿乘了2的action和环境做交互
            ep_reward += reward
            # 这一个episode的reward
            state = next_state

            # 测试的时候不用update的
        rewards.append(ep_reward)

        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        # 滑动平均奖励
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')

    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = Config()
    # 初始化一些环境和算法变量

    ########################### 训练部分   ##################################
    env, agent = env_agent_config(cfg, seed=1)
    # 配置环境和agent
    # agent是DDPG

    rewards, ma_rewards = train(cfg, env, agent)
    # 训练DDPG

    make_dir(cfg.result_path, cfg.model_path)
    # 创建result的路径和model的路径

    agent.save(path=cfg.model_path)
    # 由于决策的时候只需要actor,所以我们保存parameter的时候，只需要保存actor的参数即可

    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    # 将训练的结果rewards和ma_rewards保存下来
    plot_rewards(rewards, ma_rewards, cfg, tag="train")
    # 将训练的结果rewards和ma_rewards画出来，并保存

    ########################### 训练部分   ##################################

    ########################### 测试部分   ##################################
    env, agent = env_agent_config(cfg, seed=10)
    # 换一个随机种子，生成一个环境
    agent.load(path=cfg.model_path)
    # 将训练的actor参数load进来
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test', path=cfg.result_path)
    # 将测试的结果rewards和ma_rewards保存下来
    plot_rewards(rewards, ma_rewards, cfg, tag="test")
    # 将测试的结果rewards和ma_rewards画出来，并保存

    ########################### 测试部分   ##################################


