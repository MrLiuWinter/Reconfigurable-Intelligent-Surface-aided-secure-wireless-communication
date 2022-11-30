#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/10/14 
@email:2218330483@qq.com
@function： 
"""
import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time


def save_results(res_dic, tag='train', path=None):
    ''' 保存奖励
    '''
    Path(path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res_dic)
    df.to_csv(f"{path}/{tag}_results.csv", index=False)
    print('Results saved！')


# figure = "power"
i = 20
# figurename = str(i)
# path = curr_path + "/outputs/" + f"{str(20)}_power/" + figure


def compute_avg(reward):
    avg_reward = []
    data = []
    for i in range(len(reward)):
        data.append(reward[i])
        avg_reward.append(np.mean(data[-500:]))
    return avg_reward


# 读取 csv 数据
df = pd.read_csv("results.csv")
# print(df.shape)
# plt.plot(np.linspace(0, 9999, 10000), np.array(df)[0], "r-", label=f"{i}eposide first")
# plt.xlabel("steps")
# plt.ylabel("Rewards")
# plt.title("channel")
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()
for i in range(60, 80):
    plt.plot(np.linspace(0, 9999, 10000), np.array(df)[i], "r-", label=f"{i}eposide first")
    # plt.plot(np.linspace(0, 9999, 10000), compute_avg(np.array(df)[0] + 1), "b-", label="Instant rewards, Pmax=20dBm")
    plt.xlabel("steps")
    plt.ylabel("Rewards")
    plt.title("channel")
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

