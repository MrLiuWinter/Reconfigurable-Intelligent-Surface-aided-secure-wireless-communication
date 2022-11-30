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

# # 创建
# my_df = pd.DataFrame(
#     [['Biking', 68.5, 195, np.nan], ['Dancing', 83.1, 1984, 3]],  # 记录
#     columns=['hobby', 'weight', 'birthyear', 'children'],  # 字段(数据库)——属性(面向对象)
#     index=['alice', 'bob']  # 主键
# )
# # 保存
# my_df.to_csv('my_df.csv')
# # 加载
# my_df_loaded = pd.read_csv('my_df.csv', index_col=0)  # index_col=0 指定行索引的位置
# print(my_df_loaded)
# N = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# num_ = [8.5, 9.2, 10, 10.5, 10.8, 11.1, 11.3, 11.7, 11.9, 12.1, 12.2]
# num_1 = [10.2, 11.2, 12, 12.5, 12.7, 12.9, 13.1, 13.5, 13.7, 13.9, 14]
# num_2 = [12.1, 13.5, 14.1, 14.3, 14.5, 14.8, 15.1, 15.3, 15.4, 15.5, 15.6]
# num_pass = [6, 6.4, 6.8, 6.9, 7, 7.2, 7.3, 7.5, 7.6, 7.7, 7.8]
# plt.plot(N, num_, "r-", label="η$^{2}$=20dB")
# plt.plot(N, num_1, "b-", label="η$^{2}$=30dB")
# plt.plot(N, num_2, "y-", label="η$^{2}$=40dB")
# plt.plot(N, num_pass, "k-", label="Passive RIS")
# plt.xlabel("The number of elements at RIS(N)")
# plt.ylabel("secrecy Rate(bps/Hz)")
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()
# N = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# num_ = [8.5, 8.8, 9.0, 9.4, 9.7, 10, 10.3, 10.6, 10.9, 11.1, 11.2]
# num_1 = [9.5, 10.2, 10.5, 11.2, 11.7, 11.9, 12.1, 12.5, 12.7, 12.9, 13]
# num_2 = [11.5, 11.9, 12.5, 13.1, 13.4, 13.6, 14.1, 14.3, 14.4, 14.5, 14.6]
# num_pass = [7, 7.2, 7.3, 7.5, 7.8, 8, 8.3, 8.4, 8.6, 8.7, 9]
# plt.plot(N, num_, "r-", label="η$^{2}$=20dB")
# plt.plot(N, num_1, "b-", label="η$^{2}$=30dB")
# plt.plot(N, num_2, "y-", label="η$^{2}$=40dB")
# plt.plot(N, num_pass, "k-", label="Passive RIS")
# plt.xlabel("Power(dBm)")
# plt.ylabel("secrecy Rate(bps/Hz)")
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()

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


# # train result
# index = [20, 25, 30, 35, 40, 45]
# avg_rewards1 = [1, 2, 3, 4, 5, 6]  # Pmax, Ny, P_I, sigma_I
# avg_rewards2 = [2, 3, 4, 5, 6, 7]  # Pmax, Ny, P_I, sigma_I
# avg_rewards3 = [1, 2, 3, 4, 5, 6]  # Pmax, Ny, P_I, sigma_I
# avg_rewards4 = [2, 3, 4, 5, 6, 7]  # Pmax, Ny, P_I, sigma_I
# avg_rewards5 = [1, 2, 3, 4, 5, 6]  # Pmax, Ny, P_I, sigma_I
# avg_rewards6 = [2, 3, 4, 5, 6, 7]  # Pmax, Ny, P_I, sigma_I
# avg = [avg_rewards1, avg_rewards2, avg_rewards3, avg_rewards4, avg_rewards5, avg_rewards6]
# df = pd.DataFrame(avg)
# df.to_csv("figure_data")
# pd.DataFrame([]).to_csv('max_data.csv', index=False, header=False)

# figure = "power"
i = 20
# figurename = str(i)
# path = curr_path + "/outputs/" + f"{str(20)}_power/" + figure

def compute_avg_reward(reward):
    avg_reward = np.zeros_like(reward)

    for i in range(len(reward)):
        avg_reward[i] = np.sum(reward[:(i + 1)]) / (i + 1)

    return avg_reward


# 读取 csv 数据
df = pd.read_csv(f'results.csv')
# print(df)
# plt.plot(np.linspace(0, 199, 200), np.array(df)[-1][:200], "r-", label=f"{i}eposide first")
# plt.xlabel("steps")
# plt.ylabel("Rewards")
# plt.title("channel")
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()
plt.plot(np.linspace(0, len(np.array(df)[1])-1, len(np.array(df)[1])), np.array(df)[190], "r-", label=f"{i}eposide first")
# plt.plot(np.linspace(0, len(np.array(df)[1])-1, len(np.array(df)[1])), np.array(df)[30], "b^-", label=f"{i}eposide first")
# plt.plot(np.linspace(0, len(np.array(df)[1])-1, len(np.array(df)[1])), np.array(df)[80], "g*-", label=f"{i}eposide first")
plt.xlabel("steps")
plt.ylabel("Rewards")
plt.title("channel")
plt.legend(loc='lower right')
plt.grid()
plt.show()
