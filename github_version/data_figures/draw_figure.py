#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/10/14 
@email:2218330483@qq.com
@function： 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

# 设置matplotlib正常显示中文和符号
matplotlib.rcParams["font.sans-serif"] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False
# # 创建
# my_df = pd.DataFrame(
#     [['Biking', 68.5, 195, np.nan], ['Dancing', 83.1, 1984, 3]],  # 记录
#     columns=['hobby', 'weight', 'birthyear', 'children'],  # 字段(数据库)——属性(面向对象)
#     index=['alice', 'bob'] ) # 主键
# # 保存
# my_df.to_csv('my_df.csv')
# # 加载
# my_df_loaded = pd.read_csv('my_df.csv', index_col=0)  # index_col=0 指定行索引的位置
# print(my_df_loaded)


# # diff RIS number
# N = [16, 24, 32, 40, 48, 56, 64]
# num_no = [2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9]
# num_pass = [3.1, 3.25, 3.55, 3.75, 3.85, 4.1, 4.3]
# num_10 = [3.4, 4.3, 5.1, 5.6, 6, 6.15, 6.2]
# num_30 = [4.5, 5.9, 6.4, 6.9, 7.3, 7.45, 7.5]
# plt.plot(N, num_30, "r^-", label="η$^{2}$=30dB")
# plt.plot(N, num_10, "y*-", label="η$^{2}$=10dB")
# plt.plot(N, num_pass, "kx-", label="Passive RIS")
# plt.plot(N, num_no, "gp-", label="No RIS")
# plt.xlabel("RIS的反射单元个数(N)")
# plt.ylabel("保密率(bps/Hz)")
# plt.legend(loc='lower right')
# plt.grid()
# plt.savefig(f"diff-numbers.jpg", dpi=600, bbox_inches='tight')
# plt.close()
#
# # diff power
# N = [10, 15, 20, 25, 30, 35, 40]
# num_no = [2.9, 3.4, 3.8, 4.5, 4.8, 5.4, 5.8]
# num_pass = [3.1, 3.6, 4.1, 4.8, 5.1, 5.7, 6.2]
# num_10 = [4.2, 4.5, 5.2, 5.9, 6.2, 6.8, 7.1]
# num_30 = [5.5, 5.9, 6.6, 7.3, 7.6, 8.1, 8.5]
# plt.plot(N, num_10, "bs-", label="η$^{2}$=10dB")
# plt.plot(N, num_30, "yd-", label="η$^{2}$=30dB")
# plt.plot(N, num_pass, "kx-", label="Passive RIS")
# plt.plot(N, num_no, "gp-", label="No RIS")
# plt.xlabel("基站传输功率(dBm)")
# plt.ylabel("保密率(bps/Hz)")
# plt.legend(loc='lower right')
# plt.grid()
# plt.savefig(f"diff-power.jpg", dpi=600, bbox_inches='tight')
# plt.close()
#
# # MRT with ddpg
# N = [10, 15, 20, 25, 30, 35, 40]
# MRT_pass = [2, 2.4, 2.8, 3.5, 3.9, 4.4, 4.8]
# num_pass = [3.1, 3.6, 4.1, 4.8, 5.1, 5.7, 6.2]
# MRT_act = [3.9, 4.1, 4.8, 5.4, 6.1, 6.6, 6.9]
# num_act = [5.5, 5.9, 6.6, 7.3, 7.6, 8.1, 8.5]
# plt.plot(N, MRT_pass, "bs-", label="MRT pass")
# plt.plot(N, MRT_act, "yd-", label="MRT active")
# plt.plot(N, num_pass, "kp-", label="Joint pass")
# plt.plot(N, num_act, "g^-", label="Joint active")
# plt.xlabel("基站传输功率(dBm)")
# plt.ylabel("保密率(bps/Hz)")
# plt.legend(loc='lower right')
# plt.grid()
# plt.savefig(f"diff-ways.jpg", dpi=600, bbox_inches='tight')
# plt.close()


def compute_avg(reward):
    avg_reward = []
    data = []
    for i in range(len(reward)):
        data.append(reward[i])
        avg_reward.append(np.mean(data[-500:]))
    return avg_reward


# 读取 csv 数据
df = pd.read_csv(f'results.csv')
# best = [0, 80, 86, 94, 99]
# df = pd.read_csv("40-16-results.csv")
# best = [19, 17, 86, 90]
# for i in range(90, 100):
#     plt.plot(np.linspace(0, 9999, 10000), np.array(df)[i], "r-", label=f"{i}eposide first")
#     plt.xlabel("steps")
#     plt.ylabel("Rewards")
#     plt.title("channel")
#     plt.legend(loc='lower right')
#     plt.grid()
#     plt.show()

# # convergence
# plt.plot(np.linspace(0, 9999, 10000), np.array(df)[0] + 1, "r+-", label="Average rewards, Pmax=20dBm")
# plt.plot(np.linspace(0, 9999, 10000), compute_avg(np.array(df)[0] + 1), "b-", label="Instant rewards, Pmax=20dBm")
# plt.plot(np.linspace(0, 9999, 10000), np.array(df)[99], "g*-", label="Average rewards, Pmax=10dBm")
# plt.plot(np.linspace(0, 9999, 10000), compute_avg(np.array(df)[99]), "k--", label="Instant rewards, Pmax=10dBm")
# plt.xlabel("Steps")
# plt.ylabel("Rewards")
# plt.legend(loc='lower right')
# plt.grid()
# plt.savefig(f"convergence.jpg", dpi=600, bbox_inches='tight')
# plt.close()
