#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/09/24 
@email:2218330483@qq.com
@function： 
"""
import cmath
import numpy as np
import random
import math


def aN(N, theta):
    a = []
    for i in range(N):
        a.append(cmath.exp(-1j * np.pi * i * np.sin(theta)))
    return np.reshape(np.matrix(a), (N, 1))


def Los(N, sin_ang, pha_aoa):
    steering_aoa = []
    for i in range(1, N + 1, 1):
        steering_aoa.append(cmath.exp(-1j * np.pi * (np.floor(i / 4) * sin_ang * np.sin(pha_aoa) + (
                i - np.floor(i / 4) * 4) * sin_ang * np.cos(pha_aoa))))
    return np.reshape(steering_aoa, (N, 1))


def Nlos(N, M):
    z = []
    for i in range(1):
        curl = (np.sqrt(1 / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)))
        z.append(curl)
    return np.mean(z, 0)


k_r = 10  # reflect link rician factor
k_d = 0  # direction link rician factor
xBS = 0
yBS = 0
xIRS = 50
yIRS = 0
xk = 50
yk = 2
xeve = 45
yeve = 2
d_r = np.sqrt((xBS - xIRS) ** 2 + (yBS - yIRS) ** 2)
d_u = np.sqrt((xBS - xk) ** 2 + (yBS - yk) ** 2)
d_e = np.sqrt((xBS - xeve) ** 2 + (yBS - yeve) ** 2)
dr_u = np.sqrt((xIRS - xk) ** 2 + (yIRS - yk) ** 2)
dr_e = np.sqrt((xIRS - xeve) ** 2 + (yIRS - yeve) ** 2)

# large -scale
PL_0 = np.power(10, (-30 / 10))
pathloss_br = np.sqrt(PL_0 * math.pow(d_r, (-2.2)))
pathloss_bu = np.sqrt(PL_0 * math.pow(d_u, (-3.6)))
pathloss_ru = np.sqrt(PL_0 * math.pow(dr_u, (-2.2)))
pathloss_be = np.sqrt(PL_0 * math.pow(d_e, (-3.6)))
pathloss_re = np.sqrt(PL_0 * math.pow(dr_e, (-2.2)))
# angle
ang_G = np.arctan((yIRS - yBS) / (xIRS - xBS))
ang_u = np.arctan(abs(xk - xIRS) / (yIRS - yk))
ang_e = np.arctan(abs(xeve - xIRS) / (yIRS - yeve))


def Path_loss(M, N):
    # Los of BS -RIS
    ang_aod_bu = ang_G
    theta_aoa_bu = 0
    steering_aod_br = aN(M, ang_aod_bu)
    steering_aoa_br = aN(N, theta_aoa_bu)
    h_los = steering_aoa_br * steering_aod_br.getH()
    h_nlos = Nlos(N, M)
    G = np.matrix((np.sqrt(k_r / (k_r + 1)) * h_los + np.sqrt(1 / 1 + k_r) * h_nlos) * pathloss_br)

    # los of RIS - user
    sin_ang_ru = ang_u
    pha_aoa_ru = np.pi / 6
    h_los = Los(N, sin_ang_ru, pha_aoa_ru)
    h_nlos = Nlos(N, 1)
    hru = np.matrix((np.sqrt(k_r / (k_r + 1)) * h_los + np.sqrt(1 / 1 + k_r) * h_nlos) * pathloss_ru)

    # los of ris - eve
    sin_ang_re = ang_e
    pha_aoa_re = np.pi / 6
    h_los = Los(N, sin_ang_re, pha_aoa_re)
    h_nlos = Nlos(N, 1)
    hre = np.matrix((np.sqrt(k_r / (k_r + 1)) * h_los + np.sqrt(1 / 1 + k_r) * h_nlos) * pathloss_re)

    # bs - user
    hdu = np.matrix(pathloss_bu * Nlos(M, 1))
    hde = np.matrix(pathloss_be * Nlos(M, 1))
    return G, hdu, hde, hru, hre


# change phase shift
def Phase_shifts(angles, P_I):
    angles_ = []
    action = []
    for i in range(len(angles)):
        angles_.append(cmath.exp(1j * angles[i]))
        action.append(angles[i])
    shifts = np.diag(angles_) * np.power(10, (P_I / 20))
    return action, shifts  # phase shifts


def SNR(w, hr, shifts, G, hd, sigma, sigma_I):  # Pmax dBm
    H = (hr.getH() @ shifts @ G + hd.getH()) @ w
    signal = np.linalg.norm(H) ** 2

    H_dyna = np.linalg.norm(shifts @ hr) ** 2
    dynamic = H_dyna * sigma_I  # dB * mw

    snr = (1 / (sigma + dynamic)) * signal  # (w/w)dB
    rate = np.log2(1 + snr)  # bit/s
    return rate.item()


class channel_env(object):
    def __init__(self, M, N, Pmax, P_I, sigma, sigma_I):
        self.Pmax = Pmax
        self.P_I = P_I
        self.sigma = sigma
        self.sigma_I = sigma_I
        self.BS = M
        self.state_dim = N + 1
        self.action_dim = N
        self.action_bound = [0, np.pi * 2]
        self.G = None
        self.hdu = None
        self.hde = None
        self.hru = None
        self.hre = None
        self.goal = 0
        self.action = 2 * np.pi * np.random.rand(self.action_dim)

    def step(self, action):
        done = False
        # normalizeaction  scale action
        self.action += action
        # self.action = (action + 1) * np.pi
        self.action %= np.pi * 2

        angles, shifts = Phase_shifts(self.action, self.P_I)
        G, hdu, hde, hru, hre = self.G, self.hdu, self.hde, self.hru, self.hre

        # MRT beamforming for user
        H = hru.getH() @ shifts @ G + hdu.getH()
        w = (H.getH() / np.linalg.norm(H)) * np.sqrt(np.power(10, (self.Pmax / 10)))  # mw  M * 1

        snr_u = SNR(w, hru, shifts, G, hdu, self.sigma, self.sigma_I)
        snr_e = SNR(w, hre, shifts, G, hde, self.sigma, self.sigma_I)
        reward = max(snr_u - snr_e, 0)
        angles.append(reward)
        state = np.array(angles)
        return state, reward, done

    def reset(self):
        angles, shifts = Phase_shifts(self.action, self.P_I)
        # channel generate and reset the environment
        self.G, self.hdu, self.hde, self.hru, self.hre = Path_loss(self.BS, self.action_dim)
        w = np.ones((self.BS, 1)) * np.sqrt(np.power(10, (self.Pmax / 10)) / self.BS)  # mw  M * 1
        shifts = np.eye(self.action_dim)
        snr_u = SNR(w, self.hru, shifts, self.G, self.hdu, self.sigma, self.sigma_I)
        snr_e = SNR(w, self.hre, shifts, self.G, self.hde, self.sigma, self.sigma_I)
        reward = max(snr_u - snr_e, 0)
        angles.append(reward)
        state = np.array(angles)
        return state

    def sample_action(self):
        return np.random.rand(self.action_dim) - 0.5


if __name__ == '__main__':
    sigmas = np.power(10, -110 / 10)  # AWGN
    sigmas_I = np.power(10, -110 / 10)  # active RIS thermal noise
    env = channel_env(4, 16, -20, 0, sigmas, 0)  # BS antenns, RIS elements, Power of BS, dB of RIS, noise, thermal noise
    states = env.reset()
    actions = env.sample_action()
    _, sec, _ = env.step(actions)
    print(sec)
