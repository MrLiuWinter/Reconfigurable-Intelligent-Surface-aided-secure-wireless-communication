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
import random
import math
import numpy as np


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
sin_ang = (- np.pi / 2) + np.pi * random.random()


def aN(N, theta):
    a = []
    for i in range(N):
        a.append(cmath.exp(-1j * np.pi * i * np.sin(theta)))
    return np.reshape(np.matrix(a), (N, 1))


def Los(N, sin_ang, pha_aoa):
    steering_aoa = []
    for i in range(0, N, 1):
        steering_aoa.append(cmath.exp(-1j * np.pi * (np.floor(i / 4) * sin_ang * np.sin(pha_aoa) + (
                i - np.floor(i / 4) * 4) * sin_ang * np.cos(pha_aoa))))
    return np.reshape(steering_aoa, (N, 1))


def Nlos(N, M):
    curl = np.sqrt(1 / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
    return curl


def normalization(a, M, N, power_t, max_action):
    power_t = np.power(10, (power_t / 10))
    G_real = a[:, :M]
    G_imag = a[:, M:2 * M]
    G = G_real.reshape(G_real.shape[0], M, 1) + 1j * G_imag.reshape(G_imag.shape[0], M, 1)
    GG_H = []
    for i in range(G_real.shape[0]):
        GG_H.append(np.real(np.trace(G[i].conjugate().T @ G[i])))
    cur_power_t = np.sqrt(np.array(GG_H)).reshape(-1, 1)

    Phi_real = a[:, -2 * N:-N]
    Phi_imag = a[:, -N:]
    real_normal = np.sum(np.abs(Phi_real)).reshape(-1, 1) * np.sqrt(2)
    imag_normal = np.sum(np.abs(Phi_imag)).reshape(-1, 1) * np.sqrt(2)  # F范数用于矩阵，二范数用于向量；矩阵各元素的n次方和开n根

    # Normalize the transmission power and phase matrix
    current_power_t = cur_power_t.repeat(2 * M, 1) / np.sqrt(power_t)

    real_normal = real_normal.repeat(N, 1)
    imag_normal = imag_normal.repeat(N, 1)

    division_term = np.hstack((current_power_t, real_normal, imag_normal))

    action = max_action * a / division_term
    return action


def whiten(state):  # Z-score normalization
    # prior to being input to network, the state s will go through a whitening process,
    # to remove the correlation between the entries of the state s.
    state = (state - np.mean(state)) / np.std(state)
    return state


def Path_loss(M, N):
    # Los of BS -RIS  np.mean([np.pi * random.random() for _ in range(500)])
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


def SNR(w, hr, shifts, G, hd, sigma, sigma_I):
    H = (hr.getH() @ shifts @ G + hd.getH()) @ w
    signal = np.abs(H) ** 2

    H_real = np.linalg.norm(H).reshape(1, -1) ** 2
    hr_real, hr_imag = np.real(hr).reshape(1, -1), np.imag(hr).reshape(1, -1)  # N * 2
    hd_real, hd_imag = np.real(hd).reshape(1, -1), np.imag(hd).reshape(1, -1)  # M * 2
    G_real, G_imag = np.real(G).reshape(1, -1), np.imag(G).reshape(1, -1)  # M * N * 2
    power_r = np.hstack((H_real, G_real, G_imag, hr_real, hr_imag, hd_real, hd_imag))
    # power_r = H_real

    snr = signal.item() / sigma  # (w/w)dB
    rate = np.log2(1 + snr)  # bit/s
    return rate, power_r


class channel_env(object):
    def __init__(self, M, N, Pmax, P_I, sigma, sigma_I):
        self.Pmax = Pmax
        self.P_I = P_I
        self.sigma = sigma
        self.sigma_I = sigma_I
        self.BS = M
        self.RIS = N
        self.a_dim = 2 * (N + M)
        self.state_dim = self.a_dim + 3 + (M * N * 2 + M * 2 + N * 2) * 2  #
        self.action_bound = 1  # 复数
        self.phi = np.eye(self.RIS, dtype=complex)
        self.G = None
        self.hdu = None
        self.hde = None
        self.hru = None
        self.hre = None
        self.episode_t = None

    def step(self, action):
        self.episode_t += 1
        done = False
        action = normalization(action, self.BS, self.RIS, self.Pmax, self.action_bound)

        # beamforming
        w_real = action[0, :self.BS]
        w_imag = action[0, self.BS:2 * self.BS]

        # phase shift angles
        Phi_real = action[0, -2 * self.RIS:-self.RIS]
        Phi_imag = action[0, -self.RIS:]

        w = w_real.reshape(self.BS, 1) + 1j * w_imag.reshape(self.BS, 1)
        phi = np.eye(self.RIS, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(w.conjugate().T @ w).reshape(1, -1) ** 2

        G, hdu, hde, hru, hre = self.G, self.hdu, self.hde, self.hru, self.hre
        rate_u, power_u = SNR(w, hru, phi, G, hdu, self.sigma, self.sigma_I)
        rate_e, power_e = SNR(w, hre, phi, G, hde, self.sigma, self.sigma_I)
        reward = max((rate_u - rate_e), 0)

        state = np.hstack((action, power_t, power_u, power_e))
        state = whiten(state)
        return state, reward, done

    def reset(self):
        self.episode_t = 0

        # channel state information
        w = np.ones((self.BS, 1)) * np.sqrt(np.power(10, (self.Pmax / 10)) / self.BS)
        self.G, self.hdu, self.hde, self.hru, self.hre = Path_loss(self.BS, self.RIS)
        u, power_u = SNR(w, self.hru, self.phi, self.G, self.hdu, self.sigma, self.sigma_I)
        e, power_e = SNR(w, self.hre, self.phi, self.G, self.hde, self.sigma, self.sigma_I)

        w_action = np.hstack((np.real(w.reshape(1, -1)), np.imag(w.reshape(1, -1))))
        phi_action = np.hstack((np.real(np.diag(self.phi)).reshape(1, -1), np.imag(np.diag(self.phi)).reshape(1, -1)))
        init_action = np.hstack((w_action, phi_action))

        # tr{G_H G} K*M @ M*K
        power_t = np.real(w.conjugate().T @ w).reshape(1, -1) ** 2

        # state
        state = np.array(np.hstack((init_action, power_t, power_u, power_e)))
        state = whiten(state)
        return state

    def sample_action(self):
        return np.random.rand(self.a_dim).reshape(1, -1) - 0.5


if __name__ == '__main__':
    sigmas = np.power(10, -110 / 10)  # AWGN
    sigmas_I = np.power(10, -80 / 10)  # active RIS thermal noise
    env = channel_env(4, 16, 10, 0, sigmas, 0)  # BS antenns, RIS elements, Power of BS, dB of RIS, noise, thermal noise
    states = env.reset()
    actions = env.sample_action()
    _, sec, _ = env.step(actions)
    print(sec)
