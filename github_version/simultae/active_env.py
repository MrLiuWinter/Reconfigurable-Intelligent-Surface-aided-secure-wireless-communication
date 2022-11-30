#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/09/28 
@email:2218330483@qq.com
@function： 
"""
import numpy as np
import math
import cmath


k = 5
xBS = 0
yBS = 0
xIRS = 40
yIRS = 40
xk = 90
yk = 20
xeve = 70
yeve = 20

d_u = np.sqrt((xBS - xk) ** 2 + (yBS - yk) ** 2)
d_e = np.sqrt((xBS - xeve) ** 2 + (yBS - yeve) ** 2)
d_r = np.sqrt((xBS - xIRS) ** 2 + (yBS - yIRS) ** 2)
dr_u = np.sqrt((xIRS - xk) ** 2 + (yIRS - yk) ** 2)
dr_e = np.sqrt((xIRS - xeve) ** 2 + (yIRS - yeve) ** 2)

# departure angles
angRu = np.arccos(abs(yk - yIRS) / dr_u)
angRe = np.arccos(abs(yeve - yIRS) / dr_e)
angBR = np.arccos(xIRS / d_r)

# large -scale
PL_0 = np.power(10, (-30 / 10))
pathloss_br = np.sqrt(PL_0 * math.pow(d_r, (-2.2)))
pathloss_bu = np.sqrt(PL_0 * math.pow(d_u, (-3.8)))
pathloss_ru = np.sqrt(PL_0 * math.pow(dr_u, (-2.2)))
pathloss_be = np.sqrt(PL_0 * math.pow(d_e, (-3.5)))
pathloss_re = np.sqrt(PL_0 * math.pow(dr_e, (-2.2)))


def aN(N, theta):
    a = []
    for i in range(N):
        a.append(cmath.exp(1j * np.pi * i * np.sin(theta)))
    return np.reshape(np.matrix(a), (N, 1))


def Nlos(N, M):
    curl = np.sqrt(1 / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
    return curl


def Path_loss(M, N):
    # Los of BS -RIS  np.mean([np.pi * random.random() for _ in range(500)])
    steering_aod_br = aN(M, np.pi/2 - angBR)
    steering_aoa_br = aN(N, angBR)
    h_los = steering_aoa_br * steering_aod_br.getH()
    h_nlos = Nlos(N, M)
    G = np.matrix((np.sqrt(k / (k + 1)) * h_los + np.sqrt(1 / 1 + k) * h_nlos) * pathloss_br)

    # los of RIS - user
    h_los = aN(N, angRu)
    h_nlos = Nlos(N, 1)
    hru = np.matrix((np.sqrt(k / (k + 1)) * h_los + np.sqrt(1 / 1 + k) * h_nlos) * pathloss_ru)

    # los of ris - eve
    h_los = aN(N, angRe)
    h_nlos = Nlos(N, 1)
    hre = np.matrix((np.sqrt(k / (k + 1)) * h_los + np.sqrt(1 / 1 + k) * h_nlos) * pathloss_re)

    # bs - user
    hdu = np.matrix(pathloss_bu * Nlos(M, 1))
    hde = np.matrix(pathloss_be * Nlos(M, 1))
    return G, hdu, hde, hru, hre


def SNR(w, hr, shifts, G, hd, sigma, sigma_I):  # Pmax dBm
    H = (hr.getH() @ shifts @ G + hd.getH()) @ w
    signal = np.abs(H) ** 2

    H_real = np.linalg.norm(H).reshape(1, -1) ** 2 / sigma
    hr_real, hr_imag = np.real(hr).reshape(1, -1), np.imag(hr).reshape(1, -1)  # N * 2
    hd_real, hd_imag = np.real(hd).reshape(1, -1), np.imag(hd).reshape(1, -1)  # M * 2
    G_real, G_imag = np.real(G).reshape(1, -1), np.imag(G).reshape(1, -1)  # M * N * 2
    power_r = np.hstack((H_real, G_real, G_imag, hr_real, hr_imag, hd_real, hd_imag))
    power_r = H_real

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
        self.state_dim = self.a_dim + 3  # + (M * N * 2 + M * 2 + N * 2) * 2  # (2 * receive + transmit) + action
        self.action_bound = 1
        self.w = np.ones((self.BS, 1)) * np.sqrt(np.power(10, (self.Pmax / 10)) / self.BS)
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
        return state, reward, done

    def reset(self):
        self.episode_t = 0

        # channel state information
        self.G, self.hdu, self.hde, self.hru, self.hre = Path_loss(self.BS, self.RIS)
        u, power_u = SNR(self.w, self.hru, self.phi, self.G, self.hdu, self.sigma, self.sigma_I)
        e, power_e = SNR(self.w, self.hre, self.phi, self.G, self.hde, self.sigma, self.sigma_I)

        w_action = np.hstack((np.real(self.w.reshape(1, -1)), np.imag(self.w.reshape(1, -1))))
        phi_action = np.hstack((np.real(np.diag(self.phi)).reshape(1, -1), np.imag(np.diag(self.phi)).reshape(1, -1)))
        init_action = np.hstack((w_action, phi_action))

        # tr{G_H G} K*M @ M*K
        power_t = np.real(self.w.conjugate().T @ self.w).reshape(1, -1) ** 2

        # state
        state = np.array(np.hstack((init_action, power_t, power_u, power_e)))
        return state

    def sample_action(self):
        return np.random.rand(self.a_dim).reshape(1, -1) - 0.5


if __name__ == '__main__':
    sigma = np.power(10, -95 / 10)  # AWGN
    sigma_I = np.power(10, -95 / 10)  # active RIS thermal noise
    env = channel_env(4, 16, 30, 0, sigma, 0)  # BS antenns, RIS elements, Power of BS, dB of RIS, noise, thermal noise
    state = env.reset()
    action = env.sample_action()
    _, sec, _ = env.step(action)
    print(sec)
