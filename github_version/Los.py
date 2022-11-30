#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:winter
@version:
@time: 2022/11/07 
@email:2218330483@qq.com
@function： 
"""
import numpy as np


def frequencytowavelength(frequency):
    return 3000000000 / frequency


def getPositionUninformLinearArray(Nt, ae_spacing, height):
    x = np.arange(Nt) * ae_spacing
    x -= np.mean(x)
    y = np.zeros(Nt)
    z = np.repeat(np.array(height), Nt)
    return x, y, z


def getPositionsRectangular2d(Nae, ae_spacing, height):
    sq = np.floor(np.sqrt(Nae))
    x = np.arange(Nae) % sq
    y = np.floor(np.arange(Nae) / sq)
    z = np.repeat(np.array(height), Nae)
    x *= ae_spacing
    y *= ae_spacing
    x -= np.mean(x)
    y -= np.mean(y)
    return x, y, z


def randn_c(Nr, Nt):
    return (np.random.randn(Nr, Nt) + np.random.randn(Nr, Nt) * 1j) / np.sqrt(2)


def idealricianchannel(K_db, wavelength, tx, ty, tz, rx, ry, rz):
    # IT(int):时隙
    IT = 1
    K = 10 ** (K_db / 10.0)
    Nt = len(tx)
    Nr = len(rx)
    r = np.zeros((Nr, Nt), dtype=complex)
    for n in range(Nr):
        for m in range(Nt):
            r[n][m] = np.sqrt(np.square(rx[n] - tx[m]) + np.square(ry[n] - ty[m]) + np.square(rz[n] - tz[m]))
    anHlos = np.exp(-1j * 2.0 * np.pi / wavelength * r)
    # print(anHlos)
    Hlos = np.tile(anHlos.T, IT).T
    channelmatrix = np.sqrt(K / (1.0 + K)) * Hlos + randn_c(Nr, Nt) / np.sqrt(K + 1.0)
    return channelmatrix


def Rician(Nr, Nt, K):
    K = K
    dis = 10
    wavelength = frequencytowavelength(3 * 10 ** 9)
    Nt, Nr, ae_spacing, distance_tx_rx = Nt, Nr, wavelength / 2, dis
    tx, ty, tz = getPositionUninformLinearArray(Nt, ae_spacing, 0)
    rx, ry, rz = getPositionUninformLinearArray(Nr, ae_spacing, distance_tx_rx)
    H = idealricianchannel(K, wavelength, tx, ty, tz, rx, ry, rz)
    return H


print(Rician(16, 4, 10))
