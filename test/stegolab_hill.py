"""HILL of stegolab, reference implementation for the tests.

Taken from https://github.com/daniellerch/stegolab, HILL/HILL.py,
commit 051694f3de93, functions hill_cost, ternary_entropyf, calc_lambda
and embedding_simulator.
Modified: np.float replaced by float, removed in numpy 1.24,
and hide() takes and returns arrays instead of files.

Differs from conseal's HILL by design:
the residual is not divided by 4, i.e., the cost is 4 times lower,
and flat areas are regularized by 1 / (W1 + 1e-10) instead of clipping.
"""

# Copyright (c) 2020 Daniel Lerch Hostalot. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# This implementation is based on the paper:
# A New Cost Function for Spatial Image Steganography
# by Bin Li, Ming Wang, Jiwu Huang and Xiaolong Li.

import scipy.signal
import numpy as np


def hill_cost(I):
    HF1 = np.array([
        [-1, 2, -1],
        [ 2,-4,  2],
        [-1, 2, -1]
    ])
    H2 = np.ones((3, 3)).astype(float)/3**2
    HW = np.ones((15, 15)).astype(float)/15**2

    R1 = scipy.signal.convolve2d(I, HF1, mode='same', boundary='symm')
    W1 = scipy.signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm')
    rho=1./(W1+10**(-10))
    cost = scipy.signal.convolve2d(rho, HW, mode='same', boundary='symm')
    return cost

def ternary_entropyf(pP1, pM1):
    p0 = 1-pP1-pM1
    P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P*np.log2(P)
    eps = 2.2204e-16
    H[P<eps] = 0
    H[P>1-eps] = 0
    return np.sum(H)


def calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length+1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        pP1 = (np.exp(-l3 * rho_p1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        pM1 = (np.exp(-l3 * rho_m1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0
    iterations = 0
    alpha = float(message_length)/n
    # limit search to 30 iterations and require that relative payload embedded
    # is roughly within 1/1000 of the required relative payload
    while float(m1-m3)/n > alpha/1000.0 and iterations<300:
        lamb = l1+(l3-l1)/2
        pP1 = (np.exp(-lamb*rho_p1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        pM1 = (np.exp(-lamb*rho_m1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations = iterations + 1;
    return lamb



def embedding_simulator(x, rho_p1, rho_m1, m):
    n = x.shape[0]*x.shape[1]
    lamb = calc_lambda(rho_p1, rho_m1, m, n)
    pChangeP1 = (np.exp(-lamb * rho_p1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
    pChangeM1 = (np.exp(-lamb * rho_m1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
    y = x.copy()
    randChange = np.random.rand(y.shape[0], y.shape[1])
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1;
    y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1;
    return y



def hide(I, payload, T=5):

    rho = hill_cost(I)

    wet_cost = 10**10
    rho[np.isnan(rho)] = wet_cost
    rho[rho>wet_cost] = wet_cost

    rho_m1 = rho.copy()
    rho_p1 = rho.copy()
    rho_p1[I==255] = wet_cost
    rho_m1[I==0] = wet_cost


    stego = embedding_simulator(I, rho_p1, rho_m1, payload*I.shape[0]*I.shape[1])
    return stego
