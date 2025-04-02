
import conseal as cl
print(cl.__path__)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d



x0 = np.array(Image.open('campus512_gray.png'))

rhos = cl.ws._costmap.compute_cost(x0)
# # flip
# x0_bar = x0 ^ 1
# x0 = x0.astype('float32')

# # predict
# f_kb = np.array([
#     [-1, +2, -1],
#     [+2,  0, +2],
#     [-1, +2, -1],
# ], dtype='float32') / 4.
# x0_hat = convolve2d(
#     x0.astype('float32'), f_kb,
#     mode='same', boundary='symm'
# )

# # calculate local WS residuals
# betas_hat = (x0 - x0_bar) * (x0 - x0_hat)
# rhos = np.exp(-np.abs(betas_hat))
# rhos = 1 / np.abs(betas_hat)

# plt.imshow(rhos, cmap='gray')
# plt.show()


x1 = cl.lsb.simulate(x0, alpha=.1, locate=cl.LOCATION_SELECTED, rhos=rhos, seed=12345)
# x1 = x0 + delta
y = np.repeat(x0[..., None], 3, 2)
y[x0 != x1] = [255, 128, 0]

plt.imshow(y)
# plt.imshow(x1, cmap='gray')
plt.show()


exit()










import conseal as cl
from glob import glob
import jpeglib
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from tempfile import NamedTemporaryFile
from tqdm import tqdm




#
df = []
# for fname in glob('test/assets/cover/jpeg_75_gray/*.jpg'):
files = glob(str(Path(os.environ['DATA']) / 'data/alaska2/fabrika-2024-01-26/images_ahd/*.png'))[:1]
for fname in tqdm(files):
    x0 = np.array(Image.open(fname).convert('L'))
    rho_p1, rho_m1 = cl.hill.compute_cost_adjusted(x0)
    (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
        (rho_p1, rho_m1),
        .01,  # distortion
        n=x0.size,
        sender=cl.DLS,
    )
    cost_hat = (np.sum(p_p1 * rho_p1) + np.sum(p_m1 * rho_m1)) / x0.size
    H_hat = cl.tools.entropy(p_p1, p_m1) / x0.size

    print(fname, rho_p1.shape, rho_m1.shape, cost_hat, H_hat)




exit()

def attack(y1, qt):
    # cartesian calibration
    with NamedTemporaryFile(suffix='jpeg') as tmp:
        jpeglib.from_dct(Y=y1, qt=qt).write_dct(tmp.name)
        x1 = jpeglib.read_spatial(tmp.name).spatial
        x2 = x1[3:-5, 3:-5]  # crop 4x4
        jpeglib.from_spatial(x2).write_spatial(tmp.name, qt=jpeg0.qt)
        y2 = jpeglib.read_dct(tmp.name).Y

    # #
    # h1_01, _ = np.histogram(y1[:, :, 0, 1].flatten(), 8, range=(-4, 4))
    # h1_10, _ = np.histogram(y1[:, :, 1, 0].flatten(), 8, range=(-4, 4))
    # h1_11, _ = np.histogram(y1[:, :, 1, 1].flatten(), 8, range=(-4, 4))
    # h2_01, _ = np.histogram(y2[:, :, 0, 1].flatten(), 8, range=(-4, 4))
    # h2_10, _ = np.histogram(y2[:, :, 1, 0].flatten(), 8, range=(-4, 4))
    # h2_11, _ = np.histogram(y2[:, :, 1, 1].flatten(), 8, range=(-4, 4))
    # #
    # beta_01 = (h1_01[0+4] - h2_01[0+4]) / (h2_01[-1+4] + h2_01[1+4])
    # beta_10 = (h1_10[0+4] - h2_10[0+4]) / (h2_10[-1+4] + h2_10[1+4])
    # beta_11 = (h1_11[0+4] - h2_11[0+4]) / (h2_11[-1+4] + h2_11[1+4])

    #
    h1_01, _ = np.histogram(np.abs(y1[:, :, 0, 1]).flatten(), 3, range=(0, 3))
    h1_10, _ = np.histogram(np.abs(y1[:, :, 1, 0]).flatten(), 3, range=(0, 3))
    h1_11, _ = np.histogram(np.abs(y1[:, :, 1, 1]).flatten(), 3, range=(0, 3))
    h2_01, _ = np.histogram(np.abs(y2[:, :, 0, 1]).flatten(), 3, range=(0, 3))
    h2_10, _ = np.histogram(np.abs(y2[:, :, 1, 0]).flatten(), 3, range=(0, 3))
    h2_11, _ = np.histogram(np.abs(y2[:, :, 1, 1]).flatten(), 3, range=(0, 3))
    beta_01 = (h2_01[1] * (h1_01[0] - h2_01[0]) + (h1_01[1] - h2_01[1]) * (h2_01[2] - h2_01[1])) / (h2_01[1]**2 + (h2_01[2] - h2_01[1])**2)
    beta_10 = (h2_10[1] * (h1_10[0] - h2_10[0]) + (h1_10[1] - h2_10[1]) * (h2_10[2] - h2_10[1])) / (h2_10[1]**2 + (h2_10[2] - h2_10[1])**2)
    beta_11 = (h2_11[1] * (h1_11[0] - h2_11[0]) + (h1_11[1] - h2_11[1]) * (h2_11[2] - h2_11[1])) / (h2_11[1]**2 + (h2_11[2] - h2_11[1])**2)

    #
    beta_hat = np.mean([beta_01, beta_10, beta_11])
    beta_hat = np.clip(beta_hat, 0, None)
    return beta_hat


#
df = []
# for fname in glob('test/assets/cover/jpeg_75_gray/*.jpg'):
files = glob(str(Path(os.environ['DATA']) / 'data/alaska2/fabrika-2024-01-26/images_ahd/*.png'))[:200]
for fname in tqdm(files):
    x0 = np.array(Image.open(fname))
    with NamedTemporaryFile(suffix='jpeg') as tmp:
        jpeglib.from_spatial(x0).write_spatial(tmp.name, qt=75)
        jpeg0 = jpeglib.read_dct(tmp.name)
        y0, qt0 = jpeg0.Y, jpeg0.qt[0]

    #
    for alpha in [.4]:
        beta_hat = attack(y0, qt0)
        #
        y1_F5 = cl.F5.simulate_single_channel(y0=y0, alpha=alpha, seed=12345)
        beta_F5 = (y0 != y1_F5).sum() / cl.tools.nzAC(y0)
        beta_hat_F5 = attack(y1_F5, qt0)
        #
        y1_nsF5 = cl.nsF5.simulate_single_channel(y0=y0, alpha=alpha, seed=12345)
        beta_nsF5 = (y0 != y1_nsF5).sum() / cl.tools.nzAC(y0)
        beta_hat_nsF5 = attack(y1_nsF5, qt0)

    # print(beta_hat, beta, '|', np.abs(beta - beta_hat))
    # print('\n', np.abs(beta - beta_hat))

        df.append({
            'fname': Path(fname).name,
            'alpha': alpha,
            'beta_hat': beta_hat,
            'beta_F5': beta_F5,
            'beta_hat_F5': beta_hat_F5,
            'beta_nsF5': beta_nsF5,
            'beta_hat_nsF5': beta_hat_nsF5,
            'mae0': np.abs(0 - beta_hat),
            'mae1_F5': np.abs(beta_F5 - beta_hat_F5),
            'mae1_nsF5': np.abs(beta_nsF5 - beta_hat_nsF5),
        })

df = pd.DataFrame(df)
print(df)
print(df[['beta_hat', 'beta_F5', 'beta_hat_F5', 'beta_nsF5', 'beta_hat_nsF5', 'mae0', 'mae1_F5', 'mae1_nsF5']].mean())
# import numpy as np


# def soliton(
#     m: int = 100,
#     *,
#     robust: bool = True,
#     c: float = .1,
#     delta: float = .5,
# ) -> np.ndarray:
#     """

#     :param m:
#     :type m: int
#     :param c:
#     :type c: float
#     :param delta:
#     :type delta: float
#     :return:
#     :rtype: np.ndarray
#     """
#     # Soliton distribution
#     i = np.arange(1, m+1)
#     p = nu = np.concatenate([[1/m], 1/(i[1:]*i[:-1])])

#     # robust Soliton distribution
#     if robust:
#         T = c * np.log(m / delta) * np.sqrt(m)
#         mT = int(np.floor(m / T))
#         # print(m, mT)
#         assert m > mT
#         tau = np.concatenate([
#             T / (i[:mT-1] * m),
#             [T * np.log(T / delta) / m],
#             np.zeros(m - mT),
#         ])
#         p = nu + tau

#     #
#     p = (nu + tau)
#     return p / np.sum(p)


# def generate_H(
#     m: int,
#     n: int = None,
#     *,
#     c: float = .1,
#     delta: float = .5,
#     seed: int = None,
# ) -> np.ndarray:
#     """Generates parity-check matrix according to robust soliton distribution.

#     :param m: number of rows/message bits
#     :type m: int
#     :param n: number of columns/cover elements
#     :type n: int
#     :param c: constant parameter
#     :type c: float
#     :param delta: failure probability
#     :type delta: float
#     :return: parity-check matrix
#     :rtype: np.ndarray

#     :Example:

#     >>> # TODO
#     """
#     if n is None:
#         n = 2**m-1
#     # Get the robust Soliton distribution
#     rsd = soliton(m, c=c, delta=delta, robust=True)

#     # Sample column weights w[1], ..., w[n] from the RSD
#     rng = np.random.default_rng(seed)
#     column_weights = rng.choice(m, size=n, p=rsd) + 1

#     # Generate columns of H
#     H = np.zeros((m, n), dtype=int)
#     for j in range(n):
#         # Create a column with column_weights[j] ones
#         ones_positions = rng.choice(m, size=column_weights[j], replace=False)
#         H[ones_positions, j] = 1

#     return H

# # Parameters
# m = 100  # Number of rows
# # n = 10000  # Number of columns
# n = 1000
# c = .1  # Constant parameter
# delta = .05  # Failure probability

# # Generate the parity-check matrix
# H = generate_H(m, n=n, delta=delta, c=c, seed=12345)
# print("Generated Parity-Check Matrix:\n", H)

# #
# p_empirical = np.histogram(np.sum(H, axis=0), bins=m-1, range=(1, m), density=True)[0]
# p_theoretical = soliton(m=m, c=c, delta=delta)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(range(50), p_theoretical[:50], label="Theoretical RSD", linewidth=1)
# ax.plot(range(50), p_empirical[:50], label="Empirical RSD", linewidth=1, linestyle='dotted')
# plt.show()