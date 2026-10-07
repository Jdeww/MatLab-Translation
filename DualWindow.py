"""
Dual window detection
Ref: "Multiple-Window Anomaly Detection for Hyperspectral Imagery"
     by W.M. Liu, C.I. Chang
RXD with variable size window

Python translation of DualWindow.m
"""

import time

import numpy as np
import matplotlib.pyplot as plt

start = time.time()

# ------------------------------------------------------------------
# with autoencoder
# f = np.loadtxt('hymap-deepenc4_400.csv', delimiter=','); M=100; N=300; L=113; D=f

f = np.loadtxt("F_random_anom3.csv", delimiter=",")
M, N = 86, 86
D = f
L = 100

gt = np.loadtxt("gt_random_anom3.csv", delimiter=",").astype(bool)

MN, L = D.shape

mx_sz = (M, N)
X = np.zeros((M, N, L))
for i in range(L):
    # MATLAB reshape() fills column-major (Fortran) order
    X[:, :, i] = D[:, i].reshape(mx_sz, order="F")

# ------------------------------------------------------------------

# HyMap
OuterWindowSz = 11
InnerWindowSz = 1

OuterRadius = (OuterWindowSz - 1) // 2
InnerRadius = (InnerWindowSz - 1) // 2
COutWin = OuterWindowSz * OuterWindowSz
CInWin = InnerWindowSz * InnerWindowSz
DiffRadius = OuterWindowSz - InnerWindowSz
detRXD = np.zeros((M, N))

rng = np.random.default_rng()

for i in range(OuterRadius, M - OuterRadius):
    if (i + 1) % 10 == 0:
        print(f"Please wait ... i={i + 1} out of {M}")

    for j in range(OuterRadius, N - OuterRadius):
        CM_outer = np.zeros((COutWin, L))
        CM_inner = np.zeros((CInWin, L))

        # Find anomaly
        for l in range(L):
            M_outer = X[i - OuterRadius:i + OuterRadius + 1,
                        j - OuterRadius:j + OuterRadius + 1, l]
            M_inner = X[i - InnerRadius:i + InnerRadius + 1,
                        j - InnerRadius:j + InnerRadius + 1, l]

            CM_outer[:, l] = M_outer.reshape(COutWin, order="F")
            CM_inner[:, l] = M_inner.reshape(CInWin, order="F")

        if InnerWindowSz > 1:
            K_outer = np.cov(CM_outer, rowvar=False)
            K_inner = np.cov(CM_inner, rowvar=False)
            u_outer = CM_outer.mean(axis=0)
            u_inner = CM_inner.mean(axis=0)
            K_diff = K_outer - K_inner
            u_diff = (u_outer - u_inner).reshape(-1, 1)

            # DWEST: use greatest 6 eigenvectors
            evals, evecs = np.linalg.eigh(K_diff)
            z = range(L - 6, L)
            v = evecs[:, z].sum(axis=1).reshape(-1, 1)
            detRXD[i, j] = (v.T @ u_diff).item()
        else:
            r = X[i, j, :].copy()
            r = r + rng.random(r.shape) * 0.7
            K_outer = np.cov(CM_outer, rowvar=False)
            K_inv = np.linalg.inv(K_outer)
            u = CM_outer.mean()
            U = np.full(L, u)
            diff = r - U
            detRXD[i, j] = diff @ K_inv @ diff
            # detRXD[i] = (one - U) @ K_inv @ (r - U)  # UTD = 0

print(f"Please wait ... i={M} out of {M}")

plt.figure()
plt.imshow(np.abs(detRXD), cmap="gray")
plt.axis("image")
plt.axis("off")
plt.title(f"Outer window sz={OuterWindowSz}, Inner window sz={InnerWindowSz}")

detRXD = detRXD.reshape(1, MN, order="F")

# ------------------------------------------------------------------
# Produce ROC

x = detRXD
x_max = x.max()
x_min = x.min()
x_denom = x_max - x_min
x = (x - x_min) / x_denom * 1000
x = x.reshape((M, N), order="F")

nd = np.zeros(1000)
nf = np.zeros(1000)
thr = np.zeros(1000)

gt_count = np.count_nonzero(gt)
not_gt_count = np.count_nonzero(~gt)

for i in range(1000):
    b = x > i
    nd[i] = np.count_nonzero(b & gt)
    nf[i] = np.count_nonzero(b & ~gt)
    thr[i] = i

pd = nd / gt_count
pf = nf / not_gt_count

plt.figure()
plt.plot(pf, pd)
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.xlabel("False Alarms")
plt.ylabel("Detected")
plt.title("ROC")

elapsed = time.time() - start
print(f"Elapsed time: {elapsed:.4f} seconds")

trapezoid = getattr(np, "trapezoid", None) or np.trapz
area = 1 + trapezoid(pd, pf)
print(f"area = {area}")

plt.show()
