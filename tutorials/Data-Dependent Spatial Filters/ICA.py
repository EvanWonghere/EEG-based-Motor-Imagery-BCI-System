import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 0. 准备数据：合成两个互相独立的非高斯源信号 (对应笔记：S)
# ==========================================
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 源信号 1：正弦波
s1 = np.sin(2 * time)
# 源信号 2：方波 (非高斯性极强)
s2 = np.sign(np.sin(3 * time))
S = np.vstack((s1, s2)) # 真实源矩阵 S, shape: (2, 2000)

# 混合矩阵 A (模拟脑电的容积传导)
A = np.array([[1.0, 1.0], 
              [0.5, 2.0]])
X = A @ S # 观测信号 X = AS, shape: (2, 2000)

# ==========================================
# Step 1: 中心化
# ==========================================
X_mean = np.mean(X, axis=1, keepdims=True)
X_c = X - X_mean

# ==========================================
# Step 2: 纯手写白化 (Whitening) —— 严格对应你的笔记公式！
# ==========================================
# Cov(X) = E D E^T; use eigh for symmetric matrix (real eigenvalues)
Cov_X = (X_c @ X_c.T) / n_samples
eigenvalues, E = np.linalg.eigh(Cov_X)

# 公式: V = D^{-1/2} E^T
D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
V = D_inv_sqrt @ E.T

# 白化后的数据: Z = V X
Z = V @ X_c

# ==========================================
# Step 3: 调用 sklearn 完成最后的“正交旋转”(寻找独立成分)
# ==========================================
# 因为你自己手写固定点迭代太长了，我们这里用算法库完成最后一步
ica = FastICA(n_components=2, whiten='arbitrary-variance', random_state=42)
_fit = ica.fit_transform(X_c.T)
S_reconstructed = np.asarray(_fit).T  # shape: (2, 2000)

# ==========================================
# Visualization: ellipse -> sphere -> separated axes
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
theta = np.linspace(0, 2 * np.pi, 200)

# Figure 1: observed = ellipse (tilted, correlated)
axes[0].scatter(X_c[0, :], X_c[1, :], s=5, alpha=0.5, color='coral', label='Mixed $X$')
# 2-sigma ellipse from Cov(X): shows "elliptical" shape
ellipse_xy = E @ np.diag(2 * np.sqrt(eigenvalues)) @ np.array([np.cos(theta), np.sin(theta)])
axes[0].plot(ellipse_xy[0, :], ellipse_xy[1, :], 'darkred', lw=2, label='2-$\\sigma$ ellipse')
axes[0].set_title('1. Observed $X$ (mixed)\nEllipse: correlated, unequal variance', fontsize=12)
axes[0].set_aspect('equal')
axes[0].set_xlabel('Channel 1')
axes[0].set_ylabel('Channel 2')
axes[0].legend(loc='upper right', fontsize=9)

# Figure 2: whitened = sphere (circle in 2D), Cov(Z)=I
axes[1].scatter(Z[0, :], Z[1, :], s=5, alpha=0.5, color='royalblue', label='Whitened $Z$')
# Unit circle (2-sigma): whitening "squashes" ellipse into a circle
axes[1].plot(2 * np.cos(theta), 2 * np.sin(theta), 'darkblue', lw=2, label='2-$\\sigma$ circle')
axes[1].set_title('2. Whitened $Z = VX$\nSphere (circle): Cov($Z$) = $I$', fontsize=12)
axes[1].set_aspect('equal')
axes[1].set_xlabel('Whitened dim 1')
axes[1].set_ylabel('Whitened dim 2')
axes[1].legend(loc='upper right', fontsize=9)

# Figure 3: recovered = axis-aligned rectangle (sources separated)
S_reconstructed /= np.std(S_reconstructed, axis=1, keepdims=True)
axes[2].scatter(S_reconstructed[0, :], S_reconstructed[1, :], s=5, alpha=0.5, color='forestgreen', label='$\\hat{S}$')
# Axes through origin: IC1 vs IC2 are independent (rectangular cloud)
axes[2].axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
axes[2].axvline(0, color='gray', ls='--', lw=1, alpha=0.7)
# Optional: 2-sigma rectangle to emphasize "axis-aligned box"
r = 2.0
axes[2].plot([-r, r, r, -r, -r], [-r, -r, r, r, -r], 'darkgreen', lw=2, label='2-$\\sigma$ box')
axes[2].set_title('3. Recovered $\\hat{S}$ (ICA)\nSources separated (IC1 $\\perp$ IC2)', fontsize=12)
axes[2].set_aspect('equal')
axes[2].set_xlabel('IC 1 (channel 1 recovered)')
axes[2].set_ylabel('IC 2 (channel 2 recovered)')
axes[2].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# 验证白化后的协方差矩阵是否为单位矩阵 I
print("白化后数据的协方差矩阵 Cov(Z) (应近似为单位矩阵 I):")
print(np.round((Z @ Z.T) / n_samples, 4))