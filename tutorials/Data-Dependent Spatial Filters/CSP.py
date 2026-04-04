import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# Draw covariance ellipse (2-sigma) and scatter from data
# ==========================================
def draw_ellipse(ax, data, color, label, n_sigma=2):
    data = np.asarray(data)
    if data.shape[0] < 2:
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.4, color=color, label=label)
        return
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-10)
    theta = np.linspace(0, 2 * np.pi, 200)
    # Ellipse: mean + n_sigma * evecs @ sqrt(evals) @ [cos(t), sin(t)]
    xy = (evecs @ np.diag(n_sigma * np.sqrt(evals)) @ np.array([np.cos(theta), np.sin(theta)])).T + mean
    ax.fill(xy[:, 0], xy[:, 1], facecolor=color, alpha=0.15, edgecolor=color, lw=2)
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.4, color=color, label=label)


def draw_total_contour(ax, data, n_sigma=2, color='black', ls='--', lw=2, label='Total'):
    """Draw only the 2-sigma contour of combined data (no fill, no scatter)."""
    data = np.asarray(data)
    if data.shape[0] < 2:
        return
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-10)
    theta = np.linspace(0, 2 * np.pi, 200)
    xy = (evecs @ np.diag(n_sigma * np.sqrt(evals)) @ np.array([np.cos(theta), np.sin(theta)])).T + mean
    ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=ls, lw=lw, label=label)

# ==========================================
# 0. 准备数据：模拟两类脑电信号 (C = 2 通道)
# ==========================================
np.random.seed(42)
n_samples = 1000

# BCI-like: two classes with different covariance orientations.
# C1 + C2 must NOT be a multiple of I so the total contour is elliptic in sensor space.
# Class 1 (left hand): energy along one diagonal
mean1, cov1 = [0, 0], [[10, 6], [6, 8]]
X1 = np.random.multivariate_normal(mean1, cov1, n_samples).T  # (2, n_samples)

# Class 2 (right hand): energy along the other diagonal; C = C1+C2 has distinct eigenvalues
mean2, cov2 = [0, 0], [[10, -4], [-4, 8]]
X2 = np.random.multivariate_normal(mean2, cov2, n_samples).T  # (2, n_samples)

# ==========================================
# 严格按照你的笔记开始推导！
# ==========================================

# 计算两类的协方差矩阵 C1, C2
C1 = (X1 @ X1.T) / n_samples
C2 = (X2 @ X2.T) / n_samples

# Step 1: 总协方差 C
C = C1 + C2

# Step 2: 谱分解 C = E D E^T
# 注意：用 eigh 专门处理实对称矩阵，数值更稳定
eigenvalues_C, E = np.linalg.eigh(C) 

# Step 3: 白化矩阵 P = D^{-1/2} E^T
D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_C))
P = D_inv_sqrt @ E.T

# 验证白化效果 (此时 P @ C @ P.T 应该是单位矩阵 I)
print("验证 P C P^T (应为单位矩阵 I):")
print(np.round(P @ C @ P.T, 3))

# Step 4: 白化后的协方差 S1, S2
S1 = P @ C1 @ P.T
S2 = P @ C2 @ P.T

print("\n验证白化魔术 S1 + S2 = I:")
print(np.round(S1 + S2, 3))

# Step 5: 继续谱分解 S1 = B \Lambda B^T
eigenvalues_S1, B = np.linalg.eigh(S1)

# 将特征值降序排列 (最大化 Class 1)
idx = np.argsort(eigenvalues_S1)[::-1]
eigenvalues_S1 = eigenvalues_S1[idx]
B = B[:, idx]

print("\n验证同时对角化 (S1 的特征值 + S2 对应的特征值 = 1):")
eigenvalues_S2 = np.diag(B.T @ S2 @ B)
print(f"S1 的特征值: {np.round(eigenvalues_S1, 3)}")
print(f"S2 的特征值: {np.round(eigenvalues_S2, 3)}")

# Step 7: 合成最终空间滤波矩阵 W = B^T P
W = B.T @ P

# Step 8: 对原始数据进行 CSP 投影滤波
Z1 = W @ X1
Z2 = W @ X2

# ==========================================
# Visualization: sensor space -> whitened -> CSP latent space
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: raw sensor space (overlapping ellipses)
ax1 = axes[0]
X_all = np.vstack([X1.T, X2.T])
draw_total_contour(ax1, X_all, label='Total (ellipse)')
draw_ellipse(ax1, X1.T, 'coral', 'Class 1 (left hand)')
draw_ellipse(ax1, X2.T, 'royalblue', 'Class 2 (right hand)')
ax1.set_title('1. Sensor space\nTwo classes entangled', fontsize=13)
ax1.set_xlabel('Channel 1')
ax1.set_ylabel('Channel 2')
ax1.set_aspect('equal')
ax1.legend(loc='upper right', fontsize=9)

# Panel 2: whitened space (S1 + S2 = I, orthogonal cross)
ax2 = axes[1]
X1_whitened = P @ X1
X2_whitened = P @ X2
X_whitened_all = np.vstack([X1_whitened.T, X2_whitened.T])
draw_total_contour(ax2, X_whitened_all, label='Total (circle)')
draw_ellipse(ax2, X1_whitened.T, 'coral', 'Class 1 whitened')
draw_ellipse(ax2, X2_whitened.T, 'royalblue', 'Class 2 whitened')
ax2.set_title('2. Whitened space $Px$\n$S_1 + S_2 = I$ (orthogonal)', fontsize=13)
ax2.set_xlabel('Whitened dim 1')
ax2.set_ylabel('Whitened dim 2')
ax2.set_aspect('equal')
ax2.legend(loc='upper right', fontsize=9)

# Panel 3: CSP projection (variance split along axes)
ax3 = axes[2]
Z_all = np.vstack([Z1.T, Z2.T])
draw_total_contour(ax3, Z_all, label='Total')
draw_ellipse(ax3, Z1.T, 'coral', 'Class 1 (CSP)')
draw_ellipse(ax3, Z2.T, 'royalblue', 'Class 2 (CSP)')
ax3.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
ax3.axvline(0, color='gray', ls='--', lw=1, alpha=0.6)
ax3.set_title('3. CSP output $Z = WX$\nVariance split (axis-aligned)', fontsize=13)
ax3.set_xlabel('CSP dim 1')
ax3.set_ylabel('CSP dim 2')
ax3.set_aspect('equal')
ax3.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# Step 9: 提取送入分类器的特征 (log variance)
var_Z1 = np.var(Z1, axis=1)
var_Z2 = np.var(Z2, axis=1)

print("\n--- 最终提取的特征能量 (方差) 对比 ---")
print(f"Class 1 在 CSP 维度1的方差: {var_Z1[0]:.2f} (极大)  |  维度2的方差: {var_Z1[1]:.2f} (极小)")
print(f"Class 2 在 CSP 维度1的方差: {var_Z2[0]:.2f} (极小)  |  维度2的方差: {var_Z2[1]:.2f} (极大)")