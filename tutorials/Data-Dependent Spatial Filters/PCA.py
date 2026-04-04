import numpy as np

# ==========================================
# 0. 准备数据 (对应笔记：设数据集 X)
# ==========================================
np.random.seed(42)
n = 1000  # 样本数 n
d = 2     # 特征维度 d (为了方便可视化，设为2维)

# 生成一个有相关性的二维数据 (比如 x2 大致等于 0.5 * x1)
x1 = np.random.normal(50, 10, n)  # 均值50，标准差10
x2 = x1 * 0.5 + np.random.normal(20, 2, n)
X = np.column_stack((x1, x2)) # X 的 shape: (1000, 2)

print(f"原始数据 X shape: {X.shape}")
print(f"原始数据均值: {np.mean(X, axis=0)}")

# ==========================================
# Step 1: 中心化 Centering
# ==========================================
# 公式: \mu = 1/n * \sum x_i
mu = np.mean(X, axis=0)

# 公式: X_c = X - \mu
X_c = X - mu

print(f"中心化后均值 (应趋近于0): {np.mean(X_c, axis=0)}")

# ==========================================
# Step 2: 协方差矩阵 S
# ==========================================
# 公式: S = 1/n * X_c^T * X_c
S = (X_c.T @ X_c) / n

print("\n协方差矩阵 S:")
print(S)

# ==========================================
# Step 3: 特征值分解 (第一主成分推导结论)
# ==========================================
# 公式: Sw = \lambda w
eigenvalues, eigenvectors = np.linalg.eig(S)

# 将特征值从大到小排序，并同步调整特征向量的列
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
W = eigenvectors[:, idx] # W 就是我们要的投影矩阵

print(f"\n特征值 (方差): {eigenvalues}")
print("特征向量矩阵 W (每一列是一个主成分方向):")
print(W)

# ==========================================
# Step 4: SVD 法 (验证笔记结论)
# ==========================================
# 公式: X = U \Sigma V^T
U, Sigma, Vt = np.linalg.svd(X_c, full_matrices=False)
V = Vt.T  # numpy 返回的是 V^T，我们要转置回来拿到 V

# 验证特征值 = \Sigma^2 / n
svd_eigenvalues = (Sigma ** 2) / n

print("\n--- SVD 验证 ---")
print(f"SVD算出的特征值: {svd_eigenvalues}")
print("SVD算出的 V 矩阵 (比较一下上面的 W):")
print(V)
# 注：V 和 W 的某些列可能符号相反(乘了-1)，在空间中代表同一条直线的两个相反方向，完全等价！

# ==========================================
# Step 5: 降维与投影
# ==========================================
# 假设我们只想保留 1 维数据 (k=1)
w_1 = W[:, 0:1] # 取第一主成分 (最大特征值对应的列向量)

# 公式: z_i = w^T x_i (写成矩阵就是 Z = X_c @ W)
Z = X_c @ w_1

print(f"\n降维后的数据 Z shape: {Z.shape}")

# ==========================================
# Step 6: Visualization
# ==========================================
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Figure 1: centered data and principal component directions
ax1 = axes[0]
ax1.scatter(X_c[:, 0], X_c[:, 1], alpha=0.4, color='royalblue', label='Centered data $X_c$')

std_devs = np.sqrt(eigenvalues)
scale = 2

ax1.arrow(0, 0, W[0, 0] * std_devs[0] * scale, W[1, 0] * std_devs[0] * scale,
          head_width=2, head_length=3, fc='crimson', ec='crimson',
          linewidth=2.5, zorder=3, label='1st PC $w_1$ (PC1)')
ax1.arrow(0, 0, W[0, 1] * std_devs[1] * scale, W[1, 1] * std_devs[1] * scale,
          head_width=2, head_length=3, fc='forestgreen', ec='forestgreen',
          linewidth=2.5, zorder=3, label='2nd PC $w_2$ (PC2)')

ax1.set_aspect('equal')
ax1.set_title('PCA: axes of maximum variance', fontsize=14, pad=15)
ax1.set_xlabel('Feature 1 (centered)')
ax1.set_ylabel('Feature 2 (centered)')
ax1.legend()

# Figure 2: 1D projection onto first principal component
ax2 = axes[1]
Z = X_c @ W[:, 0]

ax2.hist(Z, bins=30, color='coral', edgecolor='black', alpha=0.8)
ax2.set_title('Projection onto $w_1$ (1D distribution)', fontsize=14, pad=15)
ax2.set_xlabel('Projected value $z_i$')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()