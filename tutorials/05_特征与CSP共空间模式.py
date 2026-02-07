# -*- coding: utf-8 -*-
"""
教程 05：特征与 CSP（共空间模式）
=================================
目标：理解“特征”是什么、为什么需要降维，以及 CSP 如何用“空间投影”提取对分类有用的特征。
"""
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import mne
import numpy as np
from mne.decoding import CSP
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# ---------------------------------------------------------------------------
# 一、什么是“特征”？为什么不能直接把 (通道×时间) 塞给分类器？
# ---------------------------------------------------------------------------
# 一个试次的数据形状是 (n_channels, n_times)，例如 (64, 560)。若直接拉成一维，有 64*560 个数，
# 维度太高、样本数有限，容易过拟合，且包含大量噪声。
#
# “特征” = 从原始数据里提炼出的**少数几个有区分度的数字**。
# CSP 做的是：找到一组**空间滤波器**（对通道做线性组合），使得：
#   - 对“左手想象”的试次，某几个组合后的通道方差很大；
#   - 对“右手想象”的试次，方差分布不同。
# 这样，用“方差”作为特征，两类就分得开。

# ---------------------------------------------------------------------------
# 二、CSP 直观理解（不推公式）
# ---------------------------------------------------------------------------
# 1. 对每一类（如左手、右手），算该类所有试次的“通道协方差矩阵”的平均。
# 2. CSP 找出一组方向（空间滤波器），使得：
#    - 在其中一个方向上，类 A 的方差大、类 B 的方差小；
#    - 在另一个方向上，类 B 的方差大、类 A 的方差小。
# 3. 对新试次：用这些方向做投影，得到几个“新通道”，再对每个新通道算方差（或对数方差）→ 得到特征向量。
#
# 所以 CSP 输出的是：每个试次从 (n_channels, n_times) 变成 (n_components,) 个数字，例如 4 个。

# ---------------------------------------------------------------------------
# 三、准备数据（与 04 相同：加载 → 滤波 → Epoching）
# ---------------------------------------------------------------------------
print("加载并预处理数据...")
raw_fnames = eegbci.load_data(subject=1, runs=[6, 10, 14])
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.annotations.rename(dict(T1="hands", T2="feet"))
raw.filter(l_freq=8.0, h_freq=30.0, fir_design="firwin", skip_by_annotation="edge")

events, event_id = mne.events_from_annotations(raw)
target_event_id = {k: v for k, v in event_id.items() if k in ("hands", "feet")}
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
epochs = mne.Epochs(raw, events, target_event_id, tmin=-0.5, tmax=3.0, proj=True, picks=picks, baseline=None, preload=True)

# 可选：只取中间一段（运动想象最明显的时段）做特征，效果往往更好
epochs_train = epochs.copy().crop(tmin=0.5, tmax=2.5)
X = epochs_train.get_data(copy=True)   # (n_trials, n_channels, n_times)
y = epochs_train.events[:, -1]

print("输入 CSP 的数据形状 X:", X.shape, "，标签 y:", y.shape)

# ---------------------------------------------------------------------------
# 四、用 MNE 的 CSP：拟合并变换
# ---------------------------------------------------------------------------
# n_components=4：保留 4 个空间分量（2 个偏向类 A，2 个偏向类 B，通常成对使用）
# log=True：对方差取对数，使分布更接近正态，利于后续 LDA
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp.fit(X, y)

# 变换：每个试次从 (n_channels, n_times) → (n_components,)
X_csp = csp.transform(X)
print("\nCSP 变换后的特征形状:", X_csp.shape, "  （每个试次 4 个特征）")
print("这 4 个数字就是“对数方差”，将作为 LDA 的输入。")
print()
print("【小结】CSP 不关心时间细节，只关心各通道组合后的**方差**在不同类别间的差异，用方差作为特征。")
print()
print("教程 05 结束。下一讲：06_LDA线性判别分析.py —— 如何用这 4 个特征做二分类。")
