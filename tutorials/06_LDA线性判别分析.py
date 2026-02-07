# -*- coding: utf-8 -*-
"""
教程 06：LDA —— 线性判别分析
=============================
目标：理解 LDA 是什么、如何用 CSP 得到的特征做“左手 vs 右手”的二分类。
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, ShuffleSplit
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# ---------------------------------------------------------------------------
# 一、LDA 直观理解
# ---------------------------------------------------------------------------
# 现在我们每个试次有 4 个特征（CSP 的 4 个对数方差），在 4 维空间里每个点代表一个试次。
# LDA 的目标：找到一条**直线**（或超平面），把两类点尽可能分开。
#
# 数学上：LDA 假设两类都是高斯分布、协方差相同，找最优投影方向使得“类间距离大、类内距离小”。
# 对新样本：投影到该方向，根据投影值是否超过某个阈值判断类别。
#
# 优点：简单、稳定、可解释；适合小样本、高维特征（如 CSP 只有 4 维）。

# ---------------------------------------------------------------------------
# 二、CSP + LDA 流水线
# ---------------------------------------------------------------------------
# 数据 → CSP 变换 → 得到 4 维特征 → LDA 分类。
# sklearn 的 Pipeline 可以把多步串起来：fit 时依次拟合并传递数据，predict 时依次变换再预测。

from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 三、准备数据（同 05）
# ---------------------------------------------------------------------------
print("加载并预处理数据...")
raw_fnames = eegbci.load_data(subjects=1, runs=[6, 10, 14])
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
epochs_train = epochs.copy().crop(tmin=0.5, tmax=2.5)
X = epochs_train.get_data(copy=True)
y = epochs_train.events[:, -1]

# ---------------------------------------------------------------------------
# 四、构建 Pipeline：CSP → LDA
# ---------------------------------------------------------------------------
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([("CSP", csp), ("LDA", lda)])

# ---------------------------------------------------------------------------
# 五、交叉验证：评估泛化能力
# ---------------------------------------------------------------------------
# 不把所有数据拿来训练再测，而是多次“划分训练/测试”，取平均准确率，更可靠
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
print("\n10 折交叉验证准确率:", scores.round(2))
print("平均准确率: {:.2%}".format(scores.mean()))

# 用全部数据再训练一遍，得到“最终模型”（可用于后续预测或保存）
clf.fit(X, y)
print("\n已用全部数据拟合 CSP+LDA，clf 可用于对新试次 predict。")

# ---------------------------------------------------------------------------
# 六、单次预测示例
# ---------------------------------------------------------------------------
# 取一个试次，形状 (1, n_channels, n_times)，预测类别
one_trial = X[0:1]
pred = clf.predict(one_trial)[0]
print("\n示例：第 1 个试次预测为类别", pred, "，真实标签为", y[0])
print()
print("【小结】LDA 根据 CSP 特征做线性划分；CSP+LDA 是 MI-BCI 最常用的基线方法。")
print()
print("教程 06 结束。下一讲：07_完整流水线_从数据到预测.py —— 从 Raw 到预测一条龙。")
