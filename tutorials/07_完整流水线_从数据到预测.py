# -*- coding: utf-8 -*-
"""
教程 07：完整流水线 —— 从数据到预测
=====================================
目标：把 02–06 的内容串成一条龙：加载 → 滤波 → Epoching → CSP → LDA → 交叉验证与预测。
读完本讲，你就掌握了本项目 python_backend 的核心逻辑。
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# =============================================================================
# 步骤 1：加载原始数据
# =============================================================================
print("【步骤 1】加载 PhysioNet EEGBCI 数据...")
raw_fnames = eegbci.load_data(subjects=1, runs=[6, 10, 14])
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)
raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
raw.annotations.rename(dict(T1="hands", T2="feet"))

# =============================================================================
# 步骤 2：带通滤波 8–30 Hz（Mu + Beta）
# =============================================================================
print("【步骤 2】带通滤波 8–30 Hz...")
raw.filter(l_freq=8.0, h_freq=30.0, fir_design="firwin", skip_by_annotation="edge")

# =============================================================================
# 步骤 3：事件与 Epoching
# =============================================================================
print("【步骤 3】提取事件并 Epoching...")
events, event_id = mne.events_from_annotations(raw)
target_event_id = {k: v for k, v in event_id.items() if k in ("hands", "feet")}
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
epochs = mne.Epochs(raw, events, target_event_id, tmin=-0.5, tmax=3.0, proj=True, picks=picks, baseline=None, preload=True)
# 取运动想象明显时段
epochs_crop = epochs.copy().crop(tmin=0.5, tmax=2.5)
X = epochs_crop.get_data(copy=True)
y = epochs_crop.events[:, -1]
print("   X.shape =", X.shape, ", y.shape =", y.shape)

# =============================================================================
# 步骤 4：CSP 特征提取 + LDA 分类（Pipeline）
# =============================================================================
print("【步骤 4】构建 CSP + LDA 流水线并交叉验证...")
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([("CSP", csp), ("LDA", lda)])
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
print("   交叉验证准确率: {:.2%} (±{:.2%})".format(scores.mean(), scores.std()))

# 全量拟合，得到最终模型
clf.fit(X, y)

# =============================================================================
# 步骤 5：对新试次做预测（模拟“在线”一帧）
# =============================================================================
print("【步骤 5】用训练好的模型预测一个试次...")
trial = X[0:1]   # 形状 (1, n_channels, n_times)
pred_label = clf.predict(trial)[0]
true_label = y[0]
print("   预测类别:", pred_label, "  真实类别:", true_label)
print()

# =============================================================================
# 小结：与 python_backend 的对应关系
# =============================================================================
print("========== 与项目 python_backend 的对应 ==========")
print("  preprocessing.py  → 步骤 2（滤波）、步骤 3（Epoching）")
print("  training.py       → 步骤 4（CSP+LDA、保存/加载模型）")
print("  train_model.py     → 本脚本的 1–4 + 保存 .joblib 与 replay_data.npz")
print("  replay_stream.py   → 循环：取试次 → 步骤 5 预测 → 通过 LSL 发送结果给 Unity")
print()
print("教程 07 结束。下一讲：08_LSL仿真简介.py —— 如何把预测结果“流式”发给 Unity。")
