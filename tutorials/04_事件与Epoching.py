# -*- coding: utf-8 -*-
"""
教程 04：事件与 Epoching
=========================
目标：理解“事件 (event)”和“试次 (trial)”，学会用 MNE 做 Epoching，得到 (试次, 通道, 时间) 的数组。
"""
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# ---------------------------------------------------------------------------
# 一、什么是“事件”(event)？
# ---------------------------------------------------------------------------
# 实验中，受试者会在某个时刻看到提示（如“想象左手”），然后开始想象。
# 记录里会在该时刻打一个**标记**，这就是“事件”。
# 每个事件通常用 (采样点索引, 0, 事件ID) 表示，例如 (3200, 0, 3) 表示在第 3200 个采样点发生了“类型 3”的事件。
#
# event_id：把“类型编号”映射成可读名字，例如 3→"hands", 2→"feet"。

# ---------------------------------------------------------------------------
# 二、什么是 Epoching？
# ---------------------------------------------------------------------------
# 把连续 EEG 按每个事件**切出一小段**，例如事件前 0.5 秒到事件后 3 秒。
# 每一段 = 一个“试次 (trial)”= 一次想象的完整数据。
# 最终得到 3 维数组： (n_trials, n_channels, n_times)，这就是后续 CSP/LDA 的输入形状。

# ---------------------------------------------------------------------------
# 三、加载并滤波（同 02、03）
# ---------------------------------------------------------------------------
print("加载并滤波数据...")
raw_fnames = eegbci.load_data(subjects=1, runs=[6, 10, 14])
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.annotations.rename(dict(T1="hands", T2="feet"))
raw.filter(l_freq=8.0, h_freq=30.0, fir_design="firwin", skip_by_annotation="edge")

# ---------------------------------------------------------------------------
# 四、从“标注”里提取事件
# ---------------------------------------------------------------------------
events, event_id = mne.events_from_annotations(raw)
print("\n事件 ID 映射 (名字 → 数字):", event_id)

# 只保留我们关心的类别（本数据集：手 vs 脚；BCI IV 2a 则是 769/770 等）
target_event_id = {k: v for k, v in event_id.items() if k in ("hands", "feet")}
print("本教程使用的类别:", target_event_id)

# ---------------------------------------------------------------------------
# 五、Epoching：切段
# ---------------------------------------------------------------------------
# tmin, tmax：相对“事件时刻”的时间窗（秒）。例如 tmin=-0.5, tmax=3 表示从事件前 0.5s 到事件后 3s
tmin, tmax = -0.5, 3.0
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

epochs = mne.Epochs(
    raw,
    events,
    target_event_id,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,  # 也可用 (tmin, 0) 做基线校正
    preload=True,
)

# ---------------------------------------------------------------------------
# 六、取出 (试次, 通道, 时间) 的数组和标签
# ---------------------------------------------------------------------------
# 这是和 sklearn / CSP 对接的标准格式
X = epochs.get_data(copy=True)   # shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]        # 每个试次对应的类别 ID（数字）

print("\n========== Epoching 结果 ==========")
print("X 形状 (试次, 通道, 时间点):", X.shape)
print("y 形状 (标签，每个试次一个):", y.shape)
print("y 的取值（类别 ID）:", np.unique(y))
print("\n每一行 y[i] 对应 X[i] 这段数据属于哪一类（如 hands / feet）。")
print()
print("教程 04 结束。下一讲：05_特征与CSP共空间模式.py —— 如何从多通道里提炼出少数“空间特征”。")
