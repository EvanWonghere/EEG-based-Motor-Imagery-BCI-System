# -*- coding: utf-8 -*-
"""
教程 03：滤波 —— Mu 与 Beta 节律
=================================
目标：理解为什么做 8–30 Hz 带通滤波，以及如何在 MNE 里实现。
"""
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# ---------------------------------------------------------------------------
# 一、为什么要滤波？
# ---------------------------------------------------------------------------
# 原始 EEG 里混杂了：
#   - 工频干扰（50/60 Hz）
#   - 眼电、肌电等伪迹（低频或高频）
#   - 我们关心的“运动想象”相关节律：主要在 **Mu (8–13 Hz)** 和 **Beta (13–30 Hz)**
#
# 运动皮层活动时，Mu/Beta 会在对应脑区出现“去同步化”（能量变化）。
# 所以第一步：用**带通滤波器**只保留 8–30 Hz，既去掉直流和工频，又突出运动想象相关频段。

# ---------------------------------------------------------------------------
# 二、加载数据（同教程 02）
# ---------------------------------------------------------------------------
print("加载数据...")
raw_fnames = eegbci.load_data(subject=1, runs=[6, 10, 14])
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.annotations.rename(dict(T1="hands", T2="feet"))

# ---------------------------------------------------------------------------
# 三、施加 8–30 Hz 带通滤波
# ---------------------------------------------------------------------------
# l_freq=8, h_freq=30：只保留 8 到 30 Hz 的成分
# fir_design='firwin'：用 FIR 滤波器设计，相位特性较好
# skip_by_annotation='edge'：在数据边缘处不滤波，避免边界效应
l_freq, h_freq = 8.0, 30.0
raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", skip_by_annotation="edge")

print(f"\n已施加带通滤波：{l_freq}–{h_freq} Hz（Mu + Beta 节律）。")
print("滤波后数据仍保存在 raw 对象中，后续 Epoching 会基于滤波后的信号。")
print()
print("【小知识】")
print("  Mu 节律 (8–13 Hz)：感觉运动皮层静息时明显，运动或运动想象时会减弱（去同步）。")
print("  Beta 节律 (13–30 Hz)：与运动准备、维持相关。")
print()
print("教程 03 结束。下一讲：04_事件与Epoching.py —— 如何把连续数据切成“一段段试次”。")
