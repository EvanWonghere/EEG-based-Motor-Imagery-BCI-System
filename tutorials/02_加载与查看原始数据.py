# -*- coding: utf-8 -*-
"""
教程 02：加载与查看原始数据
============================
目标：用 MNE 加载 PhysioNet EEGBCI 数据，理解 Raw 对象、通道、采样率。
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
# 一、数据来源说明
# ---------------------------------------------------------------------------
# PhysioNet EEGBCI：受试者戴 EEG 帽，按屏幕提示做“想象左手/右手/脚”等，同时记录 EEG。
# 数据格式为 .edf。运行下面代码时，MNE 会自动下载到 ~/mne_data（首次较慢）。
#
# subject=1 表示 1 号受试者；runs=[6,10,14] 是“手 vs 脚”运动想象任务。

print("正在加载 PhysioNet EEGBCI 数据（首次运行会自动下载）...")
raw_fnames = eegbci.load_data(subjects=1, runs=[6, 10, 14])

# 每个 run 一个文件，合并成一条连续记录
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

# ---------------------------------------------------------------------------
# 二、标准化通道名并设置电极位置（便于后续滤波与可视化）
# ---------------------------------------------------------------------------
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

# 把注释里的 T1/T2 改成更直观的 hands/feet（本数据集是手 vs 脚）
raw.annotations.rename(dict(T1="hands", T2="feet"))

# ---------------------------------------------------------------------------
# 三、查看“原始数据”里有什么
# ---------------------------------------------------------------------------
print("\n========== 原始数据 (Raw) 关键信息 ==========")
print("采样率 (sfreq):", raw.info["sfreq"], "Hz  →  每秒采样点数")
print("通道数:", len(raw.ch_names), "  →  每个通道是一条时间序列")
print("通道名示例:", raw.ch_names[:5], "...")
print("数据总时长:", raw.times[-1] - raw.times[0], "秒")
print("标注 (annotations) 含义: 实验里标记了哪些事件 →", raw.annotations.description)
print()

# 可选：绘制通道位置（需要图形界面，无 GUI 时可能报错，可注释掉）
# raw.plot_sensors(show=True)

print("教程 02 结束。下一讲：03_滤波_Mu与Beta节律.py —— 为什么要做 8–30 Hz 滤波。")
