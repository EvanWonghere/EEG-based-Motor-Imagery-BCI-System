# 运动想象 (MI) BCI 算法教程 —— 学习指南

本目录是一套**从零理解**运动想象脑机接口 Python 流水线的教程，面向对信号处理与机器学习不熟悉的同学。建议按编号顺序学习。

---

## 学习路线图

| 序号 | 文件 | 内容概要 |
| ------ | ------ | ---------- |
| 01 | `01_什么是运动想象与EEG.py` | 运动想象是什么、EEG 信号从哪来、本项目的目标 |
| 02 | `02_加载与查看原始数据.py` | 用 MNE 加载 PhysioNet 数据、看通道与采样率 |
| 03 | `03_滤波_Mu与Beta节律.py` | 为什么做 8–30 Hz 带通滤波、Mu/Beta 节律与运动皮层 |
| 04 | `04_事件与Epoching.py` | 事件 (event)、试次 (trial)、Epoching 切段 |
| 05 | `05_特征与CSP共空间模式.py` | 什么是空间特征、CSP 如何提取、方差与投影 |
| 06 | `06_LDA线性判别分析.py` | 什么是 LDA、如何用 CSP 特征做二分类（CSP+LDA Pipeline） |
| 07 | `07_完整流水线_从数据到预测.py` | 串联：加载 → 滤波 → Epoch → CSP → LDA → 预测 |
| 08 | `08_LSL仿真简介.py` | LSL 是什么、如何把“预测结果”发给 Unity 做仿真 |

---

## 运行前准备

1. 激活 conda 环境：`conda activate thesis`
2. 安装依赖：已包含在项目 `requirements.txt` 中（mne, numpy, scipy, scikit-learn, pylsl, python-dotenv 等）
3. 环境变量（可选）：在项目根目录复制 `.env.example` 为 `.env`，可设置 `MNE_DATA` 指定 MNE 数据集下载位置（默认 `~/mne_data`）。教程 02–07 运行时会自动加载 `.env`。
4. 部分脚本会**自动下载** PhysioNet EEGBCI 数据（约几十 MB），首次运行稍慢

---

## 建议

* 每讲先**通读注释**再运行代码，改改参数观察输出。
* 遇到不懂的 API 可查 [MNE 文档](https://mne.tools/stable/overview/index.html) 和 [scikit-learn 文档](https://scikit-learn.org/stable/user_guide.html)。
* 学完 07 后，可对照项目根目录下 `python_backend/` 中的模块（preprocessing、training、replay_stream）加深理解。
