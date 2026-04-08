# 基于脑电信号的运动想象脑机接口系统

> **本科毕业设计** — 山东师范大学 计算机科学与技术专业

完整的运动想象（Motor Imagery, MI）脑机接口系统：在多个公开数据集上进行三条分类流水线的离线训练与评估，并通过基于 WebSocket 的在线仿真系统实现闭环反馈。

## 项目概述

对**左手 (769)** 与 **右手 (770)** 运动想象脑电信号进行二分类。在统一评估框架下实现并对比三条流水线：

| 流水线 | 类别 | 说明 |
|--------|------|------|
| **CSP + LDA** | 基线方法 | 共空间模式 + 线性判别分析 |
| **FBCSP + SVM** | 改进方法 | 滤波器组共空间模式（互信息特征选择）+ 支持向量机 |
| **EEGNet** | 深度学习 | 面向 EEG 的紧凑卷积神经网络（Lawhern et al., 2018） |

在三个数据集上验证泛化能力：

| 数据集 | 被试数 | 通道数 | 采样率 |
|--------|--------|--------|--------|
| BCI Competition IV 2a | 9 | 22 | 250 Hz |
| BCI Competition IV 2b | 9 | 3 (C3/Cz/C4) | 250 Hz |
| PhysioNet EEGBCI | 109（使用 20 名） | 64 | 160 Hz |

## 项目结构

```text
├── src/                        # Python 核心包
│   ├── data/                   #   数据加载器（2a、2b、PhysioNet）
│   ├── preprocessing/          #   带通滤波、CAR、ICA、分段
│   ├── features/               #   CSP、FBCSP 特征提取
│   ├── models/                 #   LDA、SVM、EEGNet 分类器
│   ├── evaluation/             #   评估指标、交叉验证、统计检验
│   ├── visualization/          #   图表生成、脑地形图、ERD/ERS
│   ├── online/                 #   WebSocket 服务器、数据回放
│   └── utils/                  #   配置加载、日志、路径管理
├── scripts/                    # 命令行入口
│   ├── train.py                #   训练与交叉验证
│   ├── evaluate.py             #   多方法对比与 LaTeX 表格生成
│   ├── analyze.py              #   论文图表生成
│   ├── run_online.py           #   启动在线仿真系统
│   └── download_data.py        #   通过 MNE 下载数据集
├── configs/                    # YAML 实验配置
│   ├── default.yaml            #   全局默认参数
│   ├── datasets/               #   各数据集参数覆盖
│   └── experiments/            #   各实验参数覆盖
├── web_frontend/               # 浏览器端在线仿真界面
├── tests/                      # pytest 测试套件
├── docs/                       # 毕业论文（LaTeX）及参考文献
├── results/                    # 实验输出（不纳入版本控制）
├── python_backend/             # 早期原型脚本（已归档）
├── environment.yml             # Conda 环境定义
└── CLAUDE.md                   # AI 助手项目指引
```

### 配置系统

三层 YAML 合并：`configs/default.yaml` → `configs/datasets/*.yaml` → `configs/experiments/*.yaml`。每一层覆盖上一层的同名参数，实验配置只需指定与默认值不同的部分。

## 快速开始

### 1. 环境搭建

```bash
conda env create -f environment.yml
conda activate thesis

# 若环境已存在：
conda env update -f environment.yml --prune
```

将 `.env.example` 复制为 `.env`，按需设置 `MNE_DATA`（默认为 `~/mne_data`）。

### 2. 下载数据

```bash
python scripts/download_data.py                    # 下载 BCI IV 2a + 2b
python scripts/download_data.py --physionet-eegbci  # 同时下载 PhysioNet
```

### 3. 训练与评估

```bash
# 训练单条流水线（配置驱动）
python scripts/train.py --config configs/experiments/fbcsp_svm.yaml

# 指定单个被试
python scripts/train.py --config configs/experiments/fbcsp_svm.yaml --subject 1

# 多方法对比，生成 LaTeX 表格
python scripts/evaluate.py results/ --latex

# 生成论文图表
python scripts/analyze.py results/ figures/
```

### 4. 在线仿真

在线系统通过 WebSocket 将保存的试次数据回放至训练好的模型，在浏览器中提供实时视觉反馈。

```bash
# 启动 WebSocket 后端（端口 8765）+ HTTP 服务器（端口 8080）
python scripts/run_online.py --model results/fbcsp_svm_2a/models/fbcsp_svm_2a_sub1.pkl
```

在浏览器中打开 `http://localhost:8080`。闭环流程为：**提示 → 想象 → 分类 → 反馈 → 休息**。

### 5. 运行测试

```bash
python -m pytest tests/ -v
```

## 关键技术参数

- **频段范围**：8–30 Hz（Mu 节律 8–13 Hz，Beta 节律 13–30 Hz）
- **分段窗口**：−0.5 s 至 3.0 s；训练裁剪：0.5 s 至 2.5 s
- **评估方式**：10 折分层交叉验证
- **评估指标**：准确率、Cohen's κ、加权 F1、ROC-AUC、混淆矩阵
- **统计检验**：Friedman 检验、Wilcoxon 符号秩检验、配对置换检验

## 主要依赖

- **mne** — 脑电数据加载、滤波、分段、ICA
- **scikit-learn** — CSP、LDA/SVM、交叉验证
- **torch** — EEGNet 深度学习模型
- **websockets** — 在线仿真 WebSocket 服务器
- **matplotlib** — 论文图表生成

## 开发规范

- **Git 提交**：遵循 [Conventional Commits](https://www.conventionalcommits.org/)（`feat`、`fix`、`docs`、`refactor`、`test`、`chore`）
- **Python 风格**：PEP 8，全面使用类型提示
- **语言**：代码注释使用英文；论文及面向用户的文档使用简体中文
- **禁止提交**：`.env`、`data/MNE-*`、`results/`、`*.pkl`、`*.npz`

## 许可

本项目为本科毕业设计，仅用于学术目的。
