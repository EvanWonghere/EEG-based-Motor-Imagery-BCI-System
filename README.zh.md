# 基于脑电信号的运动想象系统 (EEG-Based Motor Imagery BCI System)

> **本项目为计算机专业本科毕业设计。**
> 核心目标：基于公开数据集构建离线分类模型，并通过 LSL 协议驱动 Unity 进行交互式仿真验证。

## 📖 项目简介

本项目旨在实现一个完整的运动想象（Motor Imagery, MI）脑机接口系统。由于不依赖实时脑电采集设备，系统采用**“离线数据驱动 + 在线仿真”**的架构。

1. **Python 后端**：负责读取 BCI Competition IV 2a 数据集，进行预处理、CSP 特征提取、分类模型训练，并通过 LSL 广播模拟实时数据流。
2. **Unity 前端**：作为 LSL 接收端，根据接收到的分类结果（如左手/右手想象）控制虚拟场景中的对象运动，提供可视化反馈。

## 🛠 技术栈

### 1. Python 后端 (数据处理与流式传输)

* **语言版本**: Python 3.8+
* **核心库**:
  * `mne`: EEG 数据加载（GDF/FIF格式）、滤波、Epoching。
  * `scikit-learn`: CSP (共空间模式) 特征提取、LDA/SVM 分类器。
  * `numpy`/`scipy`: 数值计算与信号处理。
  * `pylsl`: Lab Streaming Layer 协议，用于发送模拟的数据流或控制指令。

### 2. Unity 前端 (交互仿真)

* **引擎版本**: Unity 2021.3+ (LTS)
* **开发语言**: C#
* **插件**: `LSL4Unity` (用于接收 LSL 数据流)。

## 📂 项目结构

```text
Project_Root/
├── data/                   # 存放 BCI Competition IV 2a 数据集 (.gdf)
├── docs/                   # 毕业论文与引用文献
│   ├── thesis/             # 本项目的毕业设计论文
│   └── citations/          # 引用文献（PDF 等）
├── models/                 # 训练好的模型 (.joblib) 与 replay_data.npz
├── python_backend/         # Python 源代码
│   ├── preprocessing.py    # 滤波、去伪迹
│   ├── training.py         # CSP + 分类器训练
│   ├── replay_stream.py    # LSL 数据回放/仿真脚本 (核心交互入口)
│   ├── train_model.py      # 训练入口脚本
│   ├── download_datasets.py # CLI：下载 BCI IV 2a/2b / PhysioNet EEGBCI
│   ├── test_datasets.py    # 测试已下载数据集是否能正确加载
│   ├── datasets.py         # 数据集下载逻辑（MNE_DATA、MOABB）
│   ├── utils.py            # 工具函数
│   └── archive/            # 旧版脚本 (prototype, test_*)
├── unity_frontend/         # Unity 项目工程目录
│   └── Assets/
│       ├── Scripts/        # C# 脚本 (LSLReceiver.cs, GameController.cs)
│       └── Scenes/         # 仿真场景
├── tutorials/               # 可选学习脚本
├── environment.yml         # Conda 环境 thesis（pip 依赖同 requirements.txt）
├── .env.example             # 环境变量示例（复制为 .env 并填写，勿提交 .env）
├── requirements.txt
└── README.md
```

**论文与引用**：毕业论文请放在 `docs/thesis/`，引用的文章放在 `docs/citations/`。详见 `docs/README.md`。

## 🔄 系统工作流 (Pipeline)

1. **离线训练阶段**:
   * 加载 `.gdf` 数据 -> 8-30Hz 带通滤波 -> 提取 Epochs (基于 Event ID: 769, 770 等)。
   * 运行 CSP 算法提取空间特征。
   * 训练 LDA 分类器并评估准确率。
   * 保存 CSP 滤波器和 LDA 模型。
2. **在线仿真阶段 (Pseudo-Online)**:
   * **Sender (Python)**: 读取测试集数据，模拟实时采样率，通过 `pylsl` 将特征或预测结果推送到局域网。
   * **Receiver (Unity)**: 监听 LSL 端口，获取分类标签。
   * **Feedback**: Unity 根据标签执行逻辑（例如：收到"Left" -> 虚拟手向左移动）。

## ⚠️ 给 AI 助手的特别说明 (Context for AI)

* **无需硬件代码**: 本项目**不涉及**真实的 EEG 硬件连接（如 OpenBCI、NeuroScan）。所有“实时”功能均通过重放（Replay）数据集实现。
* **数据集**: 默认使用 **BCI Competition IV 2a** (4类 MI: 左手, 右手, 双脚, 舌头)。目前主要关注 **左手 (769)** vs **右手 (770)** 的二分类。
* **LSL 角色**: Python 是 Outlet (发送者)，Unity 是 Inlet (接收者)。

## 🔧 环境变量

将 `.env.example` 复制为 `.env` 并按需设置路径（如 MNE 下载/存放数据集的位置）：

```bash
cp .env.example .env
# 编辑 .env：可设置 MNE_DATA=/你的路径/mne_data（不设则默认 ~/mne_data）
```

常用变量：

* **MNE_DATA** – MNE 数据集根目录（PhysioNet EEGBCI、sample 等）。未设置时 MNE 使用 `~/mne_data`。
* 可选：数据集专用变量见 [MNE 配置](https://mne.tools/stable/overview/configuration.html)，如 `MNE_DATASETS_SAMPLE_PATH`。

使用 MNE 或项目数据的脚本在运行时会通过 `python-dotenv` 加载项目根目录的 `.env`。请勿提交 `.env`（已列入 `.gitignore`）。

**Git**：请使用 [Conventional Commits](https://www.conventionalcommits.org/)（如 `feat(scope): 描述`、`docs: ...`、`fix: ...`）。详见 `.cursor/rules/git-commits.mdc`。

**Cursor**：项目规则在 `.cursor/rules/`（[规则](https://cursor.com/cn/docs/context/rules)）。命令在 `.cursor/commands/` — 在聊天中输入 `/` 可运行如 `/setup-env`、`/download-datasets`、`/train-model`、`/run-replay`（[命令](https://cursor.com/cn/docs/context/commands)）。

## 🚀 快速开始

1. 创建并激活 **thesis** conda 环境：`conda env create -f environment.yml`，然后 `conda activate thesis`。（若已存在：`conda activate thesis && conda env update -f environment.yml --prune`。）
2. 复制 `.env.example` 为 `.env`，按需设置 `MNE_DATA`（或使用默认）。
3. 在 Cursor/VS Code 中：**Python: 选择解释器**（Ctrl+Shift+P），选择 conda 环境 `thesis` 的解释器。之后运行 Python 时新终端会自动激活 thesis。（若使用 Anaconda 而非 Miniconda，请编辑 `.vscode/settings.json`，将 `python.defaultInterpreterPath` 中的 `miniconda3` 改为 `anaconda3`。）
4. **（可选）** 下载数据集：`python python_backend/download_datasets.py`（默认下载 BCI IV 2a+2b 到 `MNE_DATA`）。可加 `--2a-only` 仅下 2a；`--physionet-eegbci` 同时下载 PhysioNet EEG Motor Movement/Imagery；`--physionet-eegbci-only` 仅下载该数据集；`--path /自定义路径` 指定目录。
5. 训练模型：`python python_backend/train_model.py`
6. 打开 Unity 项目，进入 `MainScene` 并运行。
7. 运行 `python python_backend/replay_stream.py` 开始推送数据。
