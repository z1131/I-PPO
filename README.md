# ASA-PPO 实验代码库

本目录包含《第三章-面向车云协同推理的动态资源感知路由与KV缓存联合优化》的实验代码。

## 核心功能
实现了 **ASA-PPO** 算法，这是一种基于强化学习的联合优化框架，能够同时决策：
1.  **路由 (Route)**: 任务在本地做还是去云端做？(基于显存、网络延迟、任务复杂度)
2.  **缓存 (Cache)**: 本地显存满了删谁？(基于语义价值评分)

## 📂 项目结构
| 文件名 | 说明 |
| :--- | :--- |
| `train_real_system.py` | **主启动脚本**。负责初始化 Agent 和 Environment，开启 PPO 训练循环。 |
| `real_env.py` | **真实环境与奖励逻辑**。加载 Qwen3 模型，加载混合数据集，实现 Counterfactual Reward。 |
| `model.py` | **神经网络定义**。Dual-Head Actor (Router + Caching Head) 和 Critic。 |
| `agent.py` | **PPO 智能体**。实现行动采样 (Sample) 和参数更新 (Update)。 |
| `network_architecture.md` | **架构文档**。详细记录了针对 Qwen3-1.7B 的网络维度设计。 |
| `requirements.txt` | **依赖列表**。包含 datasets, transformers, bitsandbytes 等。 |
| `run_slurm.sh` | **Slurm 启动脚本**。用于在超算平台上提交任务。 |

## 🚀 快速开始

### 1. 环境准备
确保你有 NVIDIA GPU (显存 >= 6GB)。
```bash
conda create -n asa_env python=3.10
conda activate asa_env
pip install -r requirements.txt
```

### 3.1 资源准备 (一次性)

由于服务器算力昂贵，建议在正式训练前，先运行下载脚本将模型和数据集缓存到本地：

```bash
python download_resources.py
```

该脚本会自动从 ModelScope 下载：
*   模型: `Qwen/Qwen3-1.7B`
*   数据集: `CommonsenseQA`, `OpenBookQA`, `ARC-Challenge`, `MATH`

### 3.2 启动训练

确认资源下载完成后，运行主训练脚本：

```bash
python train_real_system.py
```

### 3. 输出说明
程序会生成 `training_curve_real.png`，展示 Reward 的上升趋势。
同时会在控制台输出当前的 OOM 次数和平均奖励。
-   `Loss`: 训练是否收敛。

## 示例日志

```text
轮次  10 | 总分:  -50.0 | 炸机:  1次 | 云端占比: 10.0% ... (一开始比较笨)
...
轮次 100 | 总分:  850.0 | 炸机:  0次 | 云端占比: 45.0% ... (学会了在显存紧张时发给云端)
```

## 注意事项

-   本代码默认尝试在 **GPU** 上运行 (Qwen3-1.7B 需要约 4GB 显存)。如果没有 GPU，会自动切换到 Mock 模式或报错。
-   `real_env.py` 是核心模拟环境，已集成了真实模型加载、多轮对话逻辑和 Counterfactual Reward 计算。
# I-PPO
