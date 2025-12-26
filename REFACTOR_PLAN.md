# ASA-PPO 项目重构与优化计划

本文档总结了从单卡实验环境向 4 卡高性能训练环境迁移的改造方案。

## 1. 核心目标
将目前的串行推理训练流程升级为 **并行化、高吞吐** 的生产级训练流程，目标是在 4x GPU 环境下实现 **8-10倍** 的训练效率提升。

## 2. 关键技术选型

| 模块 | 当前实现 | 升级方案 | 优势 |
| :--- | :--- | :--- | :--- |
| **推理后端** | HuggingFace Native (`model.generate`) | **vLLM** | 支持 PagedAttention，吞吐量提升 5-10 倍，原生支持多卡张量并行 (Tensor Parallelism)。 |
| **注意力机制** | 标准 Attention | **Flash Attention 2** | 显存占用降低 50%，计算速度提升 2-3 倍，支持更长上下文。 |
| **环境并行** | 简单的 Python 循环 | **异步采样 (Async Sampling)** | 利用 vLLM 的 `AsyncLLMEngine` 实现真正的非阻塞并发采样。 |
| **数据管理** | 内存 List | **HuggingFace Datasets** | 支持流式加载 (Streaming)，避免 OOM。 |

## 3. 实施路线图 (Roadmap)

### 第一阶段：环境与基础设施 (Infrastructure)
- [ ] **新服务器配置**：
    - 规格：4x NVIDIA GPU (推荐 A10/A100/3090/4090)。
    - 系统盘：>100GB (SSD)。
    - 镜像：PyTorch 2.1.2 + CUDA 12.1 (推荐官方镜像，避开安装坑)。
- [ ] **依赖安装**：
    - `pip install vllm flash-attn`
    - `pip install ray` (vLLM 多卡依赖)

### 第二阶段：代码重构 (Refactoring)
- [ ] **重写 `core/environment.py`**：
    - 废弃 `VectorizedRealSystemEnv` 中的 `AutoModelForCausalLM`。
    - 引入 `vllm.LLM` 类进行初始化。
    - 改造 `step()` 函数，使用 `llm.generate()` 替代手动 Batch Loop。
    - 移除所有手动 Padding/Masking 代码 (vLLM 自动处理)。
- [ ] **配置多卡推理**：
    - 在初始化时设置 `tensor_parallel_size=4`，让 vLLM 自动将 1.7B 模型切分到 4 张卡上。

### 第三阶段：训练流程优化 (Training Loop)
- [ ] **采样-训练解耦**：
    - 使用 vLLM (4卡) 快速生成 32-64 个并发环境的数据。
    - 将数据汇总到主卡 (cuda:0) 进行 PPO 更新 (Actor/Critic forward/backward)。
- [ ] **参数调整**：
    - `NUM_ENVS`: 8 -> 32+ (充分喂饱 GPU)。
    - `BATCH_SIZE`: 128 -> 512+ (稳定梯度)。

## 4. 预期收益
1.  **速度**：单步采样时间从 ~30s 降低到 ~3s。
2.  **显存**：通过 Flash Attention，支持的最大 Token 数翻倍。
3.  **稳定性**：消除手动 Padding 带来的潜在 Bug (如之前的 `right-padding` 报错)。

## 5. 迁移检查清单 (Checklist)
在关闭当前服务器前，请确保已保存：
- [x] `train.py`
- [x] `core/` (agent.py, environment.py, model.py)
- [x] `configs/config.py`
- [x] `REFACTOR_PLAN.md` (本文档)
- [ ] 代码已 Push 到 GitHub 仓库
