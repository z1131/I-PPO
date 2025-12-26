# KV Cache 管理实现文档 (Scheme 2: True Attention Masking)

## 1. 核心理念

为了在不修改底层 CUDA 内核的前提下，真实模拟 KV Cache 的“删除”与“保留”对模型性能的影响，我们采用了 **True Attention Masking (真实注意力掩码)** 方案。

此方案的核心逻辑是：**“虽然物理上没有删除显存里的数据，但通过 Mask 让模型‘看不见’被删除的数据，从而在数学上等效于删除了该 Block。”**

## 2. 实现细节

### 2.1 块与 Token 的映射 (Block-to-Token Mapping)

我们定义 `BLOCK_SIZE = 16` (可在 `config.py` 中修改)。
这意味着显存被划分为若干个固定大小的槽位。

*   **Block 0**: 对应历史 Token 序列的 `[0, 15]`
*   **Block 1**: 对应历史 Token 序列的 `[16, 31]`
*   ...

### 2.2 状态维护 (State Tracking)

`RealSystemEnv` 不再维护单一的 `current_memory_blocks` 计数器，而是维护全量历史：

1.  `self.history_token_ids` (List[int]): 
    *   记录从 Episode 开始到现在，所有输入问题和生成答案的 Token ID 序列。
    *   这是一个**绝对坐标系**，保证 Token 的索引永远不变。

2.  `self.block_mask_status` (List[int]): 
    *   记录每个 Block 的存活状态。
    *   `1`: Keep (保留/可见)
    *   `0`: Evict (删除/不可见)
    *   列表长度 = `ceil(len(history) / BLOCK_SIZE)`

### 2.3 决策与执行循环 (The Loop)

#### Step 1: Agent 决策
Agent 输出 `action_cache_mask` (例如 `[1, 0, 1, ...]` )。
环境截取对应长度的 Mask，更新 `self.block_mask_status`。

#### Step 2: 构造掩码 (Mask Construction)
在进行下一次生成（Generation）或观测（Observation）时，环境动态构造 `attention_mask`：

```python
full_mask = []

# 1. 历史部分：根据 Block 状态填充
for block_idx, status in enumerate(block_mask_status):
    # 如果 status == 0 (删除)，则这 16 个 Token 的 mask 全部设为 0
    full_mask.extend([status] * BLOCK_SIZE)

# 2. 新增部分：当前的新问题
# 新问题必须可见，设为 1
full_mask.extend([1] * len(new_question_ids))
```

#### Step 3: 模型推理 (Inference)
调用 Hugging Face 的 `model.generate()`：

```python
model.generate(
    input_ids = history_ids + new_question_ids,
    attention_mask = full_mask  # <--- 注入灵魂
)
```

### 2.4 结果反馈

*   **如果删对了**：Agent 删除了无关紧要的 Block（Mask=0）。模型虽然看不见这些历史，但依然能根据剩余信息答对问题。Agent 获得高分，且节省了“显存”（Active Blocks 减少）。
*   **如果删错了**：Agent 删除了关键信息的 Block。模型看不见上下文，导致回答错误（ROUGE 低）。Agent 受到惩罚。

## 3. 显存模拟 (OOM Logic)

虽然物理显存占用是随 History 线性增长的（因为我们传入了全量 Input），但为了训练 Agent，我们计算**“虚拟显存”**：

$$ 	ext{Active Blocks} = 	ext{sum} (\text{block\_mask\_status}) $$

*   如果 `Active Blocks > MAX_BLOCKS`，则触发 OOM（炸机）。
*   这意味着 Agent 必须在 `MAX_BLOCKS` 的限制内，通过将不重要的 Block 设为 0 来腾出空间。

## 4. 优势总结

1.  **逻辑闭环**：真正实现了“内容重要性”与“模型性能”的关联。Agent 必须学会语义理解才能得分。
2.  **工程稳健**：避免了底层 KV Cache 修改带来的索引错乱风险。每次全量计算保证了 Positional Embedding 和 Token ID 的绝对正确。
