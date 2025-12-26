import os
import torch

class Config:
    # --- 路径与资源配置 ---
    # 模型配置
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    # 如果需要，可以使用 ModelScope 缓存路径逻辑或自定义路径
    
    # 云端 API 配置
    CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "sk-277e287188874fc98b55edad9e498fbb")
    CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    CLOUD_MODEL_NAME = os.getenv("CLOUD_MODEL_NAME", "qwen-max")

    # 日志与检查点
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    # --- 训练超参数 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # PPO 超参数
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    EPS_CLIP = 0.2
    K_EPOCHS = 4
    
    # 模型架构
    INPUT_DIM = 2048  # Qwen3-1.7B 维度
    HIDDEN_DIM = 256
    ACTION_DIM = 2
    
    # 环境设置
    NUM_ENVS = 8    # 并行环境数量 (向量化)
    MAX_BLOCKS = 100
    BLOCK_SIZE = 16  # KV Cache 管理中每块的 Token 数
    MAX_TURNS = 5
    LAMBDA_OOM = -10.0
    
    # 训练循环
    NUM_EPISODES = 200 # 总 Episode 数 (在循环中会被 NUM_ENVS 除)
    BATCH_SIZE = 128   # PPO 更新的 Batch Size (SGD 的 Mini-batch 大小)
    
    # --- 数据集 ---
    # 格式: (数据集名称, 子集名称, 划分列表)
    DATASETS = [
        ('opencompass/commonsense_qa', None, ['train', 'validation']),
        ('opencompass/openbookqa', None, ['train', 'validation']),
        ('modelscope/ai2_arc', 'ARC-Challenge', ['train', 'validation']),
        ('AI-ModelScope/MATH-500', 'default', ['test'])
    ]
    
    # OpenBookQA 特定哈希值 (用于手动回退加载)
    OBQA_HASHES = [
        "e75c8e10b17b225fe6ba56975e5a127ceaee377fa051718b77b6293864d969d9", # validation
        "cdeb713ead8b41f116c430120af974c8beddc7caa52d84bb28bc9a6cea7778bd"  # train
    ]