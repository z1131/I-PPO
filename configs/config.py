import os
import torch

class Config:
    # --- 路径配置 ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
    MODELS_DIR = os.path.join(ASSETS_DIR, "models")
    DATASETS_DIR = os.path.join(ASSETS_DIR, "datasets")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    
    # --- 模型配置 ---
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    
    # 云端 API 配置
    CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "sk-gy23faA0CAruuR7GgHlFwA")
    CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://llmapi.paratera.com")
    CLOUD_MODEL_NAME = os.getenv("CLOUD_MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507")

    # --- 训练超参数 ---
    # 自动识别可用 GPU 列表
    if torch.cuda.is_available():
        # 默认使用所有可见 GPU，或者用户指定 [0, 1, 2, 3]
        GPU_COUNT = torch.cuda.device_count()
        GPU_IDS = list(range(GPU_COUNT)) 
        DEVICE = f'cuda:{GPU_IDS[0]}' # Agent 默认放在第一张卡
    else:
        GPU_IDS = []
        DEVICE = 'cpu'
    
    # PPO 超参数
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    EPS_CLIP = 0.2
    K_EPOCHS = 4
    
    # 模型架构
    INPUT_DIM = 2048  # Qwen3-1.7B 维度 (务必确认模型维度)
    HIDDEN_DIM = 256
    ACTION_DIM = 2
    
    # 环境设置
    NUM_ENVS = 32    # 并行环境数量
    MAX_BLOCKS = 100
    BLOCK_SIZE = 16
    
    # --- 关键优化: 限制生成长度以对齐 PPO Update ---
    # 设为 64 用于快速验证 (保证 Update 包含 Reward)
    # 正式跑 LongBench 时可调大到 512 或 1024
    MAX_NEW_TOKENS = 64 
    
    LAMBDA_OOM = -10.0
    
    # 训练循环
    NUM_EPISODES = 200
    BATCH_SIZE = 512 # Mini-batch size

    # 奖励调整系数
    LOCAL_DELAY_COEF = 0.05
    CLOUD_DELAY_COEF = 0.05
    GATE_PENALTY_COEF = 0.1
    BLOCK_PENALTY_COEF = 0.01
    
    # --- 数据处理配置 ---
    SKIP_FAILED_SAMPLES = True  # 失败样本跳过而非 mock
    MAX_SAMPLE_FAILURES_LOG = 100  # 最多记录的失败样本数
    
    # --- 数据集配置 ---
    # 格式: (数据集名称, 子集名称, 划分列表, 来源)
    # 来源: 'modelscope' 或 'huggingface'
    DATASETS = [
        # 短文本 QA (100-500 tokens)
        ('opencompass/commonsense_qa', None, ['train', 'validation'], 'modelscope'),
        ('opencompass/openbookqa', None, ['train', 'validation'], 'modelscope'),
        ('modelscope/ai2_arc', 'ARC-Challenge', ['train', 'validation'], 'modelscope'),
        ('AI-ModelScope/MATH-500', 'default', ['test'], 'modelscope'),
        # 中等长度 (~5000 tokens)
        ('nyu-mll/quality', None, ['train', 'validation'], 'huggingface'),
        # 长文本 (8k-32k tokens) - LongBench 单文档 QA
        ('THUDM/LongBench', 'qasper', ['test'], 'huggingface'),
        ('THUDM/LongBench', 'multifieldqa_en', ['test'], 'huggingface'),
    ]
