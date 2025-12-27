
import sys
import os
import torch
import traceback
import logging

# 将项目根目录加入 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from core.environment import VectorizedRealSystemEnv
from core.agent import PPOAgent

# 设置简易日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VERIFY")

def mock_data_pool(env):
    """注入虚拟数据，跳过 loader"""
    logger.info("💉 注入 Mock 数据 (Hello World)...")
    env.data_pool = [
        {'question': 'Hello world, this is a test prompt.', 'answer': 'World'},
        {'question': 'What is the implementation of PPO?', 'answer': 'Constraint optimization'},
        {'question': 'Testing batch inference mechanics.', 'answer': 'Batch'},
        {'question': 'Long context debugging test case.', 'answer': 'Debug'}
    ]
    # 强制覆盖 tokenizer (如果模型加载慢，这里可以 mock tokenizer，但为了验证 pipeline 还是用真的)
    # env.tokenizer = ... 

def run_test():
    print("="*50)
    print("🚀 开始 I-PPO 全流程代码验证")
    print("="*50)

    try:
        # 1. 动态修改配置 (缩小规模以便快速运行)
        logger.info("⚙️  调整 Config 为测试模式...")
        Config.NUM_ENVS = 4         # 只开 4 个环境
        Config.MAX_NEW_TOKENS = 10  # 只生成 10 个 token
        Config.BATCH_SIZE = 8       # 极小 batch
        Config.GPU_IDS = [0] if torch.cuda.is_available() else [] # 单卡测试
        
        # 2. 初始化环境
        logger.info("🛠️  初始化环境 (VectorizedRealSystemEnv)...")
        env = VectorizedRealSystemEnv()
        mock_data_pool(env) # 注入数据
        
        # 3. 初始化 Agent
        logger.info("🧠 初始化 Agent (PPOAgent)...")
        agent = PPOAgent(Config)
        
        # 4. Reset
        logger.info("🔄 Testing Env Reset...")
        obs = env.reset()
        assert 'hidden_states' in obs, "Obs 缺失 hidden_states"
        assert obs['hidden_states'].shape[0] == Config.NUM_ENVS, f"Batch 维度错误: {obs['hidden_states'].shape}"
        logger.info("   -> Reset Success. Obs Shape: OK")

        # 5. Step Loop
        logger.info("👣 Testing Interaction Loop (Step & Store)...")
        steps = 5
        for s in range(steps):
            state = {
                'hidden_states': obs['hidden_states'],
                'resource_states': obs['resource_states']
            }
            
            # Action
            action_route, action_cache_mask, gate_usage = agent.get_action(state)
            
            # Env Step
            next_obs, rewards, dones, infos = env.step(action_route, action_cache_mask, gate_usage)
            
            # Store
            log_probs = agent.current_log_prob
            for i in range(Config.NUM_ENVS):
                single_state = {
                    'hidden_states': state['hidden_states'][i],
                    'resource_states': state['resource_states'][i]
                }
                agent.store_experience(
                    i, single_state, 
                    (action_route[i], action_cache_mask[i]), 
                    rewards[i], dones[i], log_prob=log_probs[i]
                )
            
            obs = next_obs
            # logger.info(f"   -> Step {s+1}/{steps} complete.")
        
        logger.info("   -> Interaction Loop Success.")

        # 6. Bootstrap Value
        logger.info("🔮 Testing Value Bootstrap...")
        state_next = {
            'hidden_states': obs['hidden_states'],
            'resource_states': obs['resource_states']
        }
        next_values = agent.get_value(state_next)
        assert next_values.shape == (Config.NUM_ENVS,), f"Value shape mismatch: {next_values.shape}"
        
        # 7. Update
        logger.info("📉 Testing PPO Update (Backward Pass)...")
        loss = agent.update(next_values)
        logger.info(f"   -> Update Success. Loss: {loss:.4f}")

        print("="*50)
        print("✅ 验证通过！代码逻辑链路正常。")
        print("="*50)
        
    except Exception as e:
        print("\n" + "!"*50)
        print("❌ 验证失败！捕获到异常：")
        print("!"*50 + "\n")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {e}")
        print("\n--- 完整堆栈信息 (Traceback) ---")
        traceback.print_exc()
        print("-------------------------------\n")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
