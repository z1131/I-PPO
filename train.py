
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from tqdm import tqdm

from configs.config import Config
from utils.logger import setup_logger
from core.environment import VectorizedRealSystemEnv
from core.agent import PPOAgent

def train():
    # 1. 设置日志
    logger = setup_logger()
    logger.info(f"开始 ASA-PPO 训练 (向量化 {Config.NUM_ENVS}x)...")
    
    # 2. 配置信息
    logger.info(f"设备: {Config.DEVICE}")
    logger.info(f"模型: {Config.MODEL_NAME}")
    logger.info(f"并行环境数: {Config.NUM_ENVS}")
    
    # 3. 初始化向量化环境
    logger.info("正在初始化向量化环境...")
    try:
        env = VectorizedRealSystemEnv()
    except Exception as e:
        logger.critical(f"环境初始化失败: {e}")
        return

    # 4. 初始化 Agent
    logger.info("正在初始化 ASA-PPO 智能体...")
    agent = PPOAgent(Config)
    
    # 5. 训练循环
    # 注意: NUM_EPISODES 现在指的是 "更新周期" 或 "Episode 批次"
    # 处理的总 Episode 数 = 循环次数 * NUM_ENVS
    num_updates = Config.NUM_EPISODES
    total_rewards_history = []
    
    logger.info(f"开始训练，共 {num_updates} 个更新周期 (总计 {num_updates * Config.NUM_ENVS} 个 episode)...")
    
    # 初始观测 (Batch)
    obs = env.reset()
    
    # 使用 tqdm 包装外层循环
    for update in tqdm(range(num_updates), desc="Training Updates", unit="update"):
        # 存储当前更新周期的奖励
        episode_rewards = np.zeros(Config.NUM_ENVS)
        
        # 我们是每个更新周期运行固定步数，还是运行直到结束？
        # 由于向量化环境自动重置，'done' 是异步发生的。
        # PPO 通常每个环境收集固定的 buffer (例如 20 步)。
        # 让我们尝试模拟原来的 "每个更新周期一个 Episode" 的感觉：
        # 我们运行直到大致所有环境都完成了 MAX_TURNS 步？
        # 不，更简单的方法：运行固定步数 (例如 10 步)，然后更新。
        
        STEPS_PER_UPDATE = 10 
        
        # 使用 tqdm 包装内层循环，leave=False 表示完成后清除进度条
        for step in tqdm(range(STEPS_PER_UPDATE), desc="Collecting Steps", leave=False, unit="step"):
            # 1. 准备状态
            # obs 已经是批次字典: {'hidden_states': (32, ...), 'resource_states': (32, ...)}
            state = {
                'hidden_states': obs['hidden_states'], 
                'resource_states': obs['resource_states']
            }
            
            # 2. Agent 决策 (Batch)
            # agent.get_action 需要处理批次输入
            action_route, action_cache_mask = agent.get_action(state)
            
            # 3. 环境步进 (向量化)
            next_obs, rewards, dones, infos = env.step(action_route, action_cache_mask)
            
            # 4. 存储经验
            # 使用 agent 的 current_log_prob，它存储了向量
            log_probs = agent.current_log_prob # (NUM_ENVS,)
            
            for i in range(Config.NUM_ENVS):
                single_state = {
                    'hidden_states': state['hidden_states'][i],
                    'resource_states': state['resource_states'][i]
                }
                
                agent.store_experience(
                    single_state, 
                    (action_route[i], action_cache_mask[i]), 
                    rewards[i], 
                    None, 
                    dones[i],
                    log_prob=log_probs[i] # 传递明确的标量
                )
            
            episode_rewards += rewards # 累加用于日志记录
            obs = next_obs
        
        # 5. 更新 Agent
        loss = agent.update()
        
        # 6. 日志记录
        avg_rew = np.mean(episode_rewards)
        total_rewards_history.append(avg_rew)
        
        if (update + 1) % 1 == 0:
            logger.info(f"更新 {update+1}/{num_updates} | 平均奖励 (Batch): {avg_rew:.2f} | Loss: {loss:.4f}")

    # 7. 保存结果
    plt.plot(total_rewards_history)
    plt.title(f"ASA-PPO Training (Vectorized x{Config.NUM_ENVS})")
    plt.xlabel("Update Cycle")
    plt.ylabel("Average Reward")
    plot_path = os.path.join(Config.LOG_DIR, "training_curve_vec.png")
    plt.savefig(plot_path)
    logger.info(f"训练结束。曲线已保存至 {plot_path}")

if __name__ == '__main__':
    train()
