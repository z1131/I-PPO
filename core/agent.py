import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .model import DualHeadActor, Critic
import numpy as np

class PPOAgent:
    """
    【大脑】ASA-PPO 智能体 (支持向量化)
    """
    def __init__(self, config):
        self.config = config
        
        if isinstance(config, dict):
            self.device = config.get('device', 'cpu')
            self.lr_actor = config.get('lr_actor', 1e-4)
            self.lr_critic = config.get('lr_critic', 1e-3)
            self.gamma = config.get('gamma', 0.99)
            self.clip = config.get('eps_clip', 0.2)
            self.k_epochs = config.get('K_epochs', 4)
        else:
            self.device = getattr(config, 'DEVICE', 'cpu')
            self.lr_actor = getattr(config, 'LR_ACTOR', 1e-4)
            self.lr_critic = getattr(config, 'LR_CRITIC', 1e-3)
            self.gamma = getattr(config, 'GAMMA', 0.99)
            self.clip = getattr(config, 'EPS_CLIP', 0.2)
            self.k_epochs = getattr(config, 'K_EPOCHS', 4)
        
        self.actor = DualHeadActor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.critic.parameters(), 'lr': self.lr_critic}
        ])
        
        self.mse_loss = nn.MSELoss()
        
        self.memory = []
        # log_prob 存储需要处理 batch
        self.current_log_prob = None 
        
    def get_action(self, state):
        """
        批量动作选择 (Batch Action Selection)
        输入 state: 
            'hidden_states': (Batch, Seq_Len, Dim) 
            'resource_states': (Batch, 4)
        """
        # 1. 转换为张量
        # 状态输入已经是来自 VectorizedEnv 的批次 numpy 数组
        hidden_states = torch.FloatTensor(state['hidden_states']).to(self.device) 
        resource_states = torch.FloatTensor(state['resource_states']).to(self.device)
        
        # 确保维度 (Batch, Seq, Dim)
        if hidden_states.dim() == 2: # 如果是 (Batch, Dim) - 发生在 Seq=1 时? 
            # 等等，Env 返回 (Batch, Dim) 因为它提取了最后一个 Token 的 hidden state
            # 但是 Model 期望 (Batch, Seq, Dim)。
            # 我们需要 unsqueeze 变成 (Batch, 1, Dim) 给 Actor
            hidden_states = hidden_states.unsqueeze(1)
        
        # 2. 前向传播
        with torch.no_grad():
            router_probs, caching_probs = self.actor(hidden_states, resource_states)
        
        # 3. 采样路由 (Batch)
        dist_route = Categorical(router_probs)
        action_route = dist_route.sample() # (Batch,)
        log_prob_route = dist_route.log_prob(action_route) # (Batch,)
        
        # 4. 采样缓存 (Batch)
        dist_cache = torch.distributions.Bernoulli(caching_probs)
        action_cache_mask = dist_cache.sample() # (Batch, 100)
        log_prob_cache = dist_cache.log_prob(action_cache_mask).sum(dim=-1) # (Batch,)
        
        # 5. 总 Log Prob
        total_log_prob = log_prob_route + log_prob_cache # (Batch,)
        
        # 存储以供 store_experience 循环使用 (Numpy 方便索引)
        self.current_log_prob = total_log_prob.cpu().numpy()
        
        return action_route.cpu().numpy(), action_cache_mask.cpu().numpy()

    def store_experience(self, state, action, reward, next_state, done, log_prob=None):
        """
        存储单个经验。
        注意: PPOAgent.update() 期望字典列表。
        如果使用向量化 Env，调用者会循环调用此方法 N 次。
        """
        action_route, action_cache_mask = action
        
        # 如果明确传递了 log_prob (推荐)，则使用它。
        # 如果没有，尝试使用 self.current_log_prob (在循环中有风险)
        val_log_prob = log_prob
        if val_log_prob is None:
             # 回退逻辑 (假设 batch size 为 1 或外部管理)
             pass

        self.memory.append({
            'hidden_states': state['hidden_states'],
            'resource_states': state['resource_states'],
            'action_route': action_route,
            'action_cache_mask': action_cache_mask,
            'log_prob': torch.tensor(val_log_prob) if val_log_prob is not None else torch.tensor(0.0), # 占位修复
            'reward': reward,
            'done': done
        })

    def update(self):
        """
        标准 PPO 更新 (逻辑不变，只是现在处理更多数据)
        """
        if not self.memory:
            return 0.0
            
        # 1. 准备 Batch
        # Hidden States: list of (Dim,) -> (Batch, 1, Dim)
        raw_hidden = [torch.FloatTensor(x['hidden_states']) for x in self.memory]
        hidden_states = torch.stack(raw_hidden).unsqueeze(1).to(self.device)
        
        resource_states = torch.FloatTensor(np.array([x['resource_states'] for x in self.memory])).to(self.device)
        actions_route = torch.LongTensor([x['action_route'] for x in self.memory]).to(self.device)
        actions_cache_mask = torch.FloatTensor(np.array([x['action_cache_mask'] for x in self.memory])).to(self.device)
        old_log_probs = torch.stack([x['log_prob'] for x in self.memory]).to(self.device).detach()
        
        rewards = [x['reward'] for x in self.memory]
        is_terminals = [x['done'] for x in self.memory]
        
        # 2. Rewards to Go (折扣奖励)
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device)
        if rewards_to_go.std() > 1e-5:
            rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-5)
            
        # 3. 优化 (Optimization)
        loss_val = 0
        
        # Mini-batch 支持? Config.BATCH_SIZE 在这里使用?
        # 目前是 Full-batch 更新 (所有收集到的步骤)。
        
        for _ in range(self.k_epochs):
            # 评估
            current_router_probs, current_caching_probs = self.actor(hidden_states, resource_states)
            
            dist_route = Categorical(current_router_probs)
            log_prob_route = dist_route.log_prob(actions_route)
            
            dist_cache = torch.distributions.Bernoulli(current_caching_probs)
            log_prob_cache = dist_cache.log_prob(actions_cache_mask).sum(dim=-1)
            
            current_log_probs = log_prob_route + log_prob_cache
            
            entropy_route = dist_route.entropy()
            entropy_cache = dist_cache.entropy().mean(dim=-1)
            
            state_values = self.critic(hidden_states, resource_states).squeeze()
            
            # 匹配维度
            if state_values.dim() == 0: state_values = state_values.unsqueeze(0)
            
            advantages = rewards_to_go - state_values.detach()
            
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantages
            
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.mse_loss(state_values, rewards_to_go)
            total_entropy = entropy_route.mean() + entropy_cache.mean()
            
            loss = loss_actor + 0.5 * loss_critic - 0.01 * total_entropy
            loss_val = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.memory = []
        return loss_val
