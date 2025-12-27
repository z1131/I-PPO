import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .model import DualHeadActor, Critic
import numpy as np

class PPOAgent:
    """
    【大脑】I-PPO 智能体 (支持向量化)
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
            self.num_envs = config.get('num_envs', 1)
        else:
            self.device = getattr(config, 'DEVICE', 'cpu')
            self.lr_actor = getattr(config, 'LR_ACTOR', 1e-4)
            self.lr_critic = getattr(config, 'LR_CRITIC', 1e-3)
            self.gamma = getattr(config, 'GAMMA', 0.99)
            self.clip = getattr(config, 'EPS_CLIP', 0.2)
            self.k_epochs = getattr(config, 'K_EPOCHS', 4)
            self.num_envs = getattr(config, 'NUM_ENVS', 1)
        
        self.actor = DualHeadActor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.critic.parameters(), 'lr': self.lr_critic}
        ])
        
        self.mse_loss = nn.MSELoss()
        
        # 向量化 Memory: 为每个环境维护独立的 Buffer
        self.memories = [[] for _ in range(self.num_envs)]
        
        self.current_log_prob = None 
        self.current_gate_usage = None
        
    def get_action(self, state):
        """批量动作选择"""
        hidden_states = torch.FloatTensor(state['hidden_states']).to(self.device) 
        resource_states = torch.FloatTensor(state['resource_states']).to(self.device)
        
        if hidden_states.dim() == 2: 
            hidden_states = hidden_states.unsqueeze(1)
        
        with torch.no_grad():
            router_probs, caching_probs, gate_probs = self.actor(hidden_states, resource_states)
        
        dist_route = Categorical(router_probs)
        action_route = dist_route.sample()
        log_prob_route = dist_route.log_prob(action_route)
        
        dist_cache = torch.distributions.Bernoulli(caching_probs)
        action_cache_mask = dist_cache.sample()
        log_prob_cache = dist_cache.log_prob(action_cache_mask).sum(dim=-1)
        
        total_log_prob = log_prob_route + log_prob_cache
        
        self.current_log_prob = total_log_prob.cpu().numpy()
        self.current_gate_usage = gate_probs.squeeze(-1).cpu().numpy()
        
        return action_route.cpu().numpy(), action_cache_mask.cpu().numpy(), self.current_gate_usage
        
    def get_value(self, state):
        """辅助方法：获取当前状态的 Value (用于 Bootstrap)"""
        hidden_states = torch.FloatTensor(state['hidden_states']).to(self.device)
        resource_states = torch.FloatTensor(state['resource_states']).to(self.device)
        
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            
        with torch.no_grad():
            values = self.critic(hidden_states, resource_states).squeeze()
        return values.cpu().numpy() # (Batch,)

    def store_experience(self, env_idx, state, action, reward, done, log_prob=None):
        """存储单个经验到指定环境的 Buffer"""
        action_route, action_cache_mask = action
        
        self.memories[env_idx].append({
            'hidden_states': state['hidden_states'],
            'resource_states': state['resource_states'],
            'action_route': action_route,
            'action_cache_mask': action_cache_mask,
            'log_prob': log_prob if log_prob is not None else 0.0,
            'reward': reward,
            'done': done
        })

    def update(self, next_values=None):
        """
        PPO Update with Vectorized Support & GAE
        next_values: (num_envs,) 数组，用于 Bootstrap 未完成的 Episode
        """
        # 检查是否有数据
        if not any(self.memories):
            return 0.0
            
        # 1. 整理所有环境的数据并计算 Advantage
        all_hidden_states = []
        all_resource_states = []
        all_actions_route = []
        all_actions_cache = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        # 为了高效计算 Value，我们是否应该先 Batch 跑一遍 Critic?
        # 考虑到代码复杂度，我们可以先按 Env 跑，或者把所有 obs 拼起来跑。
        # 这里为了实现最准确的 GAE，我们需要每个 Trajectory 的 Values。
        # 简单起见，我们在 Loop 里对每个 Env 的 Trajectory 单独处理 (因为长度可能稍有不同 in theory, though synchronized here)
        
        gae_lambda = 0.95 # GAE lambda 参数
        
        for i in range(self.num_envs):
            traj = self.memories[i]
            if not traj:
                continue
                
            # 提取该环境的 Trajectory 数据
            # Hidden: (T, Dim) -> (T, 1, Dim)
            raw_h = [step['hidden_states'] for step in traj]
            env_hidden = torch.FloatTensor(np.array(raw_h)).unsqueeze(1).to(self.device)
            
            raw_r = [step['resource_states'] for step in traj]
            env_resource = torch.FloatTensor(np.array(raw_r)).to(self.device)
            
            rewards = [step['reward'] for step in traj]
            dones = [step['done'] for step in traj]
            
            # 计算 Values (当前策略)
            with torch.no_grad():
                values = self.critic(env_hidden, env_resource).squeeze(-1) # (T,)
            
            # GAE Calculation
            advantages = []
            last_advantage = 0
            
            # Bootstrap Value
            # 如果提供了 next_values，则用它；否则假设为 0 (Done)
            next_val = next_values[i] if next_values is not None else 0.0
            
            # 倒序计算
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    non_terminal = 1.0 - float(dones[t]) # 如果 done，则没有未来
                    next_term_val = next_val * non_terminal
                else:
                    non_terminal = 1.0 - float(dones[t])
                    next_term_val = values[t+1] * non_terminal
                
                delta = rewards[t] + self.gamma * next_term_val - values[t]
                advantage = delta + self.gamma * gae_lambda * non_terminal * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)
            
            # 计算 Returns = Advantage + Value
            returns = [adv + val for adv, val in zip(advantages, values.cpu().numpy())]
            
            # 存入汇总列表
            all_hidden_states.extend(raw_h)
            all_resource_states.extend(raw_r)
            all_actions_route.extend([step['action_route'] for step in traj])
            all_actions_cache.extend([step['action_cache_mask'] for step in traj])
            all_old_log_probs.extend([step['log_prob'] for step in traj])
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            
        # 2. 转换为 Tensor (Batch)
        # 注意: 此时所有环境的所有 timestep 都被 flatten 到了一起
        batch_hidden = torch.FloatTensor(np.array(all_hidden_states)).unsqueeze(1).to(self.device)
        batch_resource = torch.FloatTensor(np.array(all_resource_states)).to(self.device)
        batch_route = torch.LongTensor(np.array(all_actions_route)).to(self.device)
        batch_cache = torch.FloatTensor(np.array(all_actions_cache)).to(self.device)
        batch_log_probs = torch.FloatTensor(np.array(all_old_log_probs)).to(self.device)
        batch_advantages = torch.FloatTensor(np.array(all_advantages)).to(self.device)
        batch_returns = torch.FloatTensor(np.array(all_returns)).to(self.device)
        
        # 归一化 Advantage
        if batch_advantages.std() > 1e-5:
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-5)
            
        # 3. Mini-batch PPO Update
        total_steps = batch_hidden.size(0)
        indices = np.arange(total_steps)
        mini_batch_size = self.config.BATCH_SIZE
        
        avg_loss = 0
        updates_count = 0
        
        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_steps, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                
                # Slicing
                mb_hidden = batch_hidden[mb_idx]
                mb_resource = batch_resource[mb_idx]
                mb_route = batch_route[mb_idx]
                mb_cache = batch_cache[mb_idx]
                mb_old_log_probs = batch_log_probs[mb_idx]
                mb_advantages = batch_advantages[mb_idx]
                mb_returns = batch_returns[mb_idx]
                
                # Forward Actor
                curr_route_probs, curr_cache_probs, _ = self.actor(mb_hidden, mb_resource)
                
                dist_route = Categorical(curr_route_probs)
                log_prob_route = dist_route.log_prob(mb_route)
                
                dist_cache = torch.distributions.Bernoulli(curr_cache_probs)
                log_prob_cache = dist_cache.log_prob(mb_cache).sum(dim=-1)
                
                curr_log_probs = log_prob_route + log_prob_cache
                
                entropy = dist_route.entropy().mean() + dist_cache.entropy().mean(dim=-1).mean()
                
                # Forward Critic
                state_values = self.critic(mb_hidden, mb_resource).squeeze()
                
                # Ratios
                ratio = torch.exp(curr_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * mb_advantages
                
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.mse_loss(state_values, mb_returns)
                
                loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.item()
                updates_count += 1
                
        # 清空所有环境的 Memory
        self.memories = [[] for _ in range(self.num_envs)]
        
        return avg_loss / updates_count if updates_count > 0 else 0.0
