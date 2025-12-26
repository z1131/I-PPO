
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregation(nn.Module):
    """
    【核心模块 1】注意力语义聚合 (ASA)
    
    作用：把不管多长的输入序列 (Hidden States)，都通过注意力机制压缩成一个固定长度的向量。
    就像人看文章，不管文章多长，最后脑子里记住的只是几个关键点。
    """
    def __init__(self, input_dim):
        """
        input_dim: 输入特征的维度 (比如 1024)
        """
        super(AttentionAggregation, self).__init__()
        
        # 这里的 query 是一个可训练的参数，相当于智能体脑子里的“关注点”。
        # 它会去和输入的每一个词进行比对，看看哪个词更重要。
        self.query = nn.Parameter(torch.randn(input_dim, 1))
        
    def forward(self, hidden_states, mask=None):
        """
        前向传播计算。
        输入:
            hidden_states: (Batch, Seq_Len, Dim)
            mask: (Batch, Seq_Len) 0/1 掩码，1表示有效，0表示Padding
        """
        # (Batch, Seq_Len, 1)
        scores = torch.matmul(hidden_states, self.query)
        
        # Masking strategy
        if mask is not None:
             # 将 Padding 位置的分数设为极小值 (-1e9)，这样 Softmax 后概率为 0
             # mask 需要扩展维度适配 scores
             extended_mask = mask.unsqueeze(-1) # (Batch, Seq, 1)
             scores = scores.masked_fill(extended_mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=1) 
        
        # (Batch, 1, Dim)
        v_sem = torch.matmul(hidden_states.transpose(1, 2), weights).squeeze(-1)
        
        return v_sem

class DualHeadActor(nn.Module):
    """
    【核心模块 2】双头策略网络 (Dual-Head Actor)
    """
    def __init__(self, config):
        super(DualHeadActor, self).__init__()
        # Config can be a dict or Config object. Supporting both for compatibility.
        if isinstance(config, dict):
             input_dim = config.get('input_dim', 2048) 
             hidden_dim = config.get('hidden_dim', 256) 
             num_blocks = config.get('max_blocks', 100)
        else:
             # Assuming Config object from configs.config
             input_dim = getattr(config, 'INPUT_DIM', 2048)
             hidden_dim = getattr(config, 'HIDDEN_DIM', 256)
             num_blocks = getattr(config, 'MAX_BLOCKS', 100)
        
        self.asa_module = AttentionAggregation(input_dim)
        resource_dim = 4
        
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + resource_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.router_head = nn.Linear(hidden_dim, 2)
        self.caching_head = nn.Linear(hidden_dim, num_blocks)
        
    def forward(self, hidden_states, resource_states):
        """
        输入:
             hidden_states: (Batch, Seq_Len, Dim)
             resource_states: [剩余显存, 网络延迟, 任务长度, 信息熵]
        """
        # 1. 动态生成 Mask
        # resource_states[:, 2] 存储了真实的 Sequence Length
        # hidden_states.shape[1] 是当前 Batch 的最大填充长度
        batch_size, max_len, _ = hidden_states.shape
        seq_lengths = resource_states[:, 2].long() # 取出真实长度
        
        # 生成 Mask: (Batch, Max_Len)
        # arange: [0, 1, ..., max-1]
        # mask[i, j] = 1 if j < seq_len[i] else 0
        mask = torch.arange(max_len).expand(batch_size, max_len).to(hidden_states.device) < seq_lengths.unsqueeze(1)
        
        # 2. ASA 聚合 (带 Mask)
        v_sem = self.asa_module(hidden_states, mask)
        
        # 3. 拼接
        combined_input = torch.cat([v_sem, resource_states], dim=1)
        
        # 4. 共享特征提取
        features = self.shared_net(combined_input)
        
        # 5. 双头输出
        router_logits = self.router_head(features)
        router_probs = F.softmax(router_logits, dim=-1)
        
        caching_scores = torch.sigmoid(self.caching_head(features))
        
        return router_probs, caching_scores

class Critic(nn.Module):
    """
    【辅助模块】评论家 (Critic)
    
    作用：给 Actor 目前的表现打分 (Value Function)，告诉它这步走得好不好。
    """
    def __init__(self, config):
        super(Critic, self).__init__()
        if isinstance(config, dict):
             input_dim = config.get('input_dim', 2048) 
             hidden_dim = config.get('hidden_dim', 256) 
        else:
             input_dim = getattr(config, 'INPUT_DIM', 2048)
             hidden_dim = getattr(config, 'HIDDEN_DIM', 256)
             
        resource_dim = 4
        
        self.asa_module = AttentionAggregation(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + resource_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出一个标量：价值 Value
        )
        
    def forward(self, hidden_states, resource_states):
        # 生成 mask（和 Actor 一样的逻辑）
        batch_size, max_len, _ = hidden_states.shape
        seq_lengths = resource_states[:, 2].long()
        mask = torch.arange(max_len).expand(batch_size, max_len).to(hidden_states.device) < seq_lengths.unsqueeze(1)
        
        v_sem = self.asa_module(hidden_states, mask)
        combined = torch.cat([v_sem, resource_states], dim=1)
        value = self.net(combined)
        return value
