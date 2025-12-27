
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyPointGate(nn.Module):
    """线性门控：判断当前 token 是否为关键点。"""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, hidden_vec):
        """
        hidden_vec: (Batch, Dim)
        返回 gated 向量和门控概率。
        """
        gate_logits = self.linear(hidden_vec)  # (Batch, 1)
        gate_probs = torch.sigmoid(gate_logits)
        gated_hidden = hidden_vec * gate_probs
        return gated_hidden, gate_probs


class TransformerAggregator(nn.Module):
    """单层 Transformer，用于替换原来的 ASA 聚合。"""

    def __init__(self, input_dim, nhead=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, hidden_states):
        """
        hidden_states: (Batch, Seq_Len, Dim)
        返回 Transformer 编码后的 [CLS] 等价向量（此处取第一个位置）。
        """
        transformed = self.encoder(hidden_states)
        return transformed[:, 0, :]

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
        resource_dim = 4

        self.gate = KeyPointGate(input_dim)
        self.transformer = TransformerAggregator(input_dim)
        gate_dim = 1

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + resource_dim + gate_dim, hidden_dim),
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
        # 1. 使用最后一个 token 的 hidden state 做门控
        last_hidden = hidden_states[:, -1, :]
        gated_hidden, gate_probs = self.gate(last_hidden)
        
        # 2. Transformer 聚合 (序列长度通常为 1)
        transformer_out = self.transformer(gated_hidden.unsqueeze(1))
        
        # 3. 拼接门控概率与资源特征
        combined_input = torch.cat([transformer_out, resource_states, gate_probs], dim=1)
        
        # 4. 共享特征提取
        features = self.shared_net(combined_input)
        
        # 5. 双头输出
        router_logits = self.router_head(features)
        router_probs = F.softmax(router_logits, dim=-1)
        
        caching_scores = torch.sigmoid(self.caching_head(features))
        
        return router_probs, caching_scores, gate_probs

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

        self.gate = KeyPointGate(input_dim)
        self.transformer = TransformerAggregator(input_dim)
        gate_dim = 1

        self.net = nn.Sequential(
            nn.Linear(input_dim + resource_dim + gate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出一个标量：价值 Value
        )
        
    def forward(self, hidden_states, resource_states):
        last_hidden = hidden_states[:, -1, :]
        gated_hidden, gate_probs = self.gate(last_hidden)
        transformer_out = self.transformer(gated_hidden.unsqueeze(1))
        combined = torch.cat([transformer_out, resource_states, gate_probs], dim=1)
        value = self.net(combined)
        return value
