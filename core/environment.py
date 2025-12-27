import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import random
import os
import time
from openai import OpenAI
import logging
from concurrent.futures import ThreadPoolExecutor
from configs.config import Config

logger = logging.getLogger("I-PPO")

class VectorizedRealSystemEnv:
    """向量化真实系统环境：支持多 GPU 并行采样。"""

    def __init__(self):
        # 1. 配置
        self.device = Config.DEVICE # 主设备 (用于 state tensor 等)
        self.gpu_ids = Config.GPU_IDS
        self.num_envs = Config.NUM_ENVS
        self.max_blocks = Config.MAX_BLOCKS
        self.block_size = Config.BLOCK_SIZE
        # 优先使用显式设定的生成长度限制，否则回退到物理块上限
        self.max_generation_tokens = getattr(Config, 'MAX_NEW_TOKENS', self.max_blocks * self.block_size)
        self.model_name = Config.MODEL_NAME
        self.hidden_dim = getattr(Config, 'INPUT_DIM', 2048)
        self.local_delay_coef = getattr(Config, 'LOCAL_DELAY_COEF', 0.0)
        self.cloud_delay_coef = getattr(Config, 'CLOUD_DELAY_COEF', 0.0)
        self.gate_penalty_coef = getattr(Config, 'GATE_PENALTY_COEF', 0.0)
        self.block_penalty_coef = getattr(Config, 'BLOCK_PENALTY_COEF', 0.0)

        # 云端 API
        self.cloud_api_key = Config.CLOUD_API_KEY
        self.cloud_base_url = Config.CLOUD_BASE_URL
        self.cloud_model_name = Config.CLOUD_MODEL_NAME

        if self.cloud_api_key == "sk-placeholder" or not self.cloud_api_key:
            logger.warning("未设置 CLOUD_API_KEY。云端调用将失败。")
            self.cloud_client = None
        else:
            self.cloud_client = OpenAI(api_key=self.cloud_api_key, base_url=self.cloud_base_url)

        # 2. 多 GPU 模型加载
        self.models = []
        self.env_model_map = [0] * self.num_envs # 映射: env_idx -> model_idx
        
        # 线程池用于并行推理 (workers = GPU 数量)
        self.num_gpus = len(self.gpu_ids) if self.gpu_ids else 1
        self.executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        
        logger.info(f"正在多 GPU ({self.num_gpus} 卡) 上加载模型: {self.model_name}...")
        
        try:
            # A. 准备本地路径
            if os.path.exists(Config.MODELS_DIR) and os.listdir(Config.MODELS_DIR):
                 from modelscope import snapshot_download
                 local_model_path = snapshot_download(self.model_name, cache_dir=Config.MODELS_DIR)
            else:
                 logger.warning("未检测到本地模型 assets，尝试自动下载...")
                 from modelscope import snapshot_download
                 local_model_path = snapshot_download(self.model_name)
            
            # B. 加载 Tokenizer (只需一份)
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # C. 加载 N 个模型实例到不同 GPU
            if not self.gpu_ids: # CPU 模式
                logger.warning("未检测到 GPU，使用 CPU 加载单模型。")
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    torch_dtype="auto",
                    device_map="cpu",
                    trust_remote_code=True
                ).eval()
                self.models.append(model)
            else:
                for idx, gpu_id in enumerate(self.gpu_ids):
                    device_str = f"cuda:{gpu_id}"
                    logger.info(f"-> Loading model replicas on {device_str}...")
                    model = AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        torch_dtype="auto",
                        device_map=None, # 手动 to(device)
                        trust_remote_code=True
                    ).to(device_str).eval()
                    self.models.append(model)
            
            # D. 建立环境分片映射
            # 简单均分: env 0-7 -> gpu 0, env 8-15 -> gpu 1 ...
            chunk_size = (self.num_envs + self.num_gpus - 1) // self.num_gpus
            for env_idx in range(self.num_envs):
                model_idx = env_idx // chunk_size
                if model_idx >= self.num_gpus: # 兜底
                    model_idx = self.num_gpus - 1
                self.env_model_map[env_idx] = model_idx
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

        # 3. 数据集
        logger.info("正在加载数据集 (使用 data 模块)...")
        from data.loader import DatasetLoader
        
        loader = DatasetLoader(Config.DATASETS_DIR, Config.DATASETS)
        self.data_pool = loader.load_all()
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # 4. 初始化状态
        self.envs_states = [self._init_single_env_state(i) for i in range(self.num_envs)]

    def _init_single_env_state(self, env_idx):
        """为单个环境采样问题并准备 prompt，执行统一 KV Block 初始化。"""
        sample = random.choice(self.data_pool)
        prompt_text = sample['question']
        
        # 1. 编码 Prompt
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False
        ).input_ids
        
        # 2. Prompt 分块
        # 将 flat list 转为 list of lists (Blocks)
        prompt_blocks = []
        for i in range(0, len(prompt_ids), self.block_size):
            block = prompt_ids[i : i + self.block_size]
            prompt_blocks.append(block)
            
        # 3. 初始化物理槽位 (Physical Slots)
        # 策略: Head + Tail 截断
        # 如果 Prompt 过长，保留开头的一小部分(Instruction)和结尾的大部分(Context + Question)
        active_blocks = []
        
        if len(prompt_blocks) > self.max_blocks:
            # 需要截断
            keep_ratio = 1.0
            # 这里简单起见，保留前 2 个 block (通常是 system prompt)，剩下的额度给尾部
            head_slots = min(2, len(prompt_blocks))
            tail_slots = self.max_blocks - head_slots
            
            # 选取 blocks
            head_blocks = prompt_blocks[:head_slots]
            tail_blocks = prompt_blocks[-tail_slots:]
            active_blocks = head_blocks + tail_blocks
            
            # 记录这是一种截断状态（可选：在这里给个负奖励或标记？）
        else:
            # 不需要截断，全部加载
            active_blocks = prompt_blocks[:]
            
        # 4. 构建 State
        # block_mask_status: 1 代表保留，0 代表驱逐 (Action 空间)
        # 此时所有 active_blocks 都被视为 "在缓存中" (Status = 1)
        # 注意：为了让 Agent 能控制它们，我们需要把它们统一放到管理的列表里
        
        state = {
            'env_idx': env_idx, # <--- 新增
            'question': prompt_text,
            'answer': sample['answer'],
            'raw_prompt_ids': prompt_ids, # 原始完整 Prompt，用于 Debug
            
            # --- 统一 KV 管理核心 ---
            'blocks': active_blocks,  # 当前物理内存中的 Block 内容列表 [[id, ...], [id, ...]]
            'block_types': ['prompt'] * len(active_blocks), # 标记类型: 'prompt' 或 'gen'
            'block_mask_status': [1] * len(active_blocks), # 当前时刻的掩码状态 (1=Active)
            
            'current_block': [], # 当前正在生成的非完整 Block
            'generated_token_count': 0, # 总生成长度计数
            
            # --- 其他状态 ---
            'network_delay': random.uniform(0.05, 0.5),
            'last_hidden_state': np.zeros(self.hidden_dim, dtype=np.float32),
            'last_entropy': 0.0,
            'next_token_logits': None,
            'done': False,
            'routed_to_cloud': False,
            'last_forward_latency': 0.0,
            'last_cloud_latency': 0.0
        }
        
        # 补齐 Mask 长度到 MAX_BLOCKS (为了 padding agent 输入)?
        # 暂时不用，Agent 这边可以处理变长，或者我们在 collector 里 pad。
        # 为了方便，config.MAX_BLOCKS 定义了最大物理容量，agent 输出维度是固定的。
        # 这里 logic 处理逻辑：agent 输出 100 维 mask。如果当前只有 5 个 block，我们只取前 5 个 mask。
        
        self._compute_prefix_state(state)
        return state

    def reset(self):
        """重置所有环境并返回初始观测。"""
        logger.info(f"正在重置所有 {self.num_envs} 个环境...")
        self.envs_states = [self._init_single_env_state(i) for i in range(self.num_envs)]
        return self._collect_batch_obs()

    def _reset_single_env(self, env_idx):
        self.envs_states[env_idx] = self._init_single_env_state(env_idx)

    def step(self, actions_route, actions_cache_mask, gate_usages):
        """逐 token 步进：批量处理所有环境的动作与生成。"""
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        if gate_usages is None:
            gate_usages = np.zeros(self.num_envs)

        # 1. 批量预处理 (Apply Actions & Calculate Penalties)
        # 获取需要本地生成的环境索引
        local_gen_indices = []
        
        for i in range(self.num_envs):
            state = self.envs_states[i]
            gate_usage = float(gate_usages[i]) if i < len(gate_usages) else 0.0
            
            # 基础惩罚计算
            gate_penalty = -self.gate_penalty_coef * gate_usage
            block_penalty = -self.block_penalty_coef * sum(state['block_mask_status'])
            
            # 保存惩罚项以便后续累加 (存到 info 或者临时变量? 这里简单点直接算)
            # 为了 Batch 处理方便，我们先把 penalty 记在 rewards 里，最后再加 base_reward
            rewards[i] = gate_penalty + block_penalty
            
            # A) 应用缓存动作
            self._apply_cache_mask(state, actions_cache_mask[i])
            
            # B) 路由决策
            if actions_route[i] == 1:
                # Cloud Route
                base_reward = self._finish_with_cloud(state)
                delay_penalty = -self.cloud_delay_coef * state.get('last_cloud_latency', 0.0)
                dones[i] = True
                infos[i]['route'] = 'cloud'
                rewards[i] += base_reward + delay_penalty
                self._reset_single_env(i)
                continue
            
            # C) 标记为本地生成
            local_gen_indices.append(i)

        # 2. 批量本地生成 (Batch Inference)
        if local_gen_indices:
            # 执行一次 Batch Forward
            batch_results = self._generate_batch_token(local_gen_indices)
            
            # 3. 分发结果并处理状态
            for idx, result in zip(local_gen_indices, batch_results):
                state = self.envs_states[idx]
                
                # 累加本地延迟惩罚
                delay_penalty = -self.local_delay_coef * state.get('last_forward_latency', 0.0)
                rewards[idx] += delay_penalty
                
                if result['oom']:
                    rewards[idx] += Config.LAMBDA_OOM
                    dones[idx] = True
                    infos[idx]['reason'] = 'oom'
                    self._reset_single_env(idx)
                    continue
                
                if result['finished']:
                    rewards[idx] += result['reward']
                    dones[idx] = True
                    infos[idx]['route'] = 'local'
                    self._reset_single_env(idx)
                else:
                    rewards[idx] += 0.0
                    dones[idx] = False

        next_obs = self._collect_batch_obs()
        return next_obs, rewards, dones, infos

    def _collect_batch_obs(self):
        hidden_states = []
        resource_states = []

        for state in self.envs_states:
            hidden_states.append(state['last_hidden_state'])
            
            # 计算资源特征
            # Active Blocks: 掩码为 1 的块数
            active_blocks = sum(state['block_mask_status'])
            
            # Generated Ratio: 用生成的 Token 数 / 最大生成步数
            generated_ratio = state['generated_token_count'] / max(1, self.max_generation_tokens)
            
            # Entropy Norm: 归一化熵
            entropy_norm = min(state['last_entropy'], 5.0) / 5.0
            
            resource_states.append([
                active_blocks / max(1, self.max_blocks), # 显存占用率
                generated_ratio,                         # 进度
                entropy_norm,                            # 不确定性
                state['network_delay']                   # 延迟
            ])

        return {
            'hidden_states': np.array(hidden_states, dtype=np.float32),
            'resource_states': np.array(resource_states, dtype=np.float32)
        }

    # --- Token 级别工具函数 ---
    # --- Token 级别工具函数 ---
    def _apply_cache_mask(self, state, mask_vector):
        """应用 Agent 的缓存掩码决策"""
        if not state['blocks']:
            return

        if mask_vector is None:
            return

        current_num_blocks = len(state['blocks'])
        
        # 截取 Agent 动作的前 N 位 (N = 当前 Block 数)
        # 假设 mask_vector 长度 >= current_num_blocks
        # 如果 mask_vector 短了(不应该发生)，则补 1
        raw_mask = mask_vector[:current_num_blocks]
        new_mask = [int(x) for x in raw_mask]
        
        # 异常防御：如果 Agent 输出维度不对
        if len(new_mask) < current_num_blocks:
            new_mask.extend([1] * (current_num_blocks - len(new_mask)))
            
        # 只有当 Mask 发生变化时才重新计算
        # 注意：这里对比的是 list 值
        if new_mask == state['block_mask_status']:
            return

        # 更新状态
        state['block_mask_status'] = new_mask
        
        # 重新计算前缀状态 (因为 KV 变了，Transformer 输出会变)
        # 优化：不需要在这里立即串行计算。_generate_batch_token 会使用最新的 active tokens 进行统一 Batch 计算。
        # self._compute_prefix_state(state)

    def _generate_batch_token(self, env_indices):
        """批量执行 Token 生成 (支持多 GPU 并行)"""
        
        # 1. 任务分组: {model_idx: [env_idx, ...]}
        tasks_by_model = {}
        for idx in env_indices:
            model_idx = self.env_model_map[idx]
            if model_idx not in tasks_by_model:
                tasks_by_model[model_idx] = []
            tasks_by_model[model_idx].append(idx)
            
        # 2. 并行执行
        results_map = {} # {env_idx: result_dict}
        futures = []
        
        # 定义单卡推理任务
        def _gpu_forward_task(model_idx, sub_indices):
            model = self.models[model_idx]
            return self._gpu_forward_impl(model, sub_indices)

        for model_idx, sub_indices in tasks_by_model.items():
            if self.num_gpus > 1:
                # 多卡模式：提交到线程池
                future = self.executor.submit(_gpu_forward_task, model_idx, sub_indices)
                futures.append(future)
            else:
                # 单卡模式：直接运行
                results_map.update(_gpu_forward_task(model_idx, sub_indices))
        
        # 3. 收集结果
        if self.num_gpus > 1:
            for future in futures:
                try:
                    res = future.result()
                    results_map.update(res)
                except Exception as e:
                    logger.error(f"GPU 推理线程异常: {e}")
                    raise e
                    
        # 4. 按输入顺序重组
        final_results = []
        for idx in env_indices:
            final_results.append(results_map[idx])
            
        return final_results

    def _gpu_forward_impl(self, model, env_indices):
        """单卡上的 Batch 推理实现"""
        # 1. 收集输入
        batch_input_ids = []
        for idx in env_indices:
            state = self.envs_states[idx]
            tokens = self._get_active_tokens(state)
            if not tokens: # 防御空
                tokens = [self.tokenizer.pad_token_id]
            batch_input_ids.append(tokens)
            
        # 2. Left Padding
        max_len = max(len(t) for t in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for tokens in batch_input_ids:
            pad_len = max_len - len(tokens)
            padded = [self.tokenizer.pad_token_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
            padded_input_ids.append(padded)
            attention_masks.append(mask)
            
        target_device = model.device
        input_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=target_device)
        mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=target_device)
        
        # 3. Batch Forward
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                output_hidden_states=True
            )
        elapsed = time.perf_counter() - start_time
        
        # 4. 解析结果
        next_token_logits = outputs.logits[:, -1, :] # (Batch, Vocab)
        last_hidden_states = outputs.hidden_states[-1][:, -1, :] # (Batch, Hidden)
        
        probs = torch.softmax(next_token_logits, dim=-1)
        sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
        
        # 5. 更新状态 (返回 dict)
        local_results = {}
        
        for i, idx in enumerate(env_indices):
            state = self.envs_states[idx]
            token_id = int(sampled_ids[i])
            
            # 状态更新
            state['last_forward_latency'] = elapsed
            state['last_hidden_state'] = last_hidden_states[i].detach().cpu().numpy().astype(np.float32)
            
            p = probs[i]
            entropy = -(p * torch.log(p + 1e-9)).sum().item()
            state['last_entropy'] = entropy
            state['next_token_logits'] = next_token_logits[i].detach().cpu()
            
            state['generated_tokens'].append(token_id)
            state['current_block'].append(token_id)
            state['generated_token_count'] += 1
            
            finished = False
            reward = 0.0
            oom_triggered = False
            
            if len(state['current_block']) >= self.block_size:
                new_block = state['current_block']
                state['blocks'].append(new_block)
                state['block_types'].append('gen')
                state['block_mask_status'].append(1)
                state['current_block'] = []
                
                if sum(state['block_mask_status']) > self.max_blocks:
                    oom_triggered = True
            
            if token_id == self.tokenizer.eos_token_id or state['generated_token_count'] >= self.max_generation_tokens:
                finished = True
                # 注意：_finalize_local_response 内部只做 CPU 字符串计算和 API 调用，不涉及 GPU
                reward = self._finalize_local_response(state)
                
            local_results[idx] = {
                'finished': finished,
                'reward': reward,
                'oom': oom_triggered
            }
            
        return local_results

    def _compute_prefix_state(self, state):
        # 1. 获取对应的模型和设备
        env_idx = state.get('env_idx', 0) # 默认为 0 (容错)
        model_idx = self.env_model_map[env_idx]
        model = self.models[model_idx]
        target_device = model.device # 使用模型的 device
        
        # 统一从 blocks 中提取 Active Tokens
        full_tokens = self._get_active_tokens(state)
        
        if not full_tokens:
            full_tokens = [self.tokenizer.pad_token_id]

        input_ids = torch.tensor(full_tokens, dtype=torch.long, device=target_device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, device=target_device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        state['last_forward_latency'] = time.perf_counter() - start

        # 注意：这里把结果搬回 CPU (主设备)
        last_hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0).detach().cpu().numpy()
        logits = outputs.logits[:, -1, :].squeeze(0).detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

        state['last_hidden_state'] = last_hidden.astype(np.float32)
        state['last_entropy'] = entropy
        state['next_token_logits'] = logits

    def _get_active_tokens(self, state):
        """拼接当前物理内存中保留的所有 Block + 当前未成块的 Token"""
        active = []
        
        # 1. 遍历已完成的 Blocks (Prompt + Generated)
        # state['blocks'] 和 state['block_mask_status'] 是一一对应的
        for block, keep in zip(state['blocks'], state['block_mask_status']):
            if keep:
                active.extend(block)
                
        # 2. 加上当前正在生成的半块 (Current Block)
        # 这部分通常必须保留 (除非策略允许丢弃最新的? 一般不)
        if state['current_block']:
            active.extend(state['current_block'])
            
        return active

    def _finalize_local_response(self, state):
        text = self.tokenizer.decode(state['generated_tokens'], skip_special_tokens=True).strip()
        judge = self._judge_answer(state['answer'], text)
        if judge is True:
            return 1.0
        if judge is False:
            return -1.0
        # 回退: 使用 Rouge 兜底
        score = self._calc_score(text, state['answer'])
        return 1.0 if score > 0.5 else -1.0

    def _finish_with_cloud(self, state):
        start = time.perf_counter()
        cloud_answer = self._call_cloud_api(state['question'])
        state['last_cloud_latency'] = time.perf_counter() - start
        if not cloud_answer:
            return -0.5
        judge = self._judge_answer(state['answer'], cloud_answer)
        if judge is None:
            score = self._calc_score(cloud_answer, state['answer'])
            correct = score > 0.5
        else:
            correct = judge
        reward = 1.0 if correct else 0.0
        reward -= 0.1  # 网络惩罚
        return reward

    # 辅助函数
    # (数据加载逻辑已移至 data 模块)

    def _calc_score(self, pred, ref):
        scores = self.scorer.score(ref, pred)
        return scores['rougeL'].fmeasure

    def _call_cloud_api(self, prompt):
        # ... (逻辑相同) ...
        if not self.cloud_client: return None
        try:
            response = self.cloud_client.chat.completions.create(
                model=self.cloud_model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant. Answer briefly."},
                          {"role": "user", "content": prompt}],
                max_tokens=256, temperature=0.7, stream=False
            )
            return response.choices[0].message.content.strip()
        except: return None

    def _judge_answer(self, reference, candidate):
        """调用一次大模型判断生成答案是否正确，失败时返回 None。"""
        if not candidate:
            return False
        if not self.cloud_client:
            return None
        try:
            prompt = (
                "你是一个答案判定器。请判断 candidate 是否与 reference 意义上一致。\n"
                f"reference: {reference}\n"
                f"candidate: {candidate}\n"
                "如果一致，仅回答 'correct'；否则仅回答 'incorrect'。"
            )
            response = self.cloud_client.chat.completions.create(
                model=self.cloud_model_name,
                messages=[
                    {"role": "system", "content": "You judge whether two answers match."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3,
                temperature=0.0,
                stream=False
            )
            content = response.choices[0].message.content.strip().lower()
            if "correct" in content and "incorrect" not in content:
                return True
            if "incorrect" in content:
                return False
        except Exception as e:
            logger.warning(f"云端判定失败: {e}")
        return None
