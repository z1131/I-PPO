import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.msdatasets import MsDataset
from rouge_score import rouge_scorer
import random
import os
import pandas as pd
import glob
from openai import OpenAI
import logging
import math
from configs.config import Config

logger = logging.getLogger("ASA-PPO")

class VectorizedRealSystemEnv:
    """
    【向量化环境】Vectorized Environment
    同时管理 NUM_ENVS (例如 8 或 32) 个并行的游戏进程。
    共享同一个 Qwen 模型，极大提升 GPU 利用率。
    """
    def __init__(self):
        # 1. 加载配置
        self.device = Config.DEVICE
        self.num_envs = Config.NUM_ENVS
        self.max_blocks = Config.MAX_BLOCKS
        self.block_size = Config.BLOCK_SIZE
        self.max_turns = Config.MAX_TURNS
        self.model_name = Config.MODEL_NAME
        
        # 云端 API
        self.cloud_api_key = Config.CLOUD_API_KEY
        self.cloud_base_url = Config.CLOUD_BASE_URL
        self.cloud_model_name = Config.CLOUD_MODEL_NAME
        
        if self.cloud_api_key == "sk-placeholder" or not self.cloud_api_key:
            logger.warning("未设置 CLOUD_API_KEY。云端调用将失败。")
            self.cloud_client = None
        else:
            self.cloud_client = OpenAI(api_key=self.cloud_api_key, base_url=self.cloud_base_url)

        # 2. 加载模型与 Tokenizer (只加载一次)
        logger.info(f"正在加载共享模型: {self.model_name}...")
        try:
            from modelscope import snapshot_download
            local_model_path = snapshot_download(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
            self.tokenizer.padding_side = 'left' # 批量生成必须左填充
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype="auto", # 回归模型的默认精度
                device_map="auto",
                trust_remote_code=True
                # 为了学术严谨性，移除了 attn_implementation="flash_attention_2"
            )
            # 确保存在 padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

        # 3. 加载数据集
        logger.info("正在加载数据集...")
        self.data_pool = self._load_mixed_datasets()
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # 4. 初始化向量化状态 (大小为 NUM_ENVS 的列表)
        self.envs_states = [self._init_single_env_state() for _ in range(self.num_envs)]

    def _init_single_env_state(self):
        """初始化单个子环境的空白状态"""
        return {
            'history_token_ids': [],    # 绝对历史 Token ID
            'block_mask_status': [],    # 0/1 状态
            'current_sample': None,     # 当前问答对
            'turn_count': 0,
            'network_delay': random.uniform(0.05, 1.0),
            'last_hidden_state': None,  # 缓存用于观测
            'last_entropy': 0.0,
            'done': False               # 内部完成标志
        }

    def reset(self):
        """
        全局重置: 重置所有环境。
        返回: 观测批次 (Batch of Observations)
        """
        logger.info(f"正在重置所有 {self.num_envs} 个环境...")
        for i in range(self.num_envs):
            self._reset_single_env(i)
        
        # 执行初始观测的批次计算
        return self._get_batch_obs()

    def _reset_single_env(self, env_idx):
        """重置第 i 个环境的状态"""
        sample = random.choice(self.data_pool)
        self.envs_states[env_idx] = {
            'history_token_ids': [],
            'block_mask_status': [],
            'current_sample': sample,
            'turn_count': 0,
            'network_delay': random.uniform(0.05, 1.0),
            'last_hidden_state': None, 
            'last_entropy': 0.0,
            'done': False
        }
        # 注意: 我们会在 _get_batch_obs 中懒加载计算 hidden state。
        # 为了确保第一个 obs 有效，我们通常需要对问题运行模型。
        # 优化: 我们将 'last_hidden_state' 标记为 None 并在 _get_batch_obs 中处理。

    def step(self, actions_route, actions_cache_mask):
        """
        向量化步进函数 (Vectorized Step Function)。
        参数:
            actions_route: (NUM_ENVS,) 整数数组
            actions_cache_mask: (NUM_ENVS, MAX_BLOCKS) 整数数组
        返回:
            next_obs_batch, rewards, dones, infos
        """
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        # --- 1. 识别请求 ---
        # 我们需要为所有活跃环境运行本地生成以更新历史，
        # 无论 Agent 选择了 Local 还是 Cloud (反事实推断需要 Local 结果)。
        # 所以: 我们必须为每个人运行本地生成。
        
        # --- 2. 根据动作更新 Mask ---
        for i in range(self.num_envs):
            current_num_blocks = len(self.envs_states[i]['block_mask_status'])
            if current_num_blocks > 0:
                # 从动作 Mask 中取前 N 位
                new_status = actions_cache_mask[i][:current_num_blocks]
                self.envs_states[i]['block_mask_status'] = new_status.tolist()

        # --- 3. 批量本地生成 (重活) ---
        # 准备批量输入
        prompts = [state['current_sample']['question'] for state in self.envs_states]
        
        # Tokenize 新问题
        new_q_ids_batch = [self.tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
        
        # 运行批量生成
        gen_texts, gen_token_ids_batch = self._batch_generate_masked(new_q_ids_batch)
        
        # --- 4. 处理结果 & 计算奖励 ---
        for i in range(self.num_envs):
            state = self.envs_states[i]
            state['turn_count'] += 1
            
            ground_truth = state['current_sample']['answer']
            local_ans = gen_texts[i]
            local_ids = gen_token_ids_batch[i]
            
            # 更新历史
            state['history_token_ids'].extend(new_q_ids_batch[i])
            state['history_token_ids'].extend(local_ids)
            
            # 更新 Block 计数
            total_tokens = len(state['history_token_ids'])
            new_total_blocks = math.ceil(total_tokens / self.block_size)
            blocks_to_add = new_total_blocks - len(state['block_mask_status'])
            if blocks_to_add > 0:
                state['block_mask_status'].extend([1] * blocks_to_add)

            # 本地打分
            local_score = self._calc_score(local_ans, ground_truth)
            correct_local = 1 if local_score > 0.5 else 0
            
            # 检查云端 (串行执行 API 以避免速率限制/复杂性)
            # 仅在需要时调用
            route = actions_route[i]
            correct_cloud = 1 # 默认 (反事实)
            
            need_cloud = (route == 1) or (route == 0 and correct_local == 0)
            if need_cloud:
                cloud_ans = self._call_cloud_api(state['current_sample']['question'])
                if cloud_ans:
                    cloud_score = self._calc_score(cloud_ans, ground_truth)
                    correct_cloud = 1 if cloud_score > 0.5 else 0
            
            # 计算奖励 & OOM
            active_blocks = sum(state['block_mask_status'])
            is_oom = active_blocks > self.max_blocks
            
            r = 0.0
            if is_oom:
                r = Config.LAMBDA_OOM
                # logger.debug(f"环境 {i} OOM")
            else:
                if route == 0: # Local
                    if correct_local == 1: r = 1.0
                    else: r = -1.0 if correct_cloud == 1 else 0.0
                else: # Cloud
                    if correct_local == 1: r = 0.0
                    elif correct_cloud == 1: r = 1.0
                    else: r = 0.0
                
                # 延迟惩罚
                if route == 1:
                    r -= 0.1
                    if correct_local == 1: r -= 0.5
            
            rewards[i] = r
            
            # 检查 Done
            done = is_oom or (state['turn_count'] >= self.max_turns)
            dones[i] = done
            
            if done:
                # 自动重置这个特定的环境
                # 这确保了向量环境总是产生有效的 next_states
                self._reset_single_env(i)
                infos[i]['terminal_observation'] = True # 标记

        # --- 5. 计算下一观测 (批量) ---
        # 状态已更新(或重置)，计算*新*样本的隐藏状态
        next_obs = self._get_batch_obs()
        
        return next_obs, rewards, dones, infos

    def _batch_generate_masked(self, new_q_ids_batch):
        """
        使用真实 Masking 对整个 Batch 运行生成。
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        for i in range(self.num_envs):
            state = self.envs_states[i]
            hist = state['history_token_ids']
            new_q = new_q_ids_batch[i]
            mask_status = state['block_mask_status']
            
            # 构造完整输入
            full_input = hist + new_q
            
            # 构造完整 Mask
            h_mask = []
            for b_idx, status in enumerate(mask_status):
                start = b_idx * self.block_size
                end = min((b_idx + 1) * self.block_size, len(hist))
                h_mask.extend([status] * (end - start))
            
            # 修正 Mask 长度不匹配 (以防万一)
            if len(h_mask) > len(hist): h_mask = h_mask[:len(hist)]
            if len(h_mask) < len(hist): h_mask.extend([1] * (len(hist) - len(h_mask)))
                
            full_mask = h_mask + [1] * len(new_q)
            
            batch_input_ids.append(torch.tensor(full_input, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(full_mask, dtype=torch.long))

        # Pad Sequence
        padded_input = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        padded_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=padded_input,
                    attention_mask=padded_mask,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            # 回退: 返回空
            return ["Error"]*self.num_envs, [[]]*self.num_envs

        # 提取结果
        gen_texts = []
        gen_ids = []
        
        for i in range(self.num_envs):
            in_len = len(batch_input_ids[i])
            out_ids = outputs[i][in_len:].tolist()
            
            # 去除 padding
            # 如果存在 eos 则截断
            if self.tokenizer.eos_token_id in out_ids:
                eos_idx = out_ids.index(self.tokenizer.eos_token_id)
                out_ids = out_ids[:eos_idx]
                
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            
            gen_texts.append(text)
            gen_ids.append(out_ids)
            
        return gen_texts, gen_ids

    def _get_batch_obs(self):
        """
        在一次前向传播中计算所有环境的观测值 (Hidden State + Entropy + Resource)。
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        # 准备输入: History + Question
        for i in range(self.num_envs):
            state = self.envs_states[i]
            hist = state['history_token_ids']
            
            # 当前问题
            q_text = state['current_sample']['question']
            q_ids = self.tokenizer(q_text, add_special_tokens=False).input_ids
            
            full_input = hist + q_ids
            
            # Mask
            h_mask = []
            for b_idx, status in enumerate(state['block_mask_status']):
                start = b_idx * self.block_size
                end = min((b_idx + 1) * self.block_size, len(hist))
                h_mask.extend([status] * (end - start))
            
            # 修正 Mask 长度
            if len(h_mask) < len(hist): h_mask.extend([1]*(len(hist)-len(h_mask)))
            h_mask = h_mask[:len(hist)]
                
            full_mask = h_mask + [1] * len(q_ids)
            
            batch_input_ids.append(torch.tensor(full_input, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(full_mask, dtype=torch.long))

        # Pad
        padded_input = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        padded_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(padded_input, attention_mask=padded_mask, output_hidden_states=True)
            
        # 提取 Hidden States & Entropy
        # 需要为每个样本选择正确的索引 (length - 1)
        hidden_states_list = []
        resource_states_list = []
        
        last_hidden = outputs.hidden_states[-1] # (Batch, Seq, Dim)
        logits = outputs.logits # (Batch, Seq, Vocab)
        
        for i in range(self.num_envs):
            true_len = len(batch_input_ids[i])
            # 最后一个有效 token 的索引是 true_len - 1
            idx = true_len - 1
            
            h = last_hidden[i, idx, :].float().cpu().numpy() # (Dim,)
            
            l = logits[i, idx, :]
            probs = torch.softmax(l, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            entropy = np.clip(entropy, 0.0, 5.0)
            
            state = self.envs_states[i]
            active_blocks = sum(state['block_mask_status'])
            
            hidden_states_list.append(h)
            resource_states_list.append([
                active_blocks / self.max_blocks,
                state['network_delay'],
                true_len, # 任务长度
                entropy / 5.0
            ])
            
            # 缓存状态跟踪 (如果需要)
            state['last_hidden_state'] = h
            state['last_entropy'] = entropy

        return {
            'hidden_states': np.array(hidden_states_list),      # (Num_Envs, Dim)
            'resource_states': np.array(resource_states_list)   # (Num_Envs, 4)
        }

    # 辅助函数
    def _load_mixed_datasets(self):
        # ... (与之前的实现相同) ...
        data_pool = []
        for ds_name, subset, splits in Config.DATASETS:
            if isinstance(splits, str): splits = [splits]
            for split in splits:
                if 'openbookqa' in ds_name:
                    self._load_openbookqa(data_pool, split)
                    continue
                try:
                    if subset: ds = MsDataset.load(ds_name, subset_name=subset, split=split)
                    else: ds = MsDataset.load(ds_name, split=split)
                    for item in ds:
                        q = item.get('question', item.get('problem', item.get('question_stem', '')))
                        a = item.get('answerKey', item.get('solution', item.get('answer', '')))
                        if q and a: data_pool.append({'dataset': ds_name, 'question': q, 'answer': a})
                except: pass
        if not data_pool: raise RuntimeError("无数据")
        return data_pool

    def _load_openbookqa(self, data_pool, split):
        # ... (相同的鲁棒逻辑) ...
        try:
            cache_dir = os.path.expanduser("~/.cache/modelscope/hub/datasets/downloads")
            hashes = Config.OBQA_HASHES
            for target_hash in hashes:
                target_file = os.path.join(cache_dir, target_hash)
                if not os.path.exists(target_file):
                    for jf in glob.glob(os.path.join(cache_dir, "*.json")):
                         with open(jf,'r') as f:
                             if target_hash in jf: target_file = jf.replace(".json",""); break
                if os.path.exists(target_file):
                    try:
                        df = pd.read_parquet(target_file)
                        for _, row in df.iterrows():
                            data_pool.append({'dataset':'openbookqa','question':row['question_stem'],'answer':row['answerKey']})
                    except: pass
        except: pass

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