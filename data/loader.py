"""
数据加载器: 从本地 assets 目录加载已下载的数据集
"""
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("I-PPO")


class DatasetLoader:
    """
    数据集加载器: 从本地加载已下载的数据集
    
    职责:
    - 检查数据集是否已下载
    - 从本地加载 ModelScope/HuggingFace 格式数据集
    - 不负责下载 (下载由 scripts/download_assets.py 完成)
    """
    
    def __init__(self, datasets_dir: str, datasets_config: List[tuple]):
        """
        Args:
            datasets_dir: 数据集存储目录
            datasets_config: 数据集配置列表
        """
        self.datasets_dir = datasets_dir
        self.datasets_config = datasets_config
        
    def load_all(self) -> List[Dict[str, Any]]:
        """
        加载所有配置的数据集
        
        Returns:
            数据池列表，每个元素是 {'dataset': str, 'question': str, 'answer': str}
        """
        from .processor import QAProcessor
        
        data_pool = []
        processor = QAProcessor()
        
        for dataset_config in self.datasets_config:
            # 解析配置
            if len(dataset_config) == 4:
                ds_name, subset, splits, source = dataset_config
            else:
                ds_name, subset, splits = dataset_config
                source = 'modelscope'
            
            if isinstance(splits, str):
                splits = [splits]
            
            for split in splits:
                try:
                    raw_data = self._load_single_dataset(ds_name, subset, split, source)
                    processed = processor.process_dataset(ds_name, raw_data)
                    data_pool.extend(processed)
                    logger.info(f"加载成功: {ds_name}/{subset or 'default'}/{split} - {len(processed)} 条")
                except Exception as e:
                    logger.error(f"加载失败: {ds_name}/{subset}/{split} - {e}")
                    continue
        
        if not data_pool:
            raise RuntimeError("无数据可用！请先运行 scripts/download_assets.py 下载数据集。")
        
        logger.info(f"数据集加载完成，共 {len(data_pool)} 条有效数据")
        return data_pool
    
    def _load_single_dataset(self, ds_name: str, subset: Optional[str], 
                              split: str, source: str) -> Any:
        """
        加载单个数据集
        
        优先从本地缓存加载，如果不存在则尝试在线加载
        """
        if source == 'huggingface':
            return self._load_huggingface(ds_name, subset, split)
        else:
            return self._load_modelscope(ds_name, subset, split)
    
    def _load_modelscope(self, ds_name: str, subset: Optional[str], split: str) -> Any:
        """加载 ModelScope 数据集"""
        from modelscope.msdatasets import MsDataset
        
        cache_dir = os.path.join(self.datasets_dir, "modelscope")
        os.makedirs(cache_dir, exist_ok=True)
        
        if subset:
            ds = MsDataset.load(ds_name, subset_name=subset, split=split, cache_dir=cache_dir)
        else:
            ds = MsDataset.load(ds_name, split=split, cache_dir=cache_dir)
        
        return ds
    
    def _load_huggingface(self, ds_name: str, subset: Optional[str], split: str) -> Any:
        """加载 HuggingFace 数据集"""
        from datasets import load_dataset
        
        cache_dir = os.path.join(self.datasets_dir, "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        
        if subset:
            ds = load_dataset(ds_name, subset, split=split, 
                            cache_dir=cache_dir, trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=split, 
                            cache_dir=cache_dir, trust_remote_code=True)
        
        return ds
    
    def check_datasets_exist(self) -> Dict[str, bool]:
        """
        检查各数据集是否已下载
        
        Returns:
            字典: {数据集名: 是否存在}
        """
        status = {}
        for dataset_config in self.datasets_config:
            ds_name = dataset_config[0]
            # 简单检查：尝试加载
            try:
                subset = dataset_config[1]
                splits = dataset_config[2]
                source = dataset_config[3] if len(dataset_config) > 3 else 'modelscope'
                split = splits[0] if isinstance(splits, list) else splits
                self._load_single_dataset(ds_name, subset, split, source)
                status[ds_name] = True
            except:
                status[ds_name] = False
        return status
