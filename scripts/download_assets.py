#!/usr/bin/env python3
"""
I-PPO 资源下载脚本

功能:
1. 下载模型到 assets/models/
2. 下载数据集到 assets/datasets/
3. 支持 ModelScope 和 HuggingFace 双源

使用方法:
    python scripts/download_assets.py           # 下载全部
    python scripts/download_assets.py --model   # 仅下载模型
    python scripts/download_assets.py --data    # 仅下载数据集
"""
import os
import sys
import argparse

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config


def download_model():
    """下载模型到 assets/models/"""
    from modelscope import snapshot_download
    
    model_name = Config.MODEL_NAME
    model_dir = Config.MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"目标目录: {model_dir}")
    print(f"{'='*60}\n")
    
    try:
        local_path = snapshot_download(model_name, cache_dir=model_dir)
        print(f"✓ 模型下载完成: {local_path}")
        return True
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        return False


def download_datasets():
    """下载所有数据集到 assets/datasets/"""
    datasets_dir = Config.DATASETS_DIR
    os.makedirs(datasets_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"下载数据集")
    print(f"目标目录: {datasets_dir}")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for dataset_config in Config.DATASETS:
        # 解析配置
        if len(dataset_config) == 4:
            ds_name, subset, splits, source = dataset_config
        else:
            ds_name, subset, splits = dataset_config
            source = 'modelscope'
        
        if isinstance(splits, str):
            splits = [splits]
        
        for split in splits:
            dataset_id = f"{ds_name}/{subset or 'default'}/{split}"
            print(f"下载中: {dataset_id} ({source})...")
            
            try:
                if source == 'huggingface':
                    _download_huggingface(ds_name, subset, split, datasets_dir)
                else:
                    _download_modelscope(ds_name, subset, split, datasets_dir)
                
                print(f"  ✓ 成功")
                success_count += 1
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"数据集下载完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"{'='*60}\n")
    
    return fail_count == 0


def _download_modelscope(ds_name: str, subset: str, split: str, base_dir: str):
    """下载 ModelScope 数据集"""
    from modelscope.msdatasets import MsDataset
    
    cache_dir = os.path.join(base_dir, "modelscope")
    os.makedirs(cache_dir, exist_ok=True)
    
    if subset:
        ds = MsDataset.load(ds_name, subset_name=subset, split=split, cache_dir=cache_dir)
    else:
        ds = MsDataset.load(ds_name, split=split, cache_dir=cache_dir)
    
    print(f"    样本数: {len(ds)}")


def _download_huggingface(ds_name: str, subset: str, split: str, base_dir: str):
    """下载 HuggingFace 数据集"""
    from datasets import load_dataset
    
    cache_dir = os.path.join(base_dir, "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    if subset:
        ds = load_dataset(ds_name, subset, split=split, 
                         cache_dir=cache_dir, trust_remote_code=True)
    else:
        ds = load_dataset(ds_name, split=split, 
                         cache_dir=cache_dir, trust_remote_code=True)
    
    print(f"    样本数: {len(ds)}")


def main():
    parser = argparse.ArgumentParser(description='I-PPO 资源下载脚本')
    parser.add_argument('--model', action='store_true', help='仅下载模型')
    parser.add_argument('--data', action='store_true', help='仅下载数据集')
    args = parser.parse_args()
    
    print("#" * 60)
    print("#          I-PPO 实验资源预下载脚本                       #")
    print("#  运行此脚本以避免在训练时占用宝贵的计算时间             #")
    print("#" * 60)
    
    # 如果没有指定参数，则下载全部
    download_all = not args.model and not args.data
    
    success = True
    
    if download_all or args.model:
        success = download_model() and success
    
    if download_all or args.data:
        success = download_datasets() and success
    
    if success:
        print("\n[完成] 所有资源已下载到 assets/ 目录。")
    else:
        print("\n[警告] 部分资源下载失败，请检查网络连接后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
