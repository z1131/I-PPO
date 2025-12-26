
import os
import sys
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset

# Add project root to sys.path to allow importing configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config

def download_model():
    """
    下载并缓存模型。
    """
    model_name = Config.MODEL_NAME
    print(f"=== 开始下载模型: {model_name} ===")
    
    try:
        # 使用 snapshot_download 下载模型
        print("正在下载模型 (这就比较大了，耐心点)...")
        model_dir = snapshot_download(model_name)
        print(f"=== 模型 {model_name} 下载完成 ===")
        print(f"模型保存在: {model_dir}")
    except Exception as e:
        print(f"!!! 模型下载失败: {e}")

def download_datasets():
    """
    下载并缓存实验所需的所有数据集。
    """
    print("\n=== 开始下载数据集 ===")
    
    datasets_list = Config.DATASETS
    
    for ds_name, subset, splits in datasets_list:
        if isinstance(splits, str):
            splits = [splits]
            
        for split in splits:
            print(f"正在下载: {ds_name} (subset={subset}, split={split})...")
            try:
                if subset:
                    ds = MsDataset.load(ds_name, subset_name=subset, split=split)
                else:
                    ds = MsDataset.load(ds_name, split=split)
                print(f" -> 成功! 样本数: {len(ds)}")
            except Exception as e:
                print(f" -> 失败: {e}")
            
    print("=== 所有数据集下载尝试结束 ===")

if __name__ == "__main__":
    print("##################################################")
    print("#         ASA-PPO 实验资源预下载脚本             #")
    print("#  运行此脚本以避免在训练时占用宝贵的计算时间    #")
    print("##################################################\n")
    
    download_model()
    download_datasets()
    
    print("\n[完成] 所有资源应当已缓存到本地。")
