import logging
import os
from datetime import datetime

def setup_logger(name="ASA-PPO", log_dir="./logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 检查 handlers 是否已存在以避免重复日志
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 流处理器 (控制台)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        stream_handler.setFormatter(stream_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger