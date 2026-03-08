import os
from pathlib import Path

# 项目路径配置 

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集大目录
DATASET_ROOT = PROJECT_ROOT / "dataset"

# 各个独立状态的数据集目录
DATA_ORIGINAL_DIR = DATASET_ROOT / "dataset_original" # 原始未处理的数据集
DATA_BASELINE_DIR = DATASET_ROOT / "dataset_baseline" # 基准数据集（7：2：1）
DATA_AUGMENTATION_DIR = DATASET_ROOT / "dataset_augmentation" # 增强数据集 

# 原始数据集内部结构
ORIGINAL_IMAGES_DIR = DATA_ORIGINAL_DIR / "images"
ORIGINAL_LABELS_DIR = DATA_ORIGINAL_DIR / "labels"               # 原始多分类标签
ORIGINAL_LABELS_2CLASS_DIR = DATA_ORIGINAL_DIR / "labels_2class" # 转换后的2分类标签

# YAML 配置文件路径配置
BASELINE_YAML_PATH = DATA_BASELINE_DIR / "ship_dataset.yaml"





# 模型与训练配置

# 预训练权重选择
PRETRAINED_MODEL = "yolov12s.pt"

# 训练输出根目录
RUNS_DIR = PROJECT_ROOT / "runs"
TRAIN_RUNS_DIR = RUNS_DIR / "train"

# 小目标划分的相对面积占比阈值0.5%
SMALL_TARGET_THRESHOLD = 0.005

# 基础图像输入尺寸
IMG_SIZE = 640
