from pathlib import Path

# 项目路径配置 

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

#消融实验模型目录
MODEL_ROOT = PROJECT_ROOT / "models"
ABLATION_MODEL_DIR = MODEL_ROOT / "ablation"

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
AUGMENTATION_YAML_PATH = DATA_AUGMENTATION_DIR / "ship_dataset.yaml"





# 模型与训练配置

# 预训练权重选择
YOLO12_MODEL = "yolo12m.pt"
YOLO11_MODEL = "yolo11m.pt"
YOLO8_MODEL = "yolov8m.pt"
YOLO5_MODEL = "yolov5mu.pt"  
PRETRAINED_MODEL = YOLO12_MODEL

# 训练输出根目录
RUNS_DIR = PROJECT_ROOT / "runs"
TRAIN_RUNS_DIR = RUNS_DIR / "train"

# YOLOv12 消融实验模型配置
YOLO12_ABLATION_M2_P2_CFG = ABLATION_MODEL_DIR / "yolo12m_m2_p2.yaml"
YOLO12_ABLATION_M3_SPD_CFG = ABLATION_MODEL_DIR / "yolo12m_m3_spd.yaml"
YOLO12_ABLATION_M4_NECK_CFG = ABLATION_MODEL_DIR / "yolo12m_m4_neck.yaml"
YOLO12_ABLATION_M5_P2_SPD_CFG = ABLATION_MODEL_DIR / "yolo12m_m5_p2_spd.yaml"
YOLO12_ABLATION_M6_P2_SPD_NECK_CFG = ABLATION_MODEL_DIR / "yolo12m_m6_p2_spd_neck.yaml"

# 小目标划分的相对面积占比阈值0.5%
SMALL_TARGET_THRESHOLD = 0.005
# 在阈值边界附近保留缓冲带，避免统计波动导致标签抖动
SMALL_TARGET_BORDER_MARGIN = 0.0005
SAFE_SMALL_TARGET_THRESHOLD = SMALL_TARGET_THRESHOLD - SMALL_TARGET_BORDER_MARGIN

# 增强集分布控制参数
COPYPASTE_TARGET_SMALL_RATIO = 0.55

# 增强数据集规模控制
AUGMENTATION_TARGET_TOTAL_IMAGES = 11048
AUGMENTATION_WEATHER_BUDGET_RATIO = 0.8
AUGMENTATION_COPYPASTE_BUDGET_RATIO = 0.17

# 基础图像输入尺寸
IMG_SIZE = 640
