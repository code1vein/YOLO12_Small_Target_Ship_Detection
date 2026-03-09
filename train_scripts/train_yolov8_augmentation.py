import sys
from pathlib import Path

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import AUGMENTATION_YAML_PATH, YOLO8_MODEL
from train_config import train_model


def main():
    train_model(AUGMENTATION_YAML_PATH, YOLO8_MODEL, 'exp3_yolov8_augmentation', '增强数据集')


if __name__ == '__main__':
    main()