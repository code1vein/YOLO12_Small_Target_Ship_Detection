import sys
from pathlib import Path

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASELINE_YAML_PATH, YOLO12_MODEL
from train_config import train_model

def main():
    train_model(BASELINE_YAML_PATH, YOLO12_MODEL, 'exp0_baseline', '基线数据集')

if __name__ == '__main__':
    main()
