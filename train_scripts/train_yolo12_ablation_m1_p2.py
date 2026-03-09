import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import AUGMENTATION_YAML_PATH, YOLO12_ABLATION_M1_P2_CFG, YOLO12_MODEL
from train_config import train_model


def main():
    train_model(
        AUGMENTATION_YAML_PATH,
        YOLO12_MODEL,
        'exp11_yolo12_m1_p2',
        'YOLO12 消融 M1 P2 检测头组',
        model_cfg=YOLO12_ABLATION_M1_P2_CFG,
        pretrained_weights=YOLO12_MODEL,
    )


if __name__ == '__main__':
    main()