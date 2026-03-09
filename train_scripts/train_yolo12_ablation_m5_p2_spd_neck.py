import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import AUGMENTATION_YAML_PATH, YOLO12_ABLATION_M5_P2_SPD_NECK_CFG, YOLO12_MODEL
from train_config import train_model


def main():
    train_model(
        AUGMENTATION_YAML_PATH,
        YOLO12_MODEL,
        'exp15_yolo12_m5_p2_spd_neck',
        'YOLO12 消融 M5 P2 + SPD 风格下采样 + 增强 Neck 组',
        model_cfg=YOLO12_ABLATION_M5_P2_SPD_NECK_CFG,
        pretrained_weights=YOLO12_MODEL,
    )


if __name__ == '__main__':
    main()