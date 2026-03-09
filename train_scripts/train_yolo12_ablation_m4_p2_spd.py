import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import AUGMENTATION_YAML_PATH, YOLO12_ABLATION_M4_P2_SPD_CFG, YOLO12_MODEL
from train_config import train_model


def main():
    train_model(
        AUGMENTATION_YAML_PATH,
        YOLO12_MODEL,
        'exp14_yolo12_m4_p2_spd',
        'YOLO12 消融 M4 P2 + SPD 风格下采样组',
        model_cfg=YOLO12_ABLATION_M4_P2_SPD_CFG,
        pretrained_weights=YOLO12_MODEL,
    )


if __name__ == '__main__':
    main()