import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    AUGMENTATION_COPYPASTE_BUDGET_RATIO,
    AUGMENTATION_TARGET_TOTAL_IMAGES,
    AUGMENTATION_WEATHER_BUDGET_RATIO,
    COPYPASTE_TARGET_SMALL_RATIO,
    DATA_AUGMENTATION_DIR,
    DATA_BASELINE_DIR,
)
from dataset.augmentation.augmentation_utils import (
    count_dataset_images,
    collect_dataset_label_stats,
    summarize_ratio,
)
from dataset.augmentation.copypaste_data import create_copypaste_dataset
from dataset.augmentation.integrate_aigc_data import main as integrate_aigc_data
from dataset.augmentation.weather_simulation import create_weather_augmented_dataset


# 统一串联天气增强、Copy-Paste 和 AIGC 整合三个阶段
def build_augmentation_dataset(
    weather_ratio: float = 0.3,
    copypaste_ratio: float = 0.2,
    max_paste: int = 2,
    target_small_ratio: float = COPYPASTE_TARGET_SMALL_RATIO,
    target_total_images: int = AUGMENTATION_TARGET_TOTAL_IMAGES,
    seed: int = 42,
) -> None:
    if not DATA_BASELINE_DIR.exists():
        print(f"错误：找不到基线数据集 {DATA_BASELINE_DIR}")
        return

    random.seed(seed)
    np.random.seed(seed)

    baseline_total_images = count_dataset_images(
        DATA_BASELINE_DIR,
        excluded_suffixes=("_aug_noise",),
    )
    remaining_budget = max(0, target_total_images - baseline_total_images)
    weather_budget = int(round(remaining_budget * AUGMENTATION_WEATHER_BUDGET_RATIO))
    copypaste_budget = int(round(remaining_budget * AUGMENTATION_COPYPASTE_BUDGET_RATIO))
    aigc_budget = max(0, remaining_budget - weather_budget - copypaste_budget)

    if DATA_AUGMENTATION_DIR.exists():
        shutil.rmtree(DATA_AUGMENTATION_DIR)

    print("=" * 60)
    print("开始从 baseline 数据集构建 augmentation 数据集")
    print(f"  - baseline: {DATA_BASELINE_DIR}")
    print(f"  - augmentation: {DATA_AUGMENTATION_DIR}")
    print(f"  - 随机种子: {seed}")
    print(f"  - baseline 总图片数: {baseline_total_images}")
    print(f"  - augmentation 目标总图片数: {target_total_images}")
    print(f"  - 可新增图片预算: {remaining_budget}")
    print(f"  - weather 预算: {weather_budget}")
    print(f"  - copy-paste 预算: {copypaste_budget}")
    print(f"  - AIGC 预算: {aigc_budget}")
    print("=" * 60)

    # 第一步：从 baseline 原图生成天气扰动样本
    print("\n[1/3] 构建天气增强样本")
    create_weather_augmented_dataset(
        enhancement_ratio=weather_ratio,
        max_samples=weather_budget,
    )

    # 第二步：继续向增强池补充受控的 Copy-Paste 样本
    print("\n[2/3] 构建 Copy-Paste 样本")
    create_copypaste_dataset(
        augment_ratio=copypaste_ratio,
        max_paste=max_paste,
        target_small_ratio=target_small_ratio,
        max_samples=copypaste_budget,
    )

    # 第三步：将 AIGC 样本转成 YOLO 格式后并入增强池
    print("\n[3/3] 整合 AIGC 样本")
    integrate_aigc_data(max_samples=aigc_budget, seed=seed)

    # 最后统计增强训练集整体标签分布，便于检查是否偏移过大
    label_dir = DATA_AUGMENTATION_DIR / "labels" / "train"
    file_count, normal_count, small_count = collect_dataset_label_stats(label_dir)
    print("\n" + "=" * 60)
    print("augmentation 数据集构建完成")
    print(f"  - 总图片数: {count_dataset_images(DATA_AUGMENTATION_DIR)}")
    print(f"  - 训练标签文件数: {file_count}")
    print(f"  - normal ship: {normal_count}")
    print(f"  - small target ship: {small_count}")
    print(f"  - small target 占比: {summarize_ratio(normal_count, small_count)}")
    print("=" * 60)


if __name__ == "__main__":
    build_augmentation_dataset()