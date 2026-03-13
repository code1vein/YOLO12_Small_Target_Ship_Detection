import os
import shutil
import random
from pathlib import Path

import cv2
import albumentations as A
from tqdm import tqdm


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_label_lines(label_path: Path) -> list[str]:
    with open(label_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def label_has_small_target(label_path: Path) -> bool:
    for line in load_label_lines(label_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        if int(float(parts[0])) == 1:
            return True
    return False


def select_balanced_subset(
    image_names: list[str],
    src_labels_dir: Path,
    target_total: int,
    target_small_ratio: float,
) -> list[str]:
    has_small = []
    only_normal = []

    for img_name in image_names:
        label_path = src_labels_dir / f"{Path(img_name).stem}.txt"
        if label_has_small_target(label_path):
            has_small.append(img_name)
        else:
            only_normal.append(img_name)

    capped_target_total = min(target_total, len(image_names))
    target_small_count = min(len(has_small), int(round(capped_target_total * target_small_ratio)))
    selected_small = random.sample(has_small, target_small_count) if len(has_small) > target_small_count else list(has_small)

    remaining = capped_target_total - len(selected_small)
    target_normal_count = min(len(only_normal), remaining)
    selected_normal = random.sample(only_normal, target_normal_count) if len(only_normal) > target_normal_count else list(only_normal)

    selected = selected_small + selected_normal
    if len(selected) < capped_target_total:
        leftovers = [name for name in image_names if name not in set(selected)]
        needed = capped_target_total - len(selected)
        if leftovers:
            selected.extend(random.sample(leftovers, min(needed, len(leftovers))))

    random.shuffle(selected)
    return selected


def add_noise_augmented_split_samples(
    output_dir: Path,
    source_label_dir: Path,
    split_files: list[str],
    split_name: str,
) -> int:
    split_image_dir = output_dir / 'images' / split_name
    split_label_dir = output_dir / 'labels' / split_name

    small_first = []
    normal_only = []
    for img_name in split_files:
        label_path = source_label_dir / f"{Path(img_name).stem}.txt"
        if label_has_small_target(label_path):
            small_first.append(img_name)
        else:
            normal_only.append(img_name)

    candidate_files = small_first + normal_only
    # 按当前实验需求：对该划分中的所有图片都进行加噪并覆盖原图
    target_noise_count = len(candidate_files)
    if target_noise_count == 0:
        return 0

    selected_for_noise = candidate_files
    noise_transform = A.Compose([
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.10), mean_range=(0.0, 0.0), p=1.0),
            A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.25, 0.65), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), elementwise=True, p=1.0),
        ], p=1.0)
    ])

    generated_count = 0
    for img_name in tqdm(selected_for_noise, desc=f'生成 {split_name} 噪声样本'):
        image_path = split_image_dir / img_name
        label_path = split_label_dir / f"{Path(img_name).stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_path.exists():
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        noise_image = noise_transform(image=rgb_image)["image"]
        output_image = cv2.cvtColor(noise_image, cv2.COLOR_RGB2BGR)

        # 直接覆盖原图，保持文件名不变
        cv2.imwrite(str(image_path), output_image)
        generated_count += 1

    return generated_count

def split_dataset(
    dataset_dir: str, 
    images_dir_name: str = "images", 
    labels_dir_name: str = "labels_2class",
    output_dir_name: str = "yolo_dataset",
    ratios: tuple = (0.7, 0.2, 0.1),
    seed: int = 42,
    target_total_images: int | None = None,
    target_small_image_ratio: float = 0.5,
):
    """
    将数据集划分为 train, val, test 集 （7:2:1）
    """
    random.seed(seed)
    
    base_dir = Path(dataset_dir)
    src_images_dir = base_dir / images_dir_name
    src_labels_dir = base_dir / labels_dir_name
    
    if not src_images_dir.exists() or not src_labels_dir.exists():
        print(f"错误: 找不到源图片目录 {src_images_dir} 或源标签目录 {src_labels_dir}")
        return

    # 获取所有有对应标签的图片文件
    all_images = []
    
    print("正在扫描并匹配图片和标签文件...")
    for img_file in sorted(src_images_dir.iterdir(), key=lambda path: path.name):
        if img_file.suffix.lower() in VALID_EXTENSIONS:
            # 对应的标签文件名
            label_file = src_labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                all_images.append(img_file.name)
                
    total_files = len(all_images)
    print(f"找到 {total_files} 个具有对应标签的图片文件。")
    
    if total_files == 0:
        print("未找到任何成对的数据，退出。")
        return

    if target_total_images is not None and target_total_images > 0 and target_total_images < total_files:
        all_images = select_balanced_subset(
            image_names=all_images,
            src_labels_dir=src_labels_dir,
            target_total=target_total_images,
            target_small_ratio=target_small_image_ratio,
        )
        total_files = len(all_images)
        print(f"按受控抽样缩减到 {total_files} 张，目标是提升含小目标样本占比并压缩仅常规船舶样本。")
    else:
        random.shuffle(all_images)
    
    # 计算划分的索引
    train_end = int(total_files * ratios[0])
    val_end = train_end + int(total_files * ratios[1])
    
    train_files = all_images[:train_end]
    val_files = all_images[train_end:val_end]
    test_files = all_images[val_end:]
    
    print(f"按 {ratios[0]}:{ratios[1]}:{ratios[2]} 划分:")
    print(f"Train: {len(train_files)} 张")
    print(f"Val:   {len(val_files)} 张")
    print(f"Test:  {len(test_files)} 张")
    
    # 创建输出目录结构
    output_dir = base_dir / output_dir_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sub_dirs = ['train', 'val', 'test']
    
    for folder in ['images', 'labels']:
        for sub in sub_dirs:
            (output_dir / folder / sub).mkdir(parents=True, exist_ok=True)
            
    # 定义实际的拷贝函数
    def copy_files(file_list, split_name):
        for img_name in tqdm(file_list, desc=f"拷贝 {split_name} 集"):
            img_stem_name = Path(img_name).stem
            
            # 拷贝图片
            src_img = src_images_dir / img_name
            dst_img = output_dir / 'images' / split_name / img_name
            shutil.copy2(src_img, dst_img)
            
            # 拷贝对应的 txt 标签
            src_lbl = src_labels_dir / (img_stem_name + '.txt')
            dst_lbl = output_dir / 'labels' / split_name / (img_stem_name + '.txt')
            shutil.copy2(src_lbl, dst_lbl)

    print("\n开始整理文件结构...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    noise_count = add_noise_augmented_split_samples(
        output_dir=output_dir,
        source_label_dir=src_labels_dir,
        split_files=test_files,
        split_name='test',
    )
    if noise_count:
        print(f"已向 baseline 测试集补充 {noise_count} 张噪声增强样本")
    
    # 生成 YOLO 所需的 yaml 文件
    yaml_path = output_dir / 'ship_dataset.yaml'
    yaml_content = f"""# 船舶小目标检测数据集 (二分类)
# dataset_baseline/ship_dataset.yaml

path: {output_dir.absolute()}  # dataset root dir
train: images/train  # train images 
val: images/val  # val images
test: images/test  # test images (optional)

# Classes
names:
  0: Normal Ship      # 常规船舶 (面积 >= 0.5%)
  1: Small Target Ship # 小目标船舶 (面积 < 0.5%)
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"\n完成。数据集已输出至: {output_dir}")
    print(f"自动生成的 YOLO 配置文件路径: {yaml_path}")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import (
        BASELINE_TARGET_SMALL_IMAGE_RATIO,
        BASELINE_TARGET_TOTAL_IMAGES,
        DATA_ORIGINAL_DIR,
        DATA_BASELINE_DIR,
    )
    
    split_dataset(
        dataset_dir=str(DATA_ORIGINAL_DIR),
        images_dir_name="images",
        labels_dir_name="labels_2class",  # 使用我们刚刚生成的2分类标签
        output_dir_name=str(DATA_BASELINE_DIR),   # 指向统一配置中的 baseline 文件夹
        ratios=(0.7, 0.2, 0.1),           # 按照 7:2:1 划分
        target_total_images=BASELINE_TARGET_TOTAL_IMAGES,
        target_small_image_ratio=BASELINE_TARGET_SMALL_IMAGE_RATIO,
    )
