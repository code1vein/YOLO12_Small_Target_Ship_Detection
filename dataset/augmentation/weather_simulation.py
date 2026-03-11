import cv2
import random
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import sys

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_BASELINE_DIR, DATA_AUGMENTATION_DIR
from dataset.augmentation.augmentation_utils import (
    collect_dataset_label_stats,
    init_augmentation_pool,
    remove_generated_variants,
    summarize_ratio,
)

def create_weather_augmented_dataset(enhancement_ratio=0.3):
    # 仅从 baseline 原图生成天气增强
    src_dir = DATA_BASELINE_DIR
    dst_dir = DATA_AUGMENTATION_DIR
    
    if not src_dir.exists():
        print(f"错误：找不到基线数据集 {src_dir}")
        return

    init_augmentation_pool(src_dir, dst_dir)

    print(f"[*] 正在处理训练集池，应用天气增强")

    src_train_img = src_dir / "images" / "train"
    src_train_lbl = src_dir / "labels" / "train"
    train_img_dst = dst_dir / "images" / "train"
    train_lbl_dst = dst_dir / "labels" / "train"

    removed_count = remove_generated_variants(train_img_dst, train_lbl_dst, "_aug_weather")
    if removed_count:
        print(f"[*] 已清理历史天气增强样本 {removed_count} 张，避免旧结果叠加")
    
    # 模拟海面大雾、海面降雨、夜间/阴天光照不良、和图像噪点
    transform = A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.08, fog_coef_upper=0.18, alpha_coef=0.06, p=1.0),
            A.RandomRain(brightness_coefficient=0.95, drop_width=1, blur_value=3, rain_type="drizzle", p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.18, -0.05), contrast_limit=(-0.1, 0.1), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.05), mean_range=(0.0, 0.0), p=1.0)
        ], p=1.0)
    ])

    src_train_imgs = sorted(src_train_img.glob("*.jpg"))
    aug_count = 0
    
    for img_path in tqdm(src_train_imgs, desc="增强 Train 集"):
        label_path = src_train_lbl / (img_path.stem + ".txt")
            
        # 按照概率决定是否对这张 baseline 原图进行天气增强
        if random.random() < enhancement_ratio:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            augmented = transform(image=image)
            aug_image = augmented["image"]
            
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            aug_img_name = f"{img_path.stem}_aug_weather{img_path.suffix}"
            cv2.imwrite(str(train_img_dst / aug_img_name), aug_image_bgr)
            
            if label_path.exists():
                aug_lbl_name = f"{img_path.stem}_aug_weather.txt"
                with open(label_path, "r", encoding="utf-8") as src_file:
                    label_content = src_file.read()
                with open(train_lbl_dst / aug_lbl_name, "w", encoding="utf-8") as dst_file:
                    dst_file.write(label_content)
            
            aug_count += 1

    total_files, normal_count, small_count = collect_dataset_label_stats(train_lbl_dst)
            
    print("\n" + "="*50)
    print("天气数据增强完成")
    print(f"  - 原始训练集图片数量：{len(src_train_imgs)}")
    print(f"  - 生成恶劣天气样本数：{aug_count} 张 (扩增率为 {enhancement_ratio:.1%})")
    print(f"  - 当前增强池标签文件数：{total_files}")
    print(f"  - 当前 small target 占比：{summarize_ratio(normal_count, small_count)}")
    print("="*50)

if __name__ == "__main__":
    create_weather_augmented_dataset(enhancement_ratio=0.2)
