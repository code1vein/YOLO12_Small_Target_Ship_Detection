import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(
    dataset_dir: str, 
    images_dir_name: str = "images", 
    labels_dir_name: str = "labels_2class",
    output_dir_name: str = "yolo_dataset",
    ratios: tuple = (0.7, 0.2, 0.1),
    seed: int = 42
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
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []
    
    print("正在扫描并匹配图片和标签文件...")
    for img_file in src_images_dir.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            # 对应的标签文件名
            label_file = src_labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                all_images.append(img_file.name)
                
    total_files = len(all_images)
    print(f"找到 {total_files} 个具有对应标签的图片文件。")
    
    if total_files == 0:
        print("未找到任何成对的数据，退出。")
        return

    # 打乱顺序
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
    from config import DATA_ORIGINAL_DIR, DATA_BASELINE_DIR
    
    split_dataset(
        dataset_dir=str(DATA_ORIGINAL_DIR),
        images_dir_name="images",
        labels_dir_name="labels_2class",  # 使用我们刚刚生成的2分类标签
        output_dir_name=str(DATA_BASELINE_DIR),   # 指向统一配置中的 baseline 文件夹
        ratios=(0.7, 0.2, 0.1)            # 按照 7:2:1 划分
    )
