import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ORIGINAL_DIR, ORIGINAL_IMAGES_DIR, ORIGINAL_LABELS_DIR, ORIGINAL_LABELS_2CLASS_DIR

def rename_dataset_in_place():
    src_images_dir = ORIGINAL_IMAGES_DIR
    src_labels_2class_dir = ORIGINAL_LABELS_2CLASS_DIR
    src_labels_orig_dir = ORIGINAL_LABELS_DIR
    
    if not src_images_dir.exists() or not src_labels_2class_dir.exists():
        print("错误: 找不到原始的文件夹。")
        return

    # 记录字典与计数器
    mapping_record = {}
    counters = {
        "Normal_Ship": 0,
        "Small_Target_Ship": 0,
        "Mixed_Ship": 0,
        "Background": 0
    }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    print("开始扫描标签分析图片类别")
    
    rename_tasks = [] # 存储(旧路径, 新路径)元组
    
    for img_path in list(src_images_dir.iterdir()):
        if img_path.suffix.lower() not in valid_extensions:
            continue
            
        label_path = src_labels_2class_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue # 跳过没有标签的图
            
        # 读取二分类标签文件，判断属于哪种类型
        has_normal = False
        has_small = False
        
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        has_normal = True
                    elif cls_id == 1:
                        has_small = True
                        
        # 赋予命名前缀
        if has_normal and has_small:
            prefix = "Mixed_Ship"
        elif has_normal and not has_small:
            prefix = "Normal_Ship"
        elif not has_normal and has_small:
            prefix = "Small_Target_Ship"
        else:
            prefix = "Background"
            
        # 产生新的序号和文件名
        counters[prefix] += 1
        new_stem = f"{prefix}_{counters[prefix]:06d}"
        
        # 记录图片的重命名任务
        new_img_path = src_images_dir / (new_stem + img_path.suffix)
        rename_tasks.append((img_path, new_img_path))
        
        # 记录二分类标签的重命名任务
        new_label_path = src_labels_2class_dir / (new_stem + ".txt")
        rename_tasks.append((label_path, new_label_path))
        
        # 修改原始 labels 
        orig_txt = src_labels_orig_dir / (img_path.stem + ".txt")
        if orig_txt.exists():
            rename_tasks.append((orig_txt, src_labels_orig_dir / (new_stem + ".txt")))
            
        orig_json = src_labels_orig_dir / (img_path.stem + ".json")
        if orig_json.exists():
            rename_tasks.append((orig_json, src_labels_orig_dir / (new_stem + ".json")))
        
        mapping_record[img_path.name] = (new_stem + img_path.suffix)

    print(f"准备执行原地重命名，总计进行 {len(rename_tasks)} 次文件操作")
    
    # 执行重命名
    for old_path, new_path in tqdm(rename_tasks, desc="重命名中"):
        if old_path.exists():
            old_path.rename(new_path)

    # 导出映射表以备溯源
    mapping_file = DATA_ORIGINAL_DIR / "rename_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_record, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*40)
    print("原地重命名完成")
    print("重命名数据统计：")
    print(f"  - 纯常规船 (Normal_Ship)       : {counters['Normal_Ship']} 张")
    print(f"  - 纯小目标 (Small_Target_Ship) : {counters['Small_Target_Ship']} 张")
    print(f"  - 混合结构 (Mixed_Ship)        : {counters['Mixed_Ship']} 张")
    print(f"  - 空背景图 (Background)        : {counters['Background']} 张")
    print(f"\n文件映射对照表已存至: {mapping_file}")
    print("="*40)

if __name__ == "__main__":
    rename_dataset_in_place()
