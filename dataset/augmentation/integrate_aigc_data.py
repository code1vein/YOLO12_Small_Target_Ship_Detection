import sys
import json
import shutil
import random
from pathlib import Path

# 添加项目根目录到运行路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from config import DATA_AUGMENTATION_DIR, DATA_BASELINE_DIR, DATASET_ROOT, SMALL_TARGET_BORDER_MARGIN, SMALL_TARGET_THRESHOLD
from dataset.augmentation.augmentation_utils import init_augmentation_pool, remove_generated_prefixes

# 路径配置
AIGC_DIR = DATASET_ROOT / "aigc_image"
AIGC_IMAGES_DIR = AIGC_DIR / "images"
AIGC_LABELS_DIR = AIGC_DIR / "labels"
AIGC_CORRECTED_LABELS_DIR = AIGC_DIR / "labels_corrected"

CLASS_ID_TO_NAME = {
    0: "Normal Ship",
    1: "Small Target Ship",
}


# 将 AIGC 原始标签文本归一化到项目使用的 2 类 ID
def normalize_label_to_class_id(label_text):
    if not label_text:
        return None

    normalized = label_text.strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    if "small" in normalized and "ship" in normalized:
        return 1
    if "normal" in normalized and "ship" in normalized:
        return 0
    return None

# 核心转换函数
def convert_labelme_to_yolo(json_path, img_width, img_height):
    yolo_lines = []
    skipped_ambiguous = 0
    corrected_label_count = 0
    normalized_label_count = 0
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # 如果 json 里没有宽/高，使用传入的兜底
        w_img = data.get("imageWidth", img_width)
        h_img = data.get("imageHeight", img_height)

        for shape in data.get("shapes", []):
            points = shape["points"]
            # 兼容矩形或多边形
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            # 计算绝对宽和高
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            
            # 极值保护
            bbox_w = max(bbox_w, 1.0)
            bbox_h = max(bbox_h, 1.0)

            # 占比面积判断
            area_ratio = (bbox_w * bbox_h) / (w_img * h_img)
            if abs(area_ratio - SMALL_TARGET_THRESHOLD) <= SMALL_TARGET_BORDER_MARGIN:
                skipped_ambiguous += 1
                continue
            class_id = 1 if area_ratio < SMALL_TARGET_THRESHOLD else 0

            # 优先以面积规则修正类别，避免 AIGC 人工标注类别不稳定
            original_label = shape.get("label", "")
            original_class_id = normalize_label_to_class_id(original_label)
            corrected_label = CLASS_ID_TO_NAME[class_id]
            if original_class_id is not None and original_class_id != class_id:
                corrected_label_count += 1
            if original_label != corrected_label:
                normalized_label_count += 1
                shape["label"] = corrected_label

            # YOLO 归一化 xywh
            dw = 1.0 / w_img
            dh = 1.0 / h_img
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0

            n_x = x_center * dw
            n_y = y_center * dh
            n_w = bbox_w * dw
            n_h = bbox_h * dh

            # 防止越界
            n_x = max(0.000001, min(0.999999, n_x))
            n_y = max(0.000001, min(0.999999, n_y))
            n_w = max(0.000001, min(1.0, n_w))
            n_h = max(0.000001, min(1.0, n_h))

            yolo_lines.append(f"{class_id} {n_x:.6f} {n_y:.6f} {n_w:.6f} {n_h:.6f}")

    return yolo_lines, {
        "skipped_ambiguous": skipped_ambiguous,
        "corrected_label_count": corrected_label_count,
        "normalized_label_count": normalized_label_count,
        "corrected_data": data,
    }

# 主执行流：匹配、重命名并整合到集大成池
def main(max_samples: int | None = None, seed: int = 42):
    # 先同步 baseline 到增强池，再将 AIGC 数据按统一规则增量并入 train 集
    init_augmentation_pool(DATA_BASELINE_DIR, DATA_AUGMENTATION_DIR)

    # 所有 AIGC 样本统一落到增强训练集的 train 子集
    target_images_dir = DATA_AUGMENTATION_DIR / "images" / "train"
    target_labels_dir = DATA_AUGMENTATION_DIR / "labels" / "train"

    # 单独保留一份修正后的 JSON，方便后续人工复核
    AIGC_CORRECTED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    removed_count = remove_generated_prefixes(target_images_dir, target_labels_dir, "aigc_ship_")
    if removed_count:
        print(f"[*] 已清理历史 AIGC 样本 {removed_count} 张，避免重复叠加")

    json_files = sorted(AIGC_LABELS_DIR.glob("*.json"), key=lambda path: path.name)
    if not json_files:
        print(f"[-] 在 {AIGC_LABELS_DIR} 目录下没有找到 json 标注文件")
        return

    random.Random(seed).shuffle(json_files)
    if max_samples is not None:
        json_files = json_files[:max_samples]

    print(f"[*] 发现 {len(json_files)} 个 JSON 标注文件，准备进行转换和合并")

    success_count = 0
    missing_count = 0
    skipped_ambiguous_total = 0
    corrected_label_total = 0
    normalized_label_total = 0

    # 为了连续的编号格式 aigc_ship_000001.jpg
    start_index = 1

    for json_path in json_files:
        stem = json_path.stem
        
        # 宽容匹配对应的图片文件 (.jpg, .png, .jpeg)
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            possible_path = AIGC_IMAGES_DIR / f"{stem}{ext}"
            if possible_path.exists():
                img_path = possible_path
                break
        
        if not img_path:
            print(f"[-] 警告：找不到 {stem} 对应的图片文件，跳过...")
            missing_count += 1
            continue

        # 默认新图为 640x640，标注文件有真实分辨率会自动覆盖
        img_width, img_height = 640, 640

        try:
            # 转换标签格式
            yolo_lines, convert_stats = convert_labelme_to_yolo(json_path, img_width, img_height)
            skipped_ambiguous_total += convert_stats["skipped_ambiguous"]
            corrected_label_total += convert_stats["corrected_label_count"]
            normalized_label_total += convert_stats["normalized_label_count"]

            # 输出修正后的 LabelMe JSON，便于对照检查自动纠正结果
            corrected_json_path = AIGC_CORRECTED_LABELS_DIR / json_path.name
            with open(corrected_json_path, 'w', encoding='utf-8') as corrected_file:
                json.dump(convert_stats["corrected_data"], corrected_file, ensure_ascii=False, indent=2)

            if not yolo_lines:
                print(f"[-] {json_path.name} 只包含阈值边界附近目标，已跳过")
                continue

            # 生成新文件名
            new_stem = f"aigc_ship_{start_index:06d}"
            new_img_name = f"{new_stem}{img_path.suffix}"
            new_txt_name = f"{new_stem}.txt"

            target_img_path = target_images_dir / new_img_name
            target_txt_path = target_labels_dir / new_txt_name

            # 复制图片
            shutil.copy2(img_path, target_img_path)

            # 写入标签
            with open(target_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

            start_index += 1
            success_count += 1

        except Exception as e:
            print(f"[-] 解析文件出错：{json_path.name}，错误信息: {e}")

    print("=" * 40)
    print(f"[+] 整合完毕！一共转换/合并了 {success_count} 张图片与标签。")
    print(f"[-] 缺失图片跳过：{missing_count} 个")
    print(f"[-] 阈值边界样本跳过：{skipped_ambiguous_total} 个框")
    print(f"[+] 面积判定后自动纠正的错分类标签：{corrected_label_total} 个框")
    print(f"[+] 统一标准化的标签文本：{normalized_label_total} 个框")
    print(f"[+] 修正后的 JSON 保存位置：{AIGC_CORRECTED_LABELS_DIR}")
    print(f"[+] 保存位置：")
    print(f"    - 图片库: {target_images_dir}")
    print(f"    - 标签库: {target_labels_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()
