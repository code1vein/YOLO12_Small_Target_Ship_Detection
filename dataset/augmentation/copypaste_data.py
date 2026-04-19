import sys
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    COPYPASTE_TARGET_SMALL_RATIO,
    DATA_AUGMENTATION_DIR,
    DATA_BASELINE_DIR,
    SAFE_SMALL_TARGET_THRESHOLD,
)
from dataset.augmentation.augmentation_utils import (
    collect_dataset_label_stats,
    count_label_classes,
    init_augmentation_pool,
    load_label_lines,
    remove_generated_variants,
    should_skip_generated_file,
    summarize_ratio,
)

def iou(box1, box2):
    # 计算两个绝对坐标[x1, y1, x2, y2]的IoU来防止粘贴重叠
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou_val = intersect_area / float(box1_area + box2_area - intersect_area)
    return iou_val

def extract_small_ships_catalog(train_img_dir, train_lbl_dir):
    # 提取所有类别为小目标的候选框信息
    catalog = []
    print("正在扫描基线数据集，构建小目标候选库")
    for lbl_path in sorted(train_lbl_dir.glob("*.txt"), key=lambda path: path.name):
        if should_skip_generated_file(lbl_path.stem, ("_aug_noise",)):
            continue
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                c, x, y, w, h = map(float, parts)
                if int(c) == 1:
                    img_path = train_img_dir / (lbl_path.stem + ".jpg") # 假设jpg
                    if img_path.exists():
                        catalog.append({"img_path": img_path, "bbox": [x, y, w, h]})
    print(f"共收集到 {len(catalog)} 个独立的小目标样本")
    return catalog


def expand_crop_region(x1, y1, x2, y2, img_w, img_h, context_ratio=0.25):
    # 给裁剪框补一点上下文边界
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = max(2, int(box_w * context_ratio))
    pad_y = max(2, int(box_h * context_ratio))
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(img_w, x2 + pad_x)
    py2 = min(img_h, y2 + pad_y)
    return px1, py1, px2, py2


def create_soft_mask(height, width, box, feather_ratio=0.35):
    # 通过高斯模糊得到软边 mask
    mask = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1.0
    blur_base = int(max(3, min(x2 - x1, y2 - y1) * feather_ratio))
    blur_kernel = blur_base if blur_base % 2 == 1 else blur_base + 1
    return cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)


def paste_with_alpha(base_img, patch, alpha_mask, patch_x1, patch_y1):
    # 用 alpha 混合替代直接覆盖
    patch_h, patch_w = patch.shape[:2]
    roi = base_img[patch_y1:patch_y1 + patch_h, patch_x1:patch_x1 + patch_w].astype(np.float32)
    patch_float = patch.astype(np.float32)
    alpha = alpha_mask[..., None]
    blended = patch_float * alpha + roi * (1.0 - alpha)
    base_img[patch_y1:patch_y1 + patch_h, patch_x1:patch_x1 + patch_w] = blended.astype(np.uint8)


def get_vertical_band(existing_boxes, img_h):
    # 根据原图已有目标的大致垂直分布，限制新目标的落点区域
    if not existing_boxes:
        return int(img_h * 0.45), int(img_h * 0.92)

    centers = [int((box[1] + box[3]) / 2) for box in existing_boxes]
    min_center = max(int(img_h * 0.35), min(centers) - int(img_h * 0.1))
    max_center = min(int(img_h * 0.95), max(centers) + int(img_h * 0.1))
    if min_center >= max_center:
        return int(img_h * 0.45), int(img_h * 0.92)
    return min_center, max_center


def prepare_donor_patch(donor_img, donor_bbox, target_img_shape):
    # 从供体图中裁出带上下文的小目标 patch，并按目标图尺度做安全缩放
    img_h, img_w = donor_img.shape[:2]
    dx, dy, d_nw, d_nh = donor_bbox

    x1 = int((dx - d_nw / 2) * img_w)
    y1 = int((dy - d_nh / 2) * img_h)
    x2 = int((dx + d_nw / 2) * img_w)
    y2 = int((dy + d_nh / 2) * img_h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 - x1 <= 4 or y2 - y1 <= 4:
        return None

    px1, py1, px2, py2 = expand_crop_region(x1, y1, x2, y2, img_w, img_h)
    patch = donor_img[py1:py2, px1:px2]
    box_in_patch = [x1 - px1, y1 - py1, x2 - px1, y2 - py1]
    patch_h, patch_w = patch.shape[:2]
    target_h, target_w = target_img_shape[:2]

    scale = random.uniform(0.85, 1.1)
    scaled_patch_w = max(4, int(round(patch_w * scale)))
    scaled_patch_h = max(4, int(round(patch_h * scale)))
    scaled_box = [
        int(round(box_in_patch[0] * scale)),
        int(round(box_in_patch[1] * scale)),
        int(round(box_in_patch[2] * scale)),
        int(round(box_in_patch[3] * scale)),
    ]

    box_w = max(2, scaled_box[2] - scaled_box[0])
    box_h = max(2, scaled_box[3] - scaled_box[1])
    area_ratio = (box_w / target_w) * (box_h / target_h)
    if area_ratio >= SAFE_SMALL_TARGET_THRESHOLD:
        # 超过安全阈值时继续缩小，避免贴出“伪小目标”大框
        shrink = (SAFE_SMALL_TARGET_THRESHOLD / max(area_ratio, 1e-6)) ** 0.5
        scaled_patch_w = max(4, int(round(scaled_patch_w * shrink * 0.95)))
        scaled_patch_h = max(4, int(round(scaled_patch_h * shrink * 0.95)))
        scaled_box = [int(round(value * shrink * 0.95)) for value in scaled_box]
        box_w = max(2, scaled_box[2] - scaled_box[0])
        box_h = max(2, scaled_box[3] - scaled_box[1])

    if scaled_patch_w >= target_w or scaled_patch_h >= target_h:
        return None

    patch = cv2.resize(patch, (scaled_patch_w, scaled_patch_h), interpolation=cv2.INTER_LINEAR)
    mask = create_soft_mask(scaled_patch_h, scaled_patch_w, scaled_box)
    return patch, mask, scaled_box

def create_copypaste_dataset(
    augment_ratio=0.15,
    max_paste=2,
    target_small_ratio=COPYPASTE_TARGET_SMALL_RATIO,
    max_samples: int | None = None,
):
    src_dir = DATA_BASELINE_DIR
    dst_dir = DATA_AUGMENTATION_DIR
    
    if not src_dir.exists():
        print(f"错误: 找不到基线数据集 {src_dir}")
        return
        
    init_augmentation_pool(src_dir, dst_dir, excluded_suffixes=("_aug_noise",))
    print(f"启动 Copy-Paste 流水线，目标池{dst_dir}")
    
    src_train_img = src_dir / "images" / "train"
    src_train_lbl = src_dir / "labels" / "train"
    dst_train_img = dst_dir / "images" / "train"
    dst_train_lbl = dst_dir / "labels" / "train"

    removed_count = remove_generated_variants(dst_train_img, dst_train_lbl, "_copypaste")
    if removed_count:
        print(f"[*] 已清理历史 Copy-Paste 样本 {removed_count} 张，避免脏样本残留")

    _, current_normal, current_small = collect_dataset_label_stats(dst_train_lbl)
    print(f"[*] 当前增强池 small target 占比：{summarize_ratio(current_normal, current_small)}")
    print(f"[*] 目标上限占比：{target_small_ratio:.2%}")

    small_ships_catalog = extract_small_ships_catalog(src_train_img, src_train_lbl)
    
    aug_count = 0
    train_images = sorted(
        img_path
        for img_path in src_train_img.glob("*.jpg")
        if not should_skip_generated_file(img_path.stem, ("_aug_noise",))
    )
    if max_samples is not None:
        target_aug_count = min(max_samples, len(train_images))
    else:
        target_aug_count = int(round(len(train_images) * augment_ratio))
    target_aug_count = min(target_aug_count, len(train_images))
    selected_images = set(random.sample(train_images, target_aug_count)) if target_aug_count > 0 else set()
    
    for img_path in tqdm(train_images, desc="执行抠图与重混 (Copy-Paste)"):
        lbl_path = src_train_lbl / (img_path.stem + ".txt")
            
        if img_path in selected_images and len(small_ships_catalog) > 0:
            base_img = cv2.imread(str(img_path))
            if base_img is None:
                continue
            
            img_h, img_w = base_img.shape[:2]
            
            existing_boxes = []
            new_labels = load_label_lines(lbl_path)
            base_normal_count, base_small_count = count_label_classes(new_labels)
            if new_labels:
                for line in new_labels:
                    parts = line.split()
                    if len(parts) == 5:
                        _, rx, ry, rw, rh = map(float, parts)
                        x1, y1 = int((rx - rw / 2) * img_w), int((ry - rh / 2) * img_h)
                        x2, y2 = int((rx + rw / 2) * img_w), int((ry + rh / 2) * img_h)
                        existing_boxes.append([x1, y1, x2, y2])

            band_top, band_bottom = get_vertical_band(existing_boxes, img_h)
            paste_num = random.randint(1, max_paste)
            success_paste = 0
            
            for _ in range(paste_num):
                donor = random.choice(small_ships_catalog)
                donor_img = cv2.imread(str(donor["img_path"]))
                if donor_img is None:
                    continue

                prepared = prepare_donor_patch(donor_img, donor["bbox"], base_img.shape)
                if prepared is None:
                    continue

                patch, alpha_mask, box_in_patch = prepared
                patch_h, patch_w = patch.shape[:2]
                box_w = box_in_patch[2] - box_in_patch[0]
                box_h = box_in_patch[3] - box_in_patch[1]

                projected_ratio = (current_small + base_small_count + success_paste + 1) / max(
                    1,
                    current_normal + current_small + base_normal_count + base_small_count + success_paste + 1,
                )
                if projected_ratio > target_small_ratio:
                    # 小目标占比过高，则提前停止当前图片增强
                    break

                placed = False
                for _attempt in range(20):
                    patch_x1 = random.randint(0, max(1, img_w - patch_w))
                    patch_y1 = random.randint(0, max(1, img_h - patch_h))

                    candidate_box = [
                        patch_x1 + box_in_patch[0],
                        patch_y1 + box_in_patch[1],
                        patch_x1 + box_in_patch[2],
                        patch_y1 + box_in_patch[3],
                    ]
                    center_y = (candidate_box[1] + candidate_box[3]) / 2
                    overlap = any(iou(candidate_box, ex_box) > 0.02 for ex_box in existing_boxes)

                    # 同时约束重叠和垂直位置，避免出现明显不合理的粘贴样本
                    if overlap or center_y < band_top or center_y > band_bottom:
                        continue

                    if candidate_box[0] < 0 or candidate_box[1] < 0 or candidate_box[2] > img_w or candidate_box[3] > img_h:
                        continue

                    if not overlap:
                        paste_with_alpha(base_img, patch, alpha_mask, patch_x1, patch_y1)
                        existing_boxes.append(candidate_box)
                        
                        # 将新粘贴的小目标同步写回 YOLO 标签
                        ncx = ((candidate_box[0] + candidate_box[2]) / 2) / img_w
                        ncy = ((candidate_box[1] + candidate_box[3]) / 2) / img_h
                        nw = box_w / img_w
                        nh = box_h / img_h
                        
                        new_labels.append(f"1 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                        placed = True
                        success_paste += 1
                        break
            
            if success_paste > 0:
                new_img_name = f"{img_path.stem}_copypaste{img_path.suffix}"
                new_lbl_name = f"{img_path.stem}_copypaste.txt"
                
                # 只有成功完成至少一次粘贴时，才生成新的增强样本
                cv2.imwrite(str(dst_train_img / new_img_name), base_img)
                with open(dst_train_lbl / new_lbl_name, "w", encoding="utf-8") as file:
                    file.write("\n".join(new_labels) + "\n")
                aug_count += 1
                current_normal += base_normal_count
                current_small += base_small_count + success_paste
                if aug_count >= target_aug_count:
                    break
                
    print("\n" + "="*50)
    print("Copy-Paste数据增强完成")
    print(f"  - 原始训练集图片数量：{len(train_images)}")
    print(f"  - 新生成图片：{aug_count} 张")
    print(f"  - 当前 small target 占比：{summarize_ratio(current_normal, current_small)}")
    print("="*50)

if __name__ == "__main__":
    create_copypaste_dataset(augment_ratio=0.15, max_paste=2)
