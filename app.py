# 船舶小目标检测推理界面

import json
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 全局配置

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs" / "train"
DEFAULT_WEIGHTS = PROJECT_ROOT / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Normal Ship", 1: "Small Target Ship"}
CLASS_NAMES_CN = {0: "常规船舶", 1: "小目标船舶"}

CLASS_COLORS = {
    0: (0, 200, 83),    # 绿色 - 常规船舶
    1: (255, 61, 61),   # 红色 - 小目标船舶
}

_model_cache = {}
_font_cache = {}

# 船舶类别关键词（兼容不同模型命名）
SHIP_CLASS_KEYWORDS = ("ship", "boat", "vessel", "craft", "船", "艇")


# 模型加载异常
class ModelLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded by Ultralytics YOLO."""

#获取中文字体
def _get_chinese_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font

#加载模型
def get_model(weights_path: str) -> YOLO:
    weights_path = str(weights_path)
    if weights_path not in _model_cache:
        try:
            _model_cache[weights_path] = YOLO(weights_path)
        except ModuleNotFoundError as e:
            if "models.yolo" in str(e):
                msg = (
                    f"权重不兼容 "
                    "当前 Ultralytics 环境无法直接加载\n\n"
                )
                raise ModelLoadError(msg) from e
            raise ModelLoadError(f"模型加载失败：{e}") from e
        except Exception as e:
            raise ModelLoadError(f"模型加载失败：{e}") from e
    return _model_cache[weights_path]

# 扫描可用权重文件
def scan_available_weights() -> list[str]:
    weight_files = []
    for pt in PROJECT_ROOT.glob("*.pt"):
        weight_files.append(str(pt))
    for pt in RUNS_DIR.rglob("weights/*.pt"):
        weight_files.append(str(pt))
    return sorted(weight_files) if weight_files else [str(DEFAULT_WEIGHTS)]


# 检测逻辑 

# 绘制检测框和标签
def draw_detections(image: np.ndarray, results, class_filter: list[int],
                    show_labels: bool = True, show_conf: bool = True,
                    line_width: int = 1) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    scale = max(h, w) / 640
    lw = max(int(line_width * scale), 1)
    font_size = max(int(16 * scale), 16)
    pad = max(int(4 * scale), 4)

    label_info = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in class_filter:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # 用 cv2 画检测框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

            if show_labels or show_conf:
                parts = []
                if show_labels:
                    parts.append(CLASS_NAMES_CN.get(cls_id, str(cls_id)))
                if show_conf:
                    parts.append(f"{conf:.2f}")
                label_info.append((x1, y1, " ".join(parts), color))

    # 绘制中文标签
    if label_info:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = _get_chinese_font(font_size)

        for lx, ly, text, color in label_info:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw, th = right - left, bottom - top

            bg_y = ly - th - pad * 2
            if bg_y < 0:
                bg_y = ly

            draw.rectangle(
                [lx, bg_y, lx + tw + pad * 2, bg_y + th + pad * 2],
                fill=color,
            )
            draw.text(
                (lx + pad - left, bg_y + pad - top),
                text, fill=(255, 255, 255), font=font,
            )

        img = np.array(pil_img)

    return img

# 提取检测统计信息
def extract_statistics(results, class_filter: list[int]) -> dict:
    stats = {
        "total": 0,
        "per_class": {cid: {"count": 0, "avg_conf": 0, "confs": []}
                      for cid in CLASS_NAMES},
        "all_confs": [],
        "boxes": [],
    }
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in class_filter:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)

            stats["total"] += 1
            stats["per_class"][cls_id]["count"] += 1
            stats["per_class"][cls_id]["confs"].append(conf)
            stats["all_confs"].append(conf)
            stats["boxes"].append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "class_name_cn": CLASS_NAMES_CN[cls_id],
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "area": area,
            })

    for cid in CLASS_NAMES:
        confs = stats["per_class"][cid]["confs"]
        if confs:
            stats["per_class"][cid]["avg_conf"] = sum(confs) / len(confs)

    return stats

# 转换统计信息至md格式
def format_statistics_markdown(stats: dict, inference_time: float) -> str:
    md = "## 📊 检测统计\n\n"
    md += f"**检测总数：** {stats['total']} 个目标　｜　"
    md += f"**推理耗时：** {inference_time:.1f} ms\n\n"

    if stats["total"] == 0:
        md += "> 未检测到任何目标，请尝试降低置信度阈值。\n"
        return md

    md += "### 各类别统计\n\n"
    md += "| 类别 | 数量 | 占比 | 平均置信度 |\n"
    md += "|------|------|------|------------|\n"
    for cid, name_cn in CLASS_NAMES_CN.items():
        info = stats["per_class"][cid]
        count = info["count"]
        ratio = (count / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_conf = info["avg_conf"]
        md += f"| {name_cn} | {count} | {ratio:.1f}% | {avg_conf:.3f} |\n"

    if stats["all_confs"]:
        confs = stats["all_confs"]
        md += "\n### 置信度分布\n\n"
        md += f"- **最高：** {max(confs):.4f}　｜　"
        md += f"**最低：** {min(confs):.4f}　｜　"
        md += f"**平均：** {sum(confs) / len(confs):.4f}\n"

        ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        md += "\n| 区间 | 数量 | 分布 |\n"
        md += "|------|------|------|\n"
        for lo, hi in ranges:
            cnt = sum(1 for c in confs if lo <= c < hi)
            bar = "▓" * cnt + "░" * max(0, 8 - cnt)
            hi_label = "1.0" if hi > 1 else f"{hi:.1f}"
            md += f"| [{lo:.1f}, {hi_label}) | {cnt} | {bar} |\n"

    if stats["boxes"]:
        sorted_by_area = sorted(stats["boxes"], key=lambda b: b["area"])
        s, l = sorted_by_area[0], sorted_by_area[-1]
        md += "\n### 目标尺寸\n\n"
        md += f"- **最小：** {s['class_name_cn']} "
        md += f"({s['bbox'][2]-s['bbox'][0]}×{s['bbox'][3]-s['bbox'][1]}px, "
        md += f"面积={s['area']}px²)\n"
        md += f"- **最大：** {l['class_name_cn']} "
        md += f"({l['bbox'][2]-l['bbox'][0]}×{l['bbox'][3]-l['bbox'][1]}px, "
        md += f"面积={l['area']}px²)\n"

    return md


# 检测函数
def _parse_class_filter(class_filter: list[str]) -> list[int]:
    filter_ids = []
    for cf in class_filter:
        for cid, cname in CLASS_NAMES_CN.items():
            if cname in cf:
                filter_ids.append(cid)
    return filter_ids if filter_ids else list(CLASS_NAMES.keys())

# 获取文件名
def _get_stem(path) -> str:
    if path is None:
        return f"image_{int(time.time())}"
    stem = Path(str(path)).stem
    return stem if stem else f"image_{int(time.time())}"


# 单图检测公共流程
def _run_detection_pipeline(image: np.ndarray, weights_path, conf_thresh, iou_thresh,
                            class_filter, show_labels, show_conf, line_width, img_size):
    filter_ids = _parse_class_filter(class_filter)
    model = get_model(weights_path)

    t0 = time.time()
    results = model.predict(
        source=image, conf=conf_thresh, iou=iou_thresh,
        imgsz=int(img_size), verbose=False,
    )
    elapsed = (time.time() - t0) * 1000

    det_image = draw_detections(
        image, results, filter_ids,
        show_labels, show_conf, int(line_width),
    )
    stats = extract_statistics(results, filter_ids)
    return det_image, stats, elapsed


# 判断是否为船舶相关类别
def _is_ship_class(class_name: str) -> bool:
    name = class_name.lower()
    return any(k in name for k in SHIP_CLASS_KEYWORDS)


# 从预测结果中提取船舶目标
def _collect_ship_detections(results, model_names: dict | list) -> list[dict]:
    ships = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = str(model_names.get(cls_id, str(cls_id))) if isinstance(model_names, dict) else str(model_names[cls_id])
            if not _is_ship_class(cls_name):
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            ships.append({
                "class_name": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
    return ships


# 模型对比图绘制
def _draw_ship_detections(image: np.ndarray, ships: list[dict]) -> np.ndarray:
    img = image.copy()
    lw = 1
    color = (255, 0, 255)

    for item in ships:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

    return img

# 单张图片检测
def run_single_detection(image_path, weights_path, conf_thresh, iou_thresh,
                         class_filter, show_labels, show_conf, line_width, img_size):
    if image_path is None:
        return None, gr.update(interactive=False), gr.update(interactive=False), "⚠️ 请先上传图片"

    orig_stem = _get_stem(image_path)
    image = np.array(Image.open(image_path).convert("RGB"))
    try:
        det_image, stats, elapsed = _run_detection_pipeline(
            image, weights_path, conf_thresh, iou_thresh,
            class_filter, show_labels, show_conf, line_width, img_size,
        )
    except ModelLoadError as e:
        return (
            None,
            gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False),
            f"⚠️ {e}",
        )
    stats_md = format_statistics_markdown(stats, elapsed)

    # 保存检测结果
    img_save = OUTPUT_DIR / f"{orig_stem}_detect.jpg"
    Image.fromarray(det_image).save(str(img_save), quality=95)

    json_save = _export_json(stats, orig_stem)

    return (
        det_image,
        gr.update(value=str(img_save), interactive=True),
        gr.update(value=str(json_save), interactive=True) if json_save else gr.update(interactive=False),
        stats_md,
    )


# 模型对比检测
def run_compare_detection(image_path, weights_path_a, weights_path_b, conf_thresh, iou_thresh,
                          img_size):
    if image_path is None:
        return None, None, "⚠️ 请先上传图片"

    image = np.array(Image.open(image_path).convert("RGB"))
    det_a, stats_a, elapsed_a, err_a = None, None, None, None
    det_b, stats_b, elapsed_b, err_b = None, None, None, None

    try:
        model_a = get_model(weights_path_a)
        t0 = time.time()
        results_a = model_a.predict(
            source=image, conf=conf_thresh, iou=iou_thresh,
            imgsz=int(img_size), verbose=False,
        )
        elapsed_a = (time.time() - t0) * 1000
        ships_a = _collect_ship_detections(results_a, model_a.names)
        det_a = _draw_ship_detections(image, ships_a)
        stats_a = {
            "ship_count": len(ships_a),
        }
    except ModelLoadError as e:
        err_a = str(e)

    try:
        model_b = get_model(weights_path_b)
        t0 = time.time()
        results_b = model_b.predict(
            source=image, conf=conf_thresh, iou=iou_thresh,
            imgsz=int(img_size), verbose=False,
        )
        elapsed_b = (time.time() - t0) * 1000
        ships_b = _collect_ship_detections(results_b, model_b.names)
        det_b = _draw_ship_detections(image, ships_b)
        stats_b = {
            "ship_count": len(ships_b),
        }
    except ModelLoadError as e:
        err_b = str(e)

    if err_a and err_b:
        return None, None, f"⚠️ 模型A与模型B都加载失败。\n\n- A: {err_a}\n\n- B: {err_b}"

    name_a = Path(str(weights_path_a)).stem
    name_b = Path(str(weights_path_b)).stem

    compare_md = "## 🔬 模型对比结果\n\n"
    compare_md += f"**模型A：** {name_a}　｜　**模型B：** {name_b}\n\n"

    if err_a:
        compare_md += f"> ⚠️ 模型A加载失败：{err_a}\n\n"
    if err_b:
        compare_md += f"> ⚠️ 模型B加载失败：{err_b}\n\n"

    if err_a or err_b:
        compare_md += "### 当前状态\n\n"
        compare_md += "- 已展示可成功加载模型的检测结果。\n"
        compare_md += "- 若需严格对比，请将失败模型替换为 Ultralytics 兼容权重。\n"
        return det_a, det_b, compare_md

    compare_md += "| 指标 | 模型A | 模型B |\n"
    compare_md += "|------|------|------|\n"
    compare_md += f"| 船舶检测数量 | {stats_a['ship_count']} | {stats_b['ship_count']} |\n"
    compare_md += f"| 推理耗时(ms) | {elapsed_a:.1f} | {elapsed_b:.1f} |\n"
    return det_a, det_b, compare_md

# 批量图片检测
def run_batch_detection(files, weights_path, conf_thresh, iou_thresh,
                        class_filter, show_labels, show_conf, line_width, img_size):
    if not files:
        return [], gr.update(interactive=False), "⚠️ 请先上传图片"

    filter_ids = _parse_class_filter(class_filter)
    try:
        model = get_model(weights_path)
    except ModelLoadError as e:
        return [], gr.update(value=None, interactive=False), f"⚠️ {e}"

    gallery, saved = [], []
    totals = {
        "n": 0, "imgs": 0, "time": 0.0,
        "cls": {c: {"n": 0, "confs": []} for c in CLASS_NAMES},
        "per_img": [],
    }

    for f in files:
        fp = str(f)
        orig_name = Path(fp).name
        stem = _get_stem(fp)
        image = np.array(Image.open(fp).convert("RGB"))

        t0 = time.time()
        results = model.predict(
            source=image, conf=conf_thresh, iou=iou_thresh,
            imgsz=int(img_size), verbose=False,
        )
        ms = (time.time() - t0) * 1000
        totals["time"] += ms
        totals["imgs"] += 1

        det = draw_detections(image, results, filter_ids,
                              show_labels, show_conf, int(line_width))
        st = extract_statistics(results, filter_ids)

        totals["n"] += st["total"]
        for c in CLASS_NAMES:
            totals["cls"][c]["n"] += st["per_class"][c]["count"]
            totals["cls"][c]["confs"].extend(st["per_class"][c]["confs"])

        totals["per_img"].append({
            "orig": orig_name,
            "saved": f"{stem}_detect.jpg",
            "count": st["total"],
            "ms": round(ms, 1),
        })

        sp = OUTPUT_DIR / f"{stem}_detect.jpg"
        Image.fromarray(det).save(str(sp), quality=95)
        saved.append(str(sp))
        gallery.append(det)

    # 汇总统计
    md = "## 📊 批量检测统计\n\n"
    md += f"**图片数：** {totals['imgs']}　｜　"
    md += f"**目标总数：** {totals['n']}　｜　"
    md += f"**总耗时：** {totals['time']:.0f}ms"
    md += f"（平均 {totals['time'] / max(totals['imgs'], 1):.0f}ms/张）\n\n"

    md += "### 各类别汇总\n\n"
    md += "| 类别 | 数量 | 平均置信度 |\n"
    md += "|------|------|------------|\n"
    for c, cn in CLASS_NAMES_CN.items():
        info = totals["cls"][c]
        avg = (sum(info["confs"]) / len(info["confs"])) if info["confs"] else 0
        md += f"| {cn} | {info['n']} | {avg:.3f} |\n"

    md += "\n### 各图片检测情况\n\n"
    md += "| 原文件名 | 保存文件名 | 检测数 | 耗时 |\n"
    md += "|----------|-----------|--------|------|\n"
    for item in totals["per_img"]:
        md += f"| {item['orig']} | {item['saved']} | {item['count']} | {item['ms']:.0f}ms |\n"

    zip_path = _create_zip(saved)
    return (
        gallery,
        gr.update(value=str(zip_path), interactive=True) if zip_path else gr.update(interactive=False),
        md,
    )

# 导出检测结果为 JSON
def _export_json(stats: dict, stem: str) -> str | None:
    if not stats or not stats.get("boxes"):
        return None
    path = OUTPUT_DIR / f"{stem}_detect.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"total": stats["total"], "detections": stats["boxes"]},
                  f, ensure_ascii=False, indent=2)
    return str(path)

# 批量结果打包为 ZIP
def _create_zip(paths: list[str]) -> str | None:
    if not paths:
        return None
    zp = OUTPUT_DIR / f"batch_{int(time.time())}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, Path(p).name)
    return str(zp)

# 保存检测结果到用户指定目录
def save_to_custom_path(save_dir: str):
    if not save_dir or not save_dir.strip():
        return "⚠️ 请先填写保存路径"
    target = Path(save_dir.strip())
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"❌ 无法创建目录：{e}"

    files = list(OUTPUT_DIR.glob("*"))
    if not files:
        return "⚠️ 暂无检测结果，请先进行检测"

    copied = 0
    for f in files:
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".json", ".zip"):
            dest = target / f.name
            if not dest.exists():
                shutil.copy2(str(f), str(dest))
                copied += 1
    if copied == 0:
        return "ℹ️ 所有文件已存在于目标目录"
    return f"✅ 已保存 {copied} 个文件到 {target}"



# 界面构建


def build_interface():
    weights = scan_available_weights()
    cls_opts = [f"{c}: {CLASS_NAMES_CN[c]}" for c in CLASS_NAMES_CN]

    css = """
    .gradio-container {
        max-width: 1600px !important;
    }

    /* 标题栏 */
    .app-header {
        background: linear-gradient(135deg, #0c2d48 0%, #145da0 50%, #2e8bc0 100%);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 6px 24px rgba(12, 45, 72, 0.25);
    }
    .app-header h1 {
        color: #fff;
        font-size: 2em;
        margin: 0 0 6px;
        font-weight: 700;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05em;
        margin: 0;
    }

    /*  统计卡片  */
    .stat-card {
        background: linear-gradient(135deg, #e8f0fe 0%, #dbeafe 100%) !important;
        border: 1px solid #93c5fd !important;
        border-radius: 12px;
        padding: 18px !important;
        color: #1e293b !important;
    }
    .stat-card * {
        color: #1e293b !important;
    }
    .stat-card h2, .stat-card h3 {
        color: #0c4a6e !important;
    }
    .stat-card table {
        border-collapse: collapse;
    }
    .stat-card th {
        background: #bfdbfe !important;
        color: #1e3a5f !important;
        padding: 6px 12px !important;
    }
    .stat-card td {
        padding: 6px 12px !important;
        border-bottom: 1px solid #93c5fd !important;
    }
    .stat-card strong {
        color: #0369a1 !important;
    }

    /*  按钮样式  */
    .gradio-container button {
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    .gradio-container button.primary {
        background: linear-gradient(135deg, #145da0, #2e8bc0) !important;
        box-shadow: 0 4px 12px rgba(20, 93, 160, 0.3) !important;
    }
    .gradio-container button.primary:hover {
        box-shadow: 0 6px 20px rgba(20, 93, 160, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    .gradio-container button.secondary {
        border: 2px solid #cbd5e1 !important;
    }
    .gradio-container button.secondary:hover {
        border-color: #94a3b8 !important;
        background: #f8fafc !important;
    }
    /* 下载按钮 */
    .download-btn-row button, .download-btn-row a {
        min-height: 48px !important;
        font-size: 15px !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(15, 118, 110, 0.25) !important;
    }
    .download-btn-row button:hover, .download-btn-row a:hover {
        box-shadow: 0 5px 16px rgba(15, 118, 110, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Tabs 标签 */
    .gradio-container .tab-nav button {
        font-size: 16px !important;
        padding: 10px 20px !important;
    }

    /* 侧边栏内部分组 */
    .drawer-section {
        margin-bottom: 8px;
        padding: 12px;
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    .drawer-section-title {
        font-weight: 700;
        font-size: 1.25em;
        color: #2563eb;
        margin: 8px 0 10px;
    }
    /* 侧边栏内 label 字体 */
    .gradio-container .sidebar label, .gradio-container .sidebar .label-wrap span {
        font-weight: 400 !important;
    }

    /* 隐藏上传图片的来源选择栏 */
    .aligned-image .image-container .upload-container .source-selection {
        display: none !important;
    }

    /* 隐藏图片组件上悬浮的 icon 标签 */
    .aligned-image .icon-buttons,
    .aligned-image .image-container .icon-buttons {
        display: none !important;
    }
    /* 让图片标题显示在图片外部上方而非覆盖在图片上 */
    .aligned-image .image-container {
        overflow: visible !important;
    }
    .aligned-image .icon-button-wrapper {
        display: none !important;
    }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title="船舶目标检测系统") as demo:

        gr.HTML("""
        <div class="app-header">
            <h1>🚢 船舶目标检测系统</h1>
            <p>基于YOLO12的目标检测模型 · 常规船舶与小目标船舶的二分类检测</p>
        </div>
        """)

        #  右侧边栏（Gradio 原生组件） 
        with gr.Sidebar(position="right", open=False, label="⚙️ 设置") as sidebar_panel:
            gr.HTML('<div class="drawer-section-title">📐 检测参数</div>')
            with gr.Group(elem_classes=["drawer-section"]):
                weights_dd = gr.Dropdown(
                    choices=weights,
                    value=weights[0] if weights else "",
                    label="模型权重",
                    allow_custom_value=True,
                )
                refresh_btn = gr.Button("🔄 刷新权重", size="sm")
                conf_sl = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.05,
                    label="置信度阈值",
                )
                iou_sl = gr.Slider(
                    0.1, 0.95, value=0.45, step=0.05,
                    label="IOU 阈值",
                )
                imgsz_sl = gr.Slider(
                    320, 1280, value=640, step=32,
                    label="推理尺寸",
                )
                cls_cb = gr.CheckboxGroup(
                    choices=cls_opts, value=cls_opts,
                    label="检测类别",
                )

            gr.HTML('<div class="drawer-section-title">🎨 显示选项</div>')
            with gr.Group(elem_classes=["drawer-section"]):
                show_lbl = gr.Checkbox(value=True, label="显示类别标签")
                show_cf = gr.Checkbox(value=True, label="显示置信度")
                lw_sl = gr.Slider(1, 5, value=1, step=1, label="检测框线宽")

            gr.HTML('<div class="drawer-section-title">💾 保存设置</div>')
            with gr.Group(elem_classes=["drawer-section"]):
                save_dir_box = gr.Textbox(
                    label="自定义保存路径",
                    placeholder=r"例如 D:\检测结果",
                )
                save_btn = gr.Button("📂 保存到指定目录", size="sm")
                save_st = gr.Textbox(label="状态", interactive=False, lines=1)

        #  主内容区
        with gr.Tabs():
            #  单张检测 
            with gr.TabItem("📷 单张检测") as single_tab:
                with gr.Row(equal_height=True):
                    input_image = gr.Image(
                        label="上传图片",
                        type="filepath",
                        height=480,
                        sources=["upload"],
                        elem_classes=["aligned-image"],
                    )
                    output_image = gr.Image(
                        label="检测结果",
                        type="numpy",
                        height=480,
                        interactive=False,
                        elem_classes=["aligned-image"],
                    )
                with gr.Row():
                    detect_btn = gr.Button(
                        "🔍 开始检测",
                        variant="primary", size="lg", scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ 清空",
                        variant="secondary", size="lg", scale=1,
                    )
                with gr.Row(elem_classes=["download-btn-row"]):
                    dl_img_btn = gr.DownloadButton("📥 下载检测图片", size="sm", scale=1, interactive=False)
                    dl_json_btn = gr.DownloadButton("📥 下载检测数据", size="sm", scale=1, interactive=False)

                stats_output = gr.Markdown(
                    value="*上传图片后点击「开始检测」*",
                    elem_classes=["stat-card"],
                )

            # 模型对比
            with gr.TabItem("⚖️ 模型对比") as compare_tab:
                compare_image = gr.Image(
                    label="上传对比图片",
                    type="filepath",
                    height=420,
                    sources=["upload"],
                    elem_classes=["aligned-image"],
                )
                with gr.Row():
                    compare_weights_a = gr.Dropdown(
                        choices=weights,
                        value=weights[0] if weights else "",
                        label="模型A权重",
                        allow_custom_value=True,
                    )
                    compare_weights_b = gr.Dropdown(
                        choices=weights,
                        value=weights[1] if len(weights) > 1 else (weights[0] if weights else ""),
                        label="模型B权重",
                        allow_custom_value=True,
                    )
                compare_btn = gr.Button("🔍 开始对比检测", variant="primary", size="lg")
                with gr.Row(equal_height=True):
                    compare_output_a = gr.Image(
                        label="模型A检测结果",
                        type="numpy",
                        height=420,
                        interactive=False,
                        elem_classes=["aligned-image"],
                    )
                    compare_output_b = gr.Image(
                        label="模型B检测结果",
                        type="numpy",
                        height=420,
                        interactive=False,
                        elem_classes=["aligned-image"],
                    )
                compare_clear_btn = gr.Button("🗑️ 清除对比结果", variant="secondary", size="sm")
                compare_stats = gr.Markdown(
                    value="*上传图片并选择两个权重后点击「开始对比检测」*",
                    elem_classes=["stat-card"],
                )

            # 批量检测 
            with gr.TabItem("📁 批量检测") as batch_tab:
                batch_file_input = gr.File(
                    label="选择图片文件（支持多选拖拽）",
                    file_count="multiple",
                    file_types=["image"],
                )
                with gr.Row():
                    batch_detect_btn = gr.Button(
                        "🔍 批量检测",
                        variant="primary", size="lg", scale=2,
                    )
                    batch_clear_btn = gr.Button(
                        "🗑️ 清空",
                        variant="secondary", size="lg", scale=1,
                    )
                batch_gallery = gr.Gallery(
                    label="检测结果",
                    columns=3, height=420, object_fit="contain",
                )
                dl_batch_btn = gr.DownloadButton("📥 下载全部结果 (ZIP)", size="sm", interactive=False)
                batch_stats = gr.Markdown(
                    value="*上传图片后点击「批量检测」*",
                    elem_classes=["stat-card"],
                )

            # 使用说明 
            with gr.TabItem("📖 使用说明") as guide_tab:
                gr.Markdown("""
## 使用指南

### 快速上手
1. 点击右上角「⚙️ 设置」按钮打开设置侧边栏
2. 在「📐 检测参数」中选择模型权重
3. 在「单张检测」或「批量检测」页面上传图片
4. 点击「🔍 开始检测」
5. 查看结果、下载标注图片或 JSON 数据
6. 如需保存到自定义路径，在设置面板「💾 保存设置」中指定

### 参数说明
| 参数 | 说明 | 建议值 |
|------|------|--------|
| 置信度阈值 | 过滤低于此值的框，越低检出越多 | 0.25 ~ 0.5 |
| IOU 阈值 | NMS 去除重叠框的阈值 | 0.45 ~ 0.7 |
| 推理尺寸 | 输入模型的图像分辨率 | 640 / 960 / 1280 |

### 类别说明
| 类别 | 颜色 | 说明 |
|------|------|------|
| 🟢 常规船舶 | 绿色 | 面积占比 ≥ 0.5% |
| 🔴 小目标船舶 | 红色 | 面积占比 < 0.5% |

> 💡 小目标检测效果不佳？尝试增大「推理尺寸」至 960 或 1280
>
> 💡 误检过多？适当提高「置信度阈值」
                """)

        #  事件绑定 

        def refresh():
            w = scan_available_weights()
            return gr.update(choices=w, value=w[0] if w else "")

        refresh_btn.click(fn=refresh, outputs=weights_dd)

        save_btn.click(
            fn=save_to_custom_path,
            inputs=[save_dir_box],
            outputs=[save_st],
        )

        detect_btn.click(
            fn=run_single_detection,
            inputs=[input_image, weights_dd, conf_sl, iou_sl,
                    cls_cb, show_lbl, show_cf, lw_sl, imgsz_sl],
            outputs=[output_image, dl_img_btn, dl_json_btn, stats_output],
        )

        compare_btn.click(
            fn=run_compare_detection,
            inputs=[compare_image, compare_weights_a, compare_weights_b,
                    conf_sl, iou_sl, imgsz_sl],
            outputs=[compare_output_a, compare_output_b, compare_stats],
        )

        # 切换到对比页时隐藏侧边栏
        compare_tab.select(
            fn=lambda: gr.update(open=False, visible=False),
            outputs=[sidebar_panel],
        )

        # 切回其他页时恢复侧边栏
        single_tab.select(
            fn=lambda: gr.update(visible=True),
            outputs=[sidebar_panel],
        )

        batch_tab.select(
            fn=lambda: gr.update(visible=True),
            outputs=[sidebar_panel],
        )

        guide_tab.select(
            fn=lambda: gr.update(visible=True),
            outputs=[sidebar_panel],
        )

        compare_clear_btn.click(
            fn=lambda: (None, None, None, "*上传图片并选择两个权重后点击「开始对比检测」*"),
            outputs=[compare_image, compare_output_a, compare_output_b, compare_stats],
        )

        clear_btn.click(
            fn=lambda: (None, None, gr.update(value=None, interactive=False),
                        gr.update(value=None, interactive=False),
                        "*上传图片后点击「开始检测」*"),
            outputs=[input_image, output_image, dl_img_btn, dl_json_btn, stats_output],
        )

        batch_detect_btn.click(
            fn=run_batch_detection,
            inputs=[batch_file_input, weights_dd, conf_sl, iou_sl,
                    cls_cb, show_lbl, show_cf, lw_sl, imgsz_sl],
            outputs=[batch_gallery, dl_batch_btn, batch_stats],
        )

        batch_clear_btn.click(
            fn=lambda: (None, [], gr.update(value=None, interactive=False),
                        "*上传图片后点击「批量检测」*"),
            outputs=[batch_file_input, batch_gallery, dl_batch_btn, batch_stats],
        )

    return demo, css, theme


#  启动入口 

if __name__ == "__main__":
    demo, css, theme = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=css,
    )
