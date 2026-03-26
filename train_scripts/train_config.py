import sys
import csv
import json
from pathlib import Path

from ultralytics import YOLO

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMG_SIZE, PROJECT_ROOT, RUNS_DIR

# 从ultralytics的返回值中提取GFLOPs
def _extract_flops_g(info_ret):
    if isinstance(info_ret, (tuple, list)):
        for v in reversed(info_ret):
            if isinstance(v, (int, float)):
                return float(v)
    if isinstance(info_ret, dict):
        for k in ["flops", "GFLOPs", "gflops"]:
            if k in info_ret and isinstance(info_ret[k], (int, float)):
                return float(info_ret[k])
    return None

# 训练结束后保存 Params/FLOPs
def _save_complexity_report(output_dir: Path, run_name: str, dataset_desc: str):
    best_path = output_dir / "weights" / "best.pt"
    if not best_path.exists():
        print(f"[!] 未找到 best.pt，跳过复杂度统计：{best_path}")
        return

    model = YOLO(str(best_path))
    params = sum(p.numel() for p in model.model.parameters())
    params_m = params / 1e6

    flops_g = None
    try:
        info_ret = model.info(imgsz=IMG_SIZE, verbose=False)
        flops_g = _extract_flops_g(info_ret)
    except Exception as e:
        print(f"[!] 计算 FLOPs 失败：{e}")

    report = {
        "run_name": run_name,
        "dataset_desc": dataset_desc,
        "best_weight": str(best_path),
        "imgsz": IMG_SIZE,
        "params": params,
        "params_m": round(params_m, 4),
        "flops_g": round(flops_g, 4) if flops_g is not None else None,
    }

    per_run_report = output_dir / "model_complexity.json"
    with open(per_run_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary_csv = RUNS_DIR / "train" / "experiment_complexity_records.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    need_header = not summary_csv.exists()
    with open(summary_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow([
                "run_name", "dataset_desc", "best_weight", "imgsz",
                "params", "params_m", "flops_g",
            ])
        writer.writerow([
            run_name,
            dataset_desc,
            str(best_path),
            IMG_SIZE,
            params,
            round(params_m, 4),
            round(flops_g, 4) if flops_g is not None else "",
        ])

    print(f"[*] Params/FLOPs 已保存：{per_run_report}")
    print(f"[*] Params/FLOPs 总表追加：{summary_csv}")


def resolve_model_source(model_name):
    model_path = Path(model_name)
    if model_path.is_absolute():
        return str(model_path)

    local_model_path = PROJECT_ROOT / model_path
    if local_model_path.exists():
        return str(local_model_path)

    return str(model_name)


def train_model(dataset_yaml, model_name, run_name, dataset_desc, model_cfg=None, pretrained_weights=None):
    output_dir = RUNS_DIR / 'train' / run_name
    resolved_model_name = resolve_model_source(model_name)

    if model_cfg is not None:
        print(f"[*] 正在构建自定义模型：{model_cfg}")
        model = YOLO(str(model_cfg))
        weights_to_load = resolve_model_source(pretrained_weights or model_name)
        print(f"[*] 正在迁移预训练权重：{weights_to_load}")
        model.load(weights_to_load)
    else:
        print(f"[*] 正在加载模型：{resolved_model_name}")
        model = YOLO(resolved_model_name)

    print(f"[*] 开始{dataset_desc}训练，数据集由 {dataset_yaml} 提供支持。")
    print(f"[*] 训练运行日志将保存在：{output_dir}")

    model.train(
        # 数据集与架构设定
        data=str(dataset_yaml),              
        epochs=150,                          
        imgsz=IMG_SIZE,                      
        batch=16,                            # 每个迭代的样本数
        workers=8,                           # 数据加载线程数
        device=0,                            

        # 优化器与训练策略
        optimizer='auto',                    
        patience=50,                         
        val=True, 

        # 输出与日志记录
        project=str(RUNS_DIR / "train"),    
        name=run_name,                       
        save=True,                           
        save_period=-1,                      
        plots=True,                          

        # 数据增强配置
        mosaic=0.5,                          # 四图拼接
        close_mosaic=10,                     # 最后 10 个 epoch 关闭 mosaic
        scale=0.2,                           # 随机缩放
        translate=0.05,                      # 随机平移
        hsv_h=0.015,                         # 色相扰动幅度
        hsv_s=0.3,                           # 饱和度扰动幅度
        hsv_v=0.2,                           # 亮度扰动幅度
    )

    _save_complexity_report(output_dir, run_name, dataset_desc)

    print(f"[*] {dataset_desc}训练完成")
    print(f"[*] 模型权重 '{output_dir / 'weights' / 'best.pt'}'")
    print(f"[*] 各种曲线图和混淆矩阵在 '{output_dir}'")