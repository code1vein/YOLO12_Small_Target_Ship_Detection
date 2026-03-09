import sys
from pathlib import Path

from ultralytics import YOLO

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMG_SIZE, PROJECT_ROOT, RUNS_DIR


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
        batch=10,                            # 每个迭代的样本数
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

    print(f"[*] {dataset_desc}训练完成")
    print(f"[*] 模型权重 '{output_dir / 'weights' / 'best.pt'}'")
    print(f"[*] 各种曲线图和混淆矩阵在 '{output_dir}'")