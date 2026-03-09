import sys
from pathlib import Path
from ultralytics import YOLO

# 动态加载根目录的统一配置
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASELINE_YAML_PATH, PRETRAINED_MODEL, RUNS_DIR, IMG_SIZE

def main():
    # 路径配置 
    DATASET_YAML = str(BASELINE_YAML_PATH)

    # 初始化模型
    print(f"[*] 正在加载模型：{PRETRAINED_MODEL}")
    model = YOLO(PRETRAINED_MODEL)

    # 配置训练参数并启动训练
    PROJECT_DIR = 'runs/train'
    #实验名称
    RUN_NAME = 'exp0_baseline'

    print(f"[*] 开始基线训练，数据集由 {DATASET_YAML} 提供支持。")
    print(f"[*] 训练运行日志将保存在：{PROJECT_DIR}/{RUN_NAME}")

    # 参数设置
    results = model.train(
        # 数据集与架构设定
        data=DATASET_YAML,       
        epochs=150,               
        imgsz=IMG_SIZE,           # 统一使用 config 中的图片大小
        batch=10,                 
        workers=8,                
        device=0,                 

        # 优化器与策略 
        optimizer='auto',         
        patience=50,              
        val=True,                 

        # 输出与日志记录 
        project=str(RUNS_DIR / "train"),  # 使用统一的 runs 输出路径
        name=RUN_NAME,            
        save=True,                # 保存best.pt
        save_period=-1,           # 是否定期保存权重
        plots=True,             
        
        # 默认数据增强
        mosaic=1.0,               # 开启 mosaic
        mixup=0.0,                
        copy_paste=0.0,           
    )

    print(f"[*] 基准模型训练完成")
    print(f"[*] 模型权重 '{PROJECT_DIR}/{RUN_NAME}/weights/best.pt'")
    print(f"[*] 各种曲线图和混淆矩阵在 '{PROJECT_DIR}/{RUN_NAME}/'")

if __name__ == '__main__':
    main()
