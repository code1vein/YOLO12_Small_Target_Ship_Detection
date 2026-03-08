import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_ablation_curves(csv_paths_dict, save_path="ablation_mAP50_curve.png"):
    # 将多个实验的 results.csv 绘制在同一张图上，对比 mAP.
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.sans-serif"] = ["SimHei"] # 处理中文
    plt.rcParams["axes.unicode_minus"] = False   # 处理负号
    
    plt.figure(figsize=(10, 6), dpi=300) # 指定高清分辨率 DPI
    
    # 配色方案
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    
    for idx, (exp_name, csv_path) in enumerate(csv_paths_dict.items()):
        if not Path(csv_path).exists():
            print(f"警告: 找不到数据表 {csv_path}，已跳过。")
            continue
            
        # 读取 mAP@0.5
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() 
        
        # 横坐标: 训练轮次 纵坐标: mAP@0.5 的数值
        epochs = df['epoch']

        target_metric = 'metrics/mAP50(B)' 
        
        if target_metric not in df.columns:
            print(f"错误: {csv_path} 中找不到指定的评估指标列。")
            continue
            
        mAP_values = df[target_metric]
        
        sns.lineplot(
            x=epochs, 
            y=mAP_values, 
            label=exp_name, 
            linewidth=2.5,
            color=colors[idx % len(colors)]
        )

    plt.title("船舶小目标检测算法消融实验验证 —— mAP@0.5 对比", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("训练迭代轮次 (Epochs)", fontsize=14)
    plt.ylabel("平均精度 mAP@0.5", fontsize=14)
    plt.legend(title="不同实验网络模型", title_fontsize='13', fontsize='12', loc='lower right')
    
    # 将坐标轴和刻度调大
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, None)
    
    # 存为论文使用的高质量图
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.show()
    
    print(f"\n[+] 成功导出消融实验对比图：{Path(save_path).absolute()}")

if __name__ == '__main__':
    runs_dir = Path(__file__).resolve().parent.parent.parent / "runs" / "train"
    
    # 定义在同一轴里的几个实验路径
    test_dict = {
        "Baseline (未改进 YOLOv12)": str(runs_dir / "exp_D0_baseline" / "results.csv"),
        # "加入 P2 微小目标检测头": str(runs_dir / "exp_M1_P2Head" / "results.csv"),
        # "P2 + SPD-Conv 融合": str(runs_dir / "exp_M2_SPD_Conv" / "results.csv"),
        # "P2 + SPD + EIoU (本文最终模型)": str(runs_dir / "exp_M3_EIoU_Final" / "results.csv")
    }

    # 导出到本脚本的同级目录中
    output_png = Path(__file__).resolve().parent / "ablation_comparison.png"
    
    # 注释掉下面这一行以防你现在本地没有 results.csv 报错。
    # 等你跑完了 2 个以上实验，把字典解开注释，再运行这行代码：
    # plot_ablation_curves(test_dict, save_path=str(output_png))
    print("对比绘图器加载完毕... 请在具有多个 runs/results.csv 后取消代码注释进行出图。")
