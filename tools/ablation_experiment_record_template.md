# YOLO12 消融实验记录模板

## 1. 消融实验汇总表

本表用于统一记录各组实验的模型配置、控制变量、核心指标与最终结论，便于后续撰写论文中的实验结果总表与消融分析小节。

| 实验编号 | 实验组名称 | 模型配置 | 预训练权重 | 数据集 | 输入尺寸 | 训练轮数 | Batch Size | 受控变量 | 改进内容 | Params | FLOPs | Precision | Recall | mAP50 | mAP50-95 | Small Target AP | 最优 Epoch | 单次训练时长 | 最优权重路径 | 实验结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | YOLO12m 基线组 | train_baseline.py | yolo12m.pt | dataset_baseline | 640 | 150 | 10 | 统一训练超参数与训练策略 | 原始 YOLO12m | 20.139M | 33.8719 GFLOPs | 0.9041 | 0.8737 | 0.9171 | 0.6504 | 0.8530 | 143 | 39337.2s (10h55m) | runs/train/exp0_baseline/weights/best.pt | 作为后续所有消融实验对照基线 |
| M1 | YOLO12m 增强数据集组 | train_augmentation.py | yolo12m.pt | dataset_augmentation | 640 | 150 | 10 | 模型结构与训练超参数保持和基线一致 | 仅将训练数据集由 baseline 替换为 augmentation |  |  |  |  |  |  |  |  |  |  |  |
| M2 | YOLO12m + P2 组 | yolo12m_m1_p2.yaml | yolo12m.pt | dataset_augmentation | 640 | 150 | 10 | 除检测头外其余条件保持一致 | 引入 P2 微尺度高分辨率检测头 |  |  |  |  |  |  |  |  |  |  |  |
| M3 | YOLO12m + SPD 组 | yolo12m_m2_spd.yaml | yolo12m.pt | dataset_augmentation | 640 | 150 | 10 | 除下采样模块外其余条件保持一致 | 引入 SPD 风格无损下采样 |  |  |  |  |  |  |  |  |  |  |  |
| M4 | YOLO12m + Neck 组 | yolo12m_m3_neck.yaml | yolo12m.pt | dataset_augmentation | 640 | 150 | 10 | 除 Neck 结构外其余条件保持一致 | 增强 P3/P4 多尺度特征融合 |  |  |  |  |  |  |  |  |  |  |  |
| M5 | YOLO12m + P2 + SPD 组 | yolo12m_m4_p2_spd.yaml | yolo12m.pt | dataset_augmentation | 640 | 150 | 10 | 除 P2 与 SPD 外其余条件保持一致 | 联合引入 P2 与 SPD |  |  |  |  |  |  |  |  |  |  |  |
| M6 | YOLO12m + P2 + SPD + Neck 组 | yolo12m_m5_p2_spd_neck.yaml | yolo12m.pt | dataset_augmentation | 640 | 150 | 10 | 所有训练条件与基线一致 | 联合引入 P2、SPD 与增强 Neck |  |  |  |  |  |  |  |  |  |  |  |

## 2. 单组实验记录模板

### 2.1 实验基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | M0 |
| 实验组名称 | YOLO12m 基线组 |
| 实验目的 | 在 baseline 数据集上建立统一对照基线，为后续结构改进提供参照 |
| 理论假设 | 原始 YOLO12m 在当前数据集上可稳定收敛，但小目标类别仍有提升空间 |
| 启动脚本 | train_baseline.py |
| 模型配置文件 | yolo12m.pt (无结构改动) |
| 预训练权重 | yolo12m.pt |
| 数据集 YAML | dataset/dataset_baseline/ship_dataset.yaml |
| 开始时间 | 见 runs/train/exp0_baseline 训练日志 |
| 结束时间 | 见 runs/train/exp0_baseline 训练日志 |
| 运行平台 | Windows + CUDA (yolo12 虚拟环境) |
| GPU 型号 | NVIDIA GeForce RTX 3080 (10GB) |

### 2.2 变量控制说明

| 类别 | 内容 |
| --- | --- |
| 自变量 | 无（基线组） |
| 因变量 | Precision、Recall、mAP50、mAP50-95、Small Target AP |
| 控制变量 | imgsz=640、epochs=150、batch=10、优化器与增强策略保持默认统一 |
| 对照组 | 本组即对照组 |
| 比较对象 | 后续 M1-M6 改进组 |

### 2.3 训练参数设置

| 参数名称 | 数值 | 说明 |
| --- | --- | --- |
| imgsz | 640 | 输入图像尺寸 |
| epochs | 150 | 总训练轮数 |
| batch | 10 | 每次迭代样本数 |
| workers | 8 | 数据加载线程数 |
| optimizer | auto | 优化器自动选择 |
| patience | 50 | 早停容忍轮数 |
| mosaic | 0.5 | 四图拼接增强 |
| close_mosaic | 10 | 训练后期关闭 Mosaic |
| scale | 0.2 | 随机缩放幅度 |
| translate | 0.05 | 随机平移幅度 |
| hsv_h | 0.015 | 色相扰动 |
| hsv_s | 0.3 | 饱和度扰动 |
| hsv_v | 0.2 | 亮度扰动 |

### 2.4 定量评价指标

| 指标名称 | 数值 | 备注 |
| --- | --- | --- |
| Params | 20.139M | 基于 exp0_baseline 的 best.pt 统计 |
| FLOPs | 33.8719 GFLOPs | 基于 THOP 在 imgsz=640 下统计 |
| Precision | 0.9041 | 最终第150轮 |
| Recall | 0.8737 | 最终第150轮 |
| mAP50 | 0.9171 | 最终第150轮 |
| mAP50-95 | 0.6504 | 最终第150轮 |
| Small Target AP | 0.8530 | 来自 PR 曲线类别指标 |
| 最优 Epoch | 143 | mAP50-95 最优对应轮次 |
| 最优权重路径 | runs/train/exp0_baseline/weights/best.pt | best.pt 保存位置 |

### 2.5 训练过程观察

| 观察维度 | 记录 |
| --- | --- |
| 收敛速度 | 前50轮收敛明显，后期进入平台区 |
| 损失下降趋势 | train/val loss 持续下降并趋于平稳 |
| 小目标召回变化 | 小目标可检出但相对常规船仍偏弱 |
| 小目标定位精度变化 | 后期 mAP50-95 小幅提升，定位继续细化 |
| 误检类型 | 远距离海平线区域存在背景误检 |
| 漏检类型 | 低对比度、极小尺度目标存在漏检 |
| 对大目标检测的影响 | 常规船检测稳定，AP 表现较高 |
| 与对照组相比的主要差异 | 基线组，无对照差异 |

### 2.6 结果文件归档

| 文件名称 | 路径 | 用途 |
| --- | --- | --- |
| results.csv | runs/train/exp0_baseline/results.csv | 数值结果记录 |
| results.png | runs/train/exp0_baseline/results.png | 总体训练曲线 |
| confusion_matrix.png | runs/train/exp0_baseline/confusion_matrix.png | 混淆矩阵 |
| PR_curve.png | runs/train/exp0_baseline/PR_curve.png | PR 曲线 |
| F1_curve.png | runs/train/exp0_baseline/F1_curve.png | F1 曲线 |
| best.pt | runs/train/exp0_baseline/weights/best.pt | 最优模型权重 |

### 2.7 实验结果分析

1. 与基线组相比，本组在 Precision、Recall、mAP50、mAP50-95 方面的变化为：
	本组为基线组，不涉及与其他组差值，作为后续 M1-M6 的统一参照。

2. 本组在小目标检测性能上的主要增益或退化表现为：
	Small Target AP=0.853，说明小目标具备可检出能力，但相较常规目标仍有优化空间。

3. 若性能提升明显，其可能原因是：
	不适用（基线组）。

4. 若性能未达预期，其可能原因是：
	小目标尺度过小、海天背景低对比度导致特征表达不足，易出现漏检与背景混淆。

5. 该改进是否建议保留进入最终模型：
	基线结构保留为对照，不作为最终改进结论依据。

### 2.8 可直接用于论文的结论表述

本组实验表明，基线 YOLO12m 在 dataset_baseline 上可稳定收敛，最终达到 Precision=0.9041、Recall=0.8737、mAP@0.5=0.9171、mAP@0.5:0.95=0.6504。常规船类别检测表现较好，但小目标类别仍存在一定漏检与背景混淆。该结果可作为后续各改进组（M1-M6）的统一对照基线，用于验证数据增强、P2、SPD 与 Neck 改进在小目标检测场景下的增益有效性。