# YOLO12 Small Target Ship Detection

本项目为基于 YOLO12 的船舶小目标检测项目，以 iShip-1 数据集为基础，对数据集进一步增强。同时针对小目标检测设计了 P2 检测头 / SPD 风格下采样 / Neck 融合增强等网络架构改进，并编写了用于结果展示和模型对比的 Gradio 可视化界面。

## 项目简介

平视海面场景下的小目标船舶检测具有以下典型难点：

- 目标尺寸极小，经过多次下采样后容易丢失有效细节
- 海平线、浪花、反光和雾雨等背景干扰明显
- 小目标与背景对比度低，召回率和定位精度容易下降

针对这些问题，本项目从两个方向展开：

- 数据层面：重构标签体系，构建基线数据集与增强数据集，并通过天气增强、Copy-Paste 与 AIGC 样本整合提升小目标样本质量与分布
- 模型层面：以 YOLO12m 为基线，设计 M0-M6 消融实验链路，逐步验证 P2、SPD 和 Neck 改进对小目标检测性能的影响

## 项目目标

- 构建适用于平视海面小目标船舶检测的二分类数据集流程
- 建立统一可复现实验链路，包括基线训练、增强训练与结构消融
- 验证不同结构改进对小目标检测的增益效果
- 提供可视化推理界面，支持单图检测、批量检测和模型对比

## 项目目录

```text
yolo12/
├─ app.py                           # Gradio 推理与可视化界面
├─ config.py                        # 全局路径与训练配置
├─ README.md                        # 项目说明文档
├─ dataset/
│  ├─ convert_labels_to_2class.py   # 原始标签转二分类标签
│  ├─ rename_dataset.py             # 数据语义化重命名
│  ├─ split_dataset.py              # 数据集划分脚本
│  └─ augmentation/
│     ├─ build_augmentation_dataset.py  #创建增强数据集总脚本
│     ├─ weather_simulation.py          #天气增强脚本
│     ├─ copypaste_data.py              #copy-paste增强脚本
│     └─ integrate_aigc_data.py         #整合aigc生成图像脚本
├─ models/
│  └─ ablation/
│     ├─ yolo12m_m2_p2.yaml          #添加p2头的yolo12m网络架构
│     ├─ yolo12m_m3_spd.yaml         #添加spd的yolo12m网络架构
│     ├─ yolo12m_m4_neck.yaml        #添加neck的yolo12m网络架构
│     ├─ yolo12m_m5_p2_spd.yaml      #添加p2头和spd的yolo12m网络架构
│     └─ yolo12m_m6_p2_spd_neck.yaml #添加p2头，spd和neck的yolo12m网络架构
├─ train_scripts/
│  ├─ train_config.py                   # 统一训练入口与公共训练参数配置
│  ├─ train_baseline.py                 # YOLO12m 在 baseline 数据集上的基线训练脚本
│  ├─ train_augmentation.py             # YOLO12m 在 augmentation 数据集上的增强组训练脚本
│  ├─ train_yolo12_ablation_m2_p2.py    # M2：引入 P2 检测头的 YOLO12 消融训练脚本
│  ├─ train_yolo12_ablation_m3_spd.py   # M3：引入 SPD 风格下采样的 YOLO12 消融训练脚本
│  ├─ train_yolo12_ablation_m4_neck.py  # M4：增强 Neck 融合的 YOLO12 消融训练脚本
│  ├─ train_yolo12_ablation_m5_p2_spd.py # M5：联合引入 P2 与 SPD 的 YOLO12 消融训练脚本
│  ├─ train_yolo12_ablation_m6_p2_spd_neck.py # M6：联合引入 P2、SPD 与 Neck 的最终模型训练脚本
│  ├─ train_yolov11_augmentation.py     # YOLO11m 在 augmentation 数据集上的对比训练脚本
│  ├─ train_yolov8_augmentation.py      # YOLOv8m 在 augmentation 数据集上的对比训练脚本
│  └─ train_yolov5_augmentation.py      # YOLOv5m/YOLOv5u 在 augmentation 数据集上的对比训练脚本
├─ runs/                            # 训练与推理输出目录
└─ tools/                           # 实验记录模板与结果绘图工具
```


## 环境依赖

建议使用 Python 3.10 及以上版本，并准备可用的 CUDA 环境用于训练。

项目已提供 requirements.txt，推荐直接安装：

```bash
pip install -r requirements.txt
```

如果你希望手动安装，可参考下面这一组依赖：

```bash
pip install ultralytics opencv-python numpy pillow gradio albumentations requests tqdm matplotlib pandas seaborn
```

说明：

- ultralytics 会自动依赖安装 PyTorch；若使用 GPU 训练，建议根据你的 CUDA 版本手动确认 PyTorch 版本是否匹配
- 若仅进行推理和界面展示，也建议保留 ultralytics、opencv-python、gradio、pillow 等核心依赖

## Quick Start

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd yolo12
```

### 2. 准备环境

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 数据集准备（可选）
我已提供划分好的基线数据集和增强数据集可直接使用

链接：

若想要体验完整的数据集准备过程可以进行下述操作

将原始 iShip-1 数据放入 dataset/dataset_original 后，按以下顺序执行：

```bash
python dataset/convert_labels_to_2class.py
python dataset/rename_dataset.py
python dataset/split_dataset.py
```

构建增强数据集

```bash
python dataset/augmentation/build_augmentation_dataset.py
```

该步骤会串联执行天气增强、Copy-Paste 和 AIGC 样本整合，并输出到 dataset/dataset_augmentation。

### 4. 模型训练（可选）
我已提供训练好的各消融实验的模型权重可直接使用

链接：

若想要体验完整的模型训练过程可以进行下述操作

重要！训练前请将数据集中的ship_dataset.yaml中的数据集路径改为你自己的路径!

基线训练：

```bash
python train_scripts/train_baseline.py
```

增强数据集训练：

```bash
python train_scripts/train_augmentation.py
```

消融实验训练示例：

```bash
python train_scripts/train_yolo12_ablation_m2_p2.py
python train_scripts/train_yolo12_ablation_m3_spd.py
python train_scripts/train_yolo12_ablation_m4_neck.py
python train_scripts/train_yolo12_ablation_m5_p2_spd.py
python train_scripts/train_yolo12_ablation_m6_p2_spd_neck.py
```

横向对比实验训练示例：

```bash
python train_scripts/train_yolov5_augmentation.py
python train_scripts/train_yolov8_augmentation.py
python train_scripts/train_yolov11_augmentation.py
```


### 5. 启动可视化界面

```bash
python app.py
```

启动后可在浏览器中使用以下功能：

- 单图检测
- 模型对比
- 批量检测
- 结果导出

## 配置说明

项目的主要路径与训练参数集中定义在 config.py 中

## 推理界面说明

app.py 提供了一个面向展示和测试的 Gradio 前端

界面支持：

- 自动扫描项目根目录与 runs/train 下的权重文件
- 使用中文类别标签进行检测结果绘制
- 输出目标数、类别占比、置信度和目标尺寸统计
- 保存检测图片、JSON 结果和批量压缩包

## 注意事项

- 若使用本项目进行公开发布或二次开发，请优先检查数据集授权与模型权重使用协议

## 致谢与引用

如果本项目对你的研究或工程工作有帮助，请关注并引用以下基础工作：

### YOLO12

- Ultralytics YOLO12
- Official Docs: https://docs.ultralytics.com/models/yolo12/
@article{tian2025yolo12,
  title={YOLO12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@software{yolo12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLO12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {https://github.com/sunsmarterjie/yolov12},
  license = {AGPL-3.0}
}

### iShip-1 数据集

- iShip-1 Ship Detection Dataset
@InProceedings{Li_2024_ACCV,
author = {Li, Lingya and Hou, Zhixing and Ma, Ming and Xiang, Jing and Yuan, Chuangxin and Xia, Guihua},
title = {Spotlight on Small-scale Ship Detection: Empowering YOLO with Advanced Techniques and a Novel Dataset},
booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
month = {December},
year = {2024},
pages = {784-799}
}

## License

本仓库默认仅作为学习使用。

在公开传播、商用部署或二次分发前，请自行确认以下内容：

- YOLO12 相关代码与模型的许可证要求
- iShip-1 数据集的使用协议
- 第三方依赖库与外部生成服务的使用限制
