import os
import glob
import sys
from pathlib import Path

# 动态添加项目根目录至环境变量，导入全局配置 config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ORIGINAL_LABELS_DIR, ORIGINAL_LABELS_2CLASS_DIR, SMALL_TARGET_THRESHOLD

# 定义路径
src_labels_dir = str(ORIGINAL_LABELS_DIR)
dst_labels_dir = str(ORIGINAL_LABELS_2CLASS_DIR)

# 小目标的面积占比阈值 (0.5%)
THRESHOLD = SMALL_TARGET_THRESHOLD

def convert_labels():
    if not os.path.exists(src_labels_dir):
        print(f"Error: 找不到源标签文件夹 {src_labels_dir}")
        return

    # 创建目标文件夹
    os.makedirs(dst_labels_dir, exist_ok=True)
    
    # 获取所有的 txt 文件
    txt_files = glob.glob(os.path.join(src_labels_dir, "*.txt"))
    
    if not txt_files:
        print(f"Warning: 在 {src_labels_dir} 中没有找到 .txt 标签文件。")
        return

    print(f"找到 {len(txt_files)} 个标签文件，开始转换...")
    
    # 统计信息
    total_boxes = 0
    small_ship_count = 0
    normal_ship_count = 0

    for file_path in txt_files:
        filename = Path(file_path).name
        dst_path = os.path.join(dst_labels_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # 标准 YOLO 格式 
            if len(parts) == 5:
                cls_id, x, y, w, h = parts
                
                # 计算相对面积 
                area = float(w) * float(h)
                
                # 根据阈值重新分配类别
                if area < THRESHOLD:
                    new_cls = 1
                    small_ship_count += 1
                else:
                    new_cls = 0
                    normal_ship_count += 1
                    
                total_boxes += 1
                
                # 拼接新行
                new_line = f"{new_cls} {x} {y} {w} {h}\n"
                new_lines.append(new_line)
                
        # 写入新文件
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
    print("-" * 30)
    print("转换完成")
    print(f"共处理 {len(txt_files)} 个文件。")
    print(f"发现总边界框数: {total_boxes}")
    print(f"其中 常规船: {normal_ship_count}个 ")
    print(f"其中 小目标船: {small_ship_count}个 ")
    print(f"新的标签文件已保存在: {dst_labels_dir}")

if __name__ == "__main__":
    convert_labels()
