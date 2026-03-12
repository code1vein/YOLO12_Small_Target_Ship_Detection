import shutil
from pathlib import Path


def should_skip_generated_file(file_stem: str, excluded_suffixes: tuple[str, ...]) -> bool:
    return any(file_stem.endswith(suffix) for suffix in excluded_suffixes)


def init_augmentation_pool(
    src_dir: Path,
    dst_dir: Path,
    excluded_suffixes: tuple[str, ...] = (),
) -> None:
    if not dst_dir.exists():
        print(f"首次运行，正在从 {src_dir.name} 初始化增强数据池 {dst_dir.name}...")

        if not excluded_suffixes:
            shutil.copytree(src_dir, dst_dir)
        else:
            for folder in ["images", "labels"]:
                for split_name in ["train", "val", "test"]:
                    (dst_dir / folder / split_name).mkdir(parents=True, exist_ok=True)

            for split_name in ["train", "val", "test"]:
                src_image_dir = src_dir / "images" / split_name
                src_label_dir = src_dir / "labels" / split_name
                dst_image_dir = dst_dir / "images" / split_name
                dst_label_dir = dst_dir / "labels" / split_name

                if src_image_dir.exists():
                    for image_path in src_image_dir.iterdir():
                        if not image_path.is_file() or should_skip_generated_file(image_path.stem, excluded_suffixes):
                            continue
                        shutil.copy2(image_path, dst_image_dir / image_path.name)

                if src_label_dir.exists():
                    for label_path in src_label_dir.glob("*.txt"):
                        if should_skip_generated_file(label_path.stem, excluded_suffixes):
                            continue
                        shutil.copy2(label_path, dst_label_dir / label_path.name)

            yaml_path = src_dir / "ship_dataset.yaml"
            if yaml_path.exists():
                shutil.copy2(yaml_path, dst_dir / "ship_dataset.yaml")

        # 初始化后同步修正 yaml 中的数据集路径配置
        yaml_path = dst_dir / "ship_dataset.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_content = file.read()
            yaml_content = yaml_content.replace(str(src_dir.absolute()), str(dst_dir.absolute()))
            yaml_content = yaml_content.replace("dataset_baseline", "dataset_augmentation")
            with open(yaml_path, "w", encoding="utf-8") as file:
                file.write(yaml_content)
        print("增强数据池初始化完毕\n")
    else:
        print(f"检测到增强数据池已存在，将在现有的 {dst_dir.name} 上执行受控增量更新")


def count_dataset_images(dataset_dir: Path, excluded_suffixes: tuple[str, ...] = ()) -> int:
    total = 0
    for split_name in ["train", "val", "test"]:
        image_dir = dataset_dir / "images" / split_name
        if not image_dir.exists():
            continue
        total += sum(
            1
            for path in image_dir.iterdir()
            if path.is_file() and not should_skip_generated_file(path.stem, excluded_suffixes)
        )
    return total


def remove_generated_variants(image_dir: Path, label_dir: Path, suffix: str) -> int:
    # 删除指定后缀的历史增强样本
    removed = 0
    for img_path in image_dir.glob(f"*{suffix}.*"):
        # 图片和同名标签一起删除
        img_path.unlink(missing_ok=True)
        label_path = label_dir / f"{img_path.stem}.txt"
        label_path.unlink(missing_ok=True)
        removed += 1
    return removed


def remove_generated_prefixes(image_dir: Path, label_dir: Path, prefix: str) -> int:
    # 删除指定前缀的历史增强样本
    removed = 0
    for img_path in image_dir.glob(f"{prefix}*"):
        if img_path.is_file():
            img_path.unlink(missing_ok=True)
            removed += 1
    for label_path in label_dir.glob(f"{prefix}*.txt"):
        label_path.unlink(missing_ok=True)
    return removed


def load_label_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    with open(label_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def count_label_classes(lines: list[str]) -> tuple[int, int]:
    normal_count = 0
    small_count = 0
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        if class_id == 1:
            small_count += 1
        else:
            normal_count += 1
    return normal_count, small_count


def collect_dataset_label_stats(label_dir: Path) -> tuple[int, int, int]:
    # 汇总整个标签目录中的文件数与两类目标框数量
    normal_count = 0
    small_count = 0
    file_count = 0
    for label_path in label_dir.glob("*.txt"):
        file_count += 1
        current_normal, current_small = count_label_classes(load_label_lines(label_path))
        normal_count += current_normal
        small_count += current_small
    return file_count, normal_count, small_count


def summarize_ratio(normal_count: int, small_count: int) -> str:
    total = normal_count + small_count
    if total == 0:
        return "0.00%"
    return f"{small_count / total:.2%}"