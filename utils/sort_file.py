# 对指定目录下的文件进行排序，按照时间从早到晚，依次为01，02...


from pathlib import Path
from datetime import datetime
import uuid

# ==================== 配置区域 ====================

# 指定需要整理的目录
TARGET_DIR = Path(r"D:\07_Images\01_学习截屏\每个y分量的计算过程")

# 排序依据：
# modified：文件最后修改时间
# created：文件创建时间（适用于 Windows）
TIME_MODE = "modified"

# =================================================


def get_timestamp(file):
    stat = file.stat()

    if TIME_MODE == "created":
        return getattr(stat, "st_birthtime", stat.st_ctime)

    return stat.st_mtime


def main():
    if not TARGET_DIR.is_dir():
        print(f"目录不存在：{TARGET_DIR}")
        return

    # 获取当前目录下的所有文件，不递归子目录
    files = [
        p
        for p in TARGET_DIR.iterdir()
        if p.is_file() and p.resolve() != Path(__file__).resolve()
    ]

    if not files:
        print("目录下没有文件。")
        return

    # 按时间从早到晚排序，相同时间按文件名排序
    files.sort(key=lambda p: (get_timestamp(p), p.name.lower()))

    # 至少使用两位数字，例如 01、02、03
    width = max(2, len(str(len(files))))

    # 生成重命名计划
    plan = []

    print("\n重命名预览：")
    print("-" * 55)

    for index, file in enumerate(files, start=1):
        new_name = f"{index:0{width}d}{file.suffix}"
        new_path = TARGET_DIR / new_name

        timestamp = datetime.fromtimestamp(get_timestamp(file)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print(f"{file.name}  ->  {new_name}  [{timestamp}]")
        plan.append((file, new_path))

    print("-" * 55)
    print(f"共找到 {len(files)} 个文件。")

    # 确认操作
    confirm = input("\n确认执行重命名？输入 y 确认：")

    if confirm.lower() != "y":
        print("已取消操作。")
        return

    # 第一步：先全部重命名为临时名称，避免文件名冲突
    staged = []

    try:
        for index, (src, target) in enumerate(plan):
            temp = TARGET_DIR / f".__rename_tmp_{uuid.uuid4().hex}_{index}.tmp"
            src.rename(temp)
            staged.append((src, temp, target))

    except Exception as e:
        for src, temp, target in reversed(staged):
            if temp.exists():
                temp.rename(src)
        print(f"临时重命名失败：{e}")
        return

    # 第二步：重命名为最终名称
    completed = []

    try:
        for src, temp, target in staged:
            temp.rename(target)
            completed.append((src, temp, target))

    except Exception as e:
        # 出错时尝试恢复原始文件名
        for src, temp, target in reversed(completed):
            if target.exists():
                target.rename(temp)

        for src, temp, target in reversed(staged):
            if temp.exists():
                temp.rename(src)

        print(f"重命名失败，已尝试恢复原始文件名：{e}")
        return

    print("\n重命名完成！")


if __name__ == "__main__":
    main()
