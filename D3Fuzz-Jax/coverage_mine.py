import os
import subprocess


# 定义根目录
root_dir = "/home/NablaFuzz/NablaFuzz-PyTorch-Jax/output-ad/torch/union"

# 遍历 root_dir 中所有的 "all" 目录
for dirpath, dirnames, filenames in os.walk(root_dir):
    print(f"Checking directory: {dirpath}")
    if os.path.basename(dirpath) == "all":  # 只处理 "all" 目录
        for filename in filenames:
            if filename.endswith(".py"):  # 只处理 Python 文件
                filepath = os.path.join(dirpath, filename)
                print(f"Running {filepath}...")
                # 使用 coverage 运行当前测试文件
                subprocess.run(["coverage", "run", "-a", filepath])