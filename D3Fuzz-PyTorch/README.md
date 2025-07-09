<hr>

@[TOC](目录)

<hr>

# 🌟 一、系统环境


> ubuntu20.04
> pytorch1.11.0
> cuda 11.3
> Anaconda_conda 4.10.3

#  🌟 二、源代码编译

## 🌟🌟 2.1、安装常见依赖项

```bash
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```
## 🌟🌟 2.2、GPU选项（需要用GPU）
如果选择GPU选项需要有这个依赖，重点是在编译的过程中必须使用到硬件GPU，如果不使用可能会导致编译错误
```bash
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda113  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```


## 🌟🌟 2.3、获取 PyTorch 源代码

```bash
#使用git clone 克隆pytorch源码
git clone https://github.com/pytorch/pytorch
# 进入pytorch目录
cd pytorch
# if you are updating an existing checkout
git submodule sync
# 更新第三方库
git submodule update --init --recursive --jobs 0
```
## 🌟🌟 2.4、配置编译环境并进行编译

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#进行安装编译
python setup.py install
```
#  🌟 三、pytorch框架源代码python行覆盖率计算
## 🌟🌟 3.1、环境与工具

```bash
#python工具 
coverage
#以开发者选项安装编译的pytorch源码
python setup.py develop
```
## 🌟🌟 3.2、计算python行覆盖率

> 配置coverage文件

```powershell
[run]
#计算coverage覆盖率的目标路径
source = /home/pytorch/torch
```

> 编写自动化脚本 执行每一条python语句计算覆盖率

```python
import os
import subprocess

# 定义根目录
root_dir = "/home/NablaFuzz/NablaFuzz-PyTorch-Jax/output-ad/torch/union"
# 遍历 root_dir 中所有的 "all" 目录
for dirpath, dirnames, filenames in os.walk(root_dir):
    if os.path.basename(dirpath) == "all":  # 只处理 "all" 目录
        for filename in filenames:
            if filename.endswith(".py"):  # 只处理 Python 文件
                filepath = os.path.join(dirpath, filename)
                print(f"Running {filepath}...")
                # 使用 coverage 运行当前测试文件
                subprocess.run(["coverage", "run", "-a", filepath])
```

> 生成覆盖率报告

**生成命令行报告**：

```bash
coverage report -m
```
**生成HTML 报告**：

```bash
coverage html
```

**生成JSON 报告**：

```bash
coverage json -o coverage.json
```
**清除上次记录**：

```bash
coverage erase
```

#  🌟 四、pytorch框架源代码C++行覆盖率计算
## 🌟🌟 4.1、前置工具

```powershell
sudo apt-get update
#安装gcc工具
sudo apt-get install gcc

sudo apt-get update
#安装C++覆盖率工具
sudo apt-get install lcov
```
## 🌟🌟 4.2、配置基础环境

```powershell
#在编译前，设置环境变量以启用覆盖率支持
export CFLAGS="--coverage"
export CXXFLAGS="--coverage"
export LDFLAGS="--coverage"

#确保清理之前的构建缓存，以避免干扰
python setup.py clean

#使用覆盖率选项重新编译 PyTorch
python setup.py build develop
```

> 验证编译是否成功

```powershell
#使用以下命令在 PyTorch 的构建目录中查找 .gcno 文件
find /home/pytorch/build -name "*.gcno"
```
如果该文件下有这种类型的文件，就可以判断是正确编译了

## 🌟🌟 4.3、计算C++行覆盖率

> 运行python测试脚本

```python
import os
import subprocess

root_dir = "/home/NablaFuzz/NablaFuzz-PyTorch-Jax/output-ad/torch/union"
log_file_path = "./error_log_cov.txt"
with open(log_file_path, "a") as log_file:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if dirpath.endswith("/all"):
                for filename in filenames:
                     if filename.endswith(".py"):
                         filepath = os.path.join(dirpath, filename)
                         print(f"Running {filepath}...")
                         try:
                             subprocess.run(["python", filepath], check=True)
                         except subprocess.CalledProcessError as e:
                             error_msg = f"Error running {filepath}: {e}\n"
                             print(error_msg)
                             log_file.write(error_msg)
```
在运行过程中，PyTorch 的 C++ 后端会生成 `.gcda `文件，这些文件记录了运行时的覆盖率信息，确保测试运行后，`.gcda` 文件已生成，它们通常位于 PyTorch 的 C++ 对象文件目录（如 build/）

> 使用 lcov 收集覆盖率数据

```bash
lcov --directory /home/pytorch/build --capture --output-file coverage.info
```

 - directory 指定包含 .gcda 和 .gcno 文件的目录。 
 - capture 表示捕获覆盖率信息。 
 - output-file 指定生成的覆盖率数据文件

> 生成覆盖率报告

```bash
genhtml coverage.info --output-directory coverage-report
```

 - coverage.info 是上一步骤生成的覆盖率数据文件
 -  coverage-report 是输出报告的目录

> 清理覆盖率数据

清理生成的 .gcda 文件，以便进行下一次测试：

```bash
lcov --directory /path/to/pytorch/build --zerocounters
```

## 🌟🌟 4.4、注意事项
**测试范围**：Python 测试代码需要调用 PyTorch 的 C++ 后端功能，否则无法生成 C++ 覆盖率数据
**文件路径过滤**：如果需要只统计特定模块（如 torch.add）的覆盖率，可以在 lcov 中添加过滤选项：

```bash
lcov --directory /home/pytorch/build --capture --output-file coverage.info --no-external
lcov --extract coverage.info "/home/pytorch/src/torch/add/*" --output-file filtered_coverage.info
```
**性能开销**：启用覆盖率支持后，C++ 代码运行速度可能会变慢