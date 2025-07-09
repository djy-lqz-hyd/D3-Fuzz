# 🌟 一、系统环境


> ubuntu20.04
> pytorch1.11.0
> cuda 11.3
> Anaconda_conda 4.10.3
> numpy-1.21.6
> g++ gcc 10

#  🌟 二、源代码编译

## 🌟🌟 2.1、安装常见依赖项

```bash
 系统依赖
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip git

# Python 依赖
pip3 install --upgrade pip setuptools wheel
pip3 install numpy scipy coverage

# 安装 Bazel (需版本匹配，JAX 0.3.14需要Bazel 5.1.1)
wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh
chmod +x bazel-5.1.1-installer-linux-x86_64.sh
./bazel-5.1.1-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"

# GPU 支持（可选）
# 安装对应版本的 CUDA 和 cuDNN
```
## 🌟🌟 2.2、获取源码

```bash
git clone https://github.com/google/jax.git
cd jax
git checkout jax-v0.3.14  # 确认标签是否正确
```
我们这里要编辑的是jax版本为0.3.14
## 🌟🌟 2.3、配置构建参数
修改jax目录下的.bazelrc文件，添加所需参数

```powershell
build:linux --copt=-Wno-stringop-truncation
#build:linux --copt=-Wno-array-parameter 注销掉这行参数
#编译器警告（如可能未初始化）默认被视为 错误（-Werror）；
#同时还启用了 -Wno-array-parameter（只适用于 C++）而 C 编译器识别不了；
#当前构建了 C 文件（upb/table.c），这些设置再次引发错误

#加入编译条件 覆盖率条件C++插桩 覆盖率选项
build --copt=-O3
build --copt=--coverage
build --linkopt=--coverage
build --copt=-fprofile-arcs
build --copt=-ftest-coverage
build --per_file_copt=.*\\.cpp@-fpermissive
build --per_file_copt=.*\\.cpp@-Wno-class-memaccess
#build --copt=-fpermissive
#build --copt=-Wno-class-memaccess
build --copt=-Wno-error
build --copt=-Wno-maybe-uninitialized
```
## 🌟🌟 2.4、配置外部依赖
有很多包链接都不能下载，手动下载到本地引入WORKSPACE
生成对应的sha256校验码

```bash
http_archive(
    name = "org_tensorflow",
    sha256 = "a99890443df024e52d9c7b075e9916250c6cc6b778d62c384b7dcd1903d8f4f1",  # 保留原始 SHA256 校验值
    strip_prefix = "tensorflow-d250676d7776cfbca38e8690b75e1376afecf58d",
    urls = ["file:///home/tensorflow-d250676d7776cfbca38e8690b75e1376afecf58d.tar.gz"]
)
http_archive(
    name = "org_tensorflow_runtime",
    sha256 = "c554b8c9ed2e34363f9366d4be0ea4fb905dbd824ea5deb5dfe23b42b8eb432a",  # 使用上一步计算的校验值
    strip_prefix = "runtime-1a28370b26c23d9d7c9399896ea5eba23bec029f",
    urls = ["file:///home/runtime-1a28370b26c23d9d7c9399896ea5eba23bec029f.tar.gz"]
)
http_archive(
    name = "org_llvm_llvm_project",
    sha256 = "4138bb3f83bfb822f3ce795cd7e3e7a7018709d8af4e7b54194cbd81931cc93c",  # 使用上一步计算的校验值
    strip_prefix = "llvm-project-4821508d4db75a535d02b8938f81fac6de66cc26",
    urls = ["file:///home/llvm-project-4821508d4db75a535d02b8938f81fac6de66cc26.tar.gz"]
)
```

## 🌟🌟 2.5、编译
```bash
#清理缓存
bazel clean --expunge
#运行编译
 bazel build   --compilation_mode=opt   --define=enable_coverage=true   --define=android=false   //build:build_wheel
```
编译成功
![在这里插入图片描述](../../../../AppData/typora/picture/f693c62d4dcd4348a778be45386211c3.png)
#  🌟 三、安装Jax
## 🌟🌟 3.1、生成 Wheel 包

```bash
# 创建输出目录
# 执行脚本生成 wheel 包
./bazel-bin/build/build_wheel \
  --output_path=/tmp/jax_wheel \
  --cpu=$(uname -m)  # 自动检测 CPU 架构（如 x86_64）
# 查看生成的 wheel 文件
ls -l /tmp/jax_wheel/*.whl
```
## 🌟🌟 3.2、安装 Wheel 包

```bash
# 安装生成的 wheel 包 Jaxlib
pip install /tmp/jax_wheel/*.whl
#设置环境
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
#安装适应对应包
pip install "numpy<2" "scipy==1.5.4"
# 验证安装
python -c "import jax; print(jax.__version__)"
# 应输出 0.3.14
#安装JAX 不需要C++插桩，只需要Python安装即可
pip install -e .  # installs jax
```

#  🌟 四、查找JAX和JAXlib的源码路径
## 🌟🌟 4.1、JAXlib

```powershell
python -c "import jaxlib; print(jaxlib.__file__)"
#/root/miniconda3/envs/Jax/lib/python3.9/site-packages/jaxlib/__init__.py
```
对应的C++动态库（如.so文件）通常位于：

> /path/to/site-packages/jaxlib/_lib/