# 🌟 1、System environment


> ubuntu20.04
> pytorch1.11.0
> cuda 11.3
> Anaconda_conda 4.10.3
> numpy-1.21.6
> g++ gcc 10

#  🌟 2、Source code compilation

## 🌟🌟 2.1、Install common dependencies

```bash
#System Dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip git

# Python Dependencies
pip3 install --upgrade pip setuptools wheel
pip3 install numpy scipy coverage

# Install Bazel (versions must match, JAX 0.3.14 requires Bazel 5.1.1)
wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh
chmod +x bazel-5.1.1-installer-linux-x86_64.sh
./bazel-5.1.1-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"

# GPU support (optional)
# Install the corresponding versions of CUDA and cuDNN
```
## 🌟🌟 2.2、Get the source code

```bash
git clone https://github.com/google/jax.git
cd jax
git checkout jax-v0.3.14  # Confirm that the label is correct
```
The jax version we want to edit here is 0.3.14
## 🌟🌟 2.3、配置构建参数
Modify the .bazelrc file in the jax directory and add the required parameters

```powershell
build:linux --copt=-Wno-stringop-truncation

#build:linux --copt=-Wno-array-parameter cancels this parameter line
#Compiler warnings (such as possible uninitialization) are treated as errors by default (-Werror);
#Also -Wno-array-parameter is enabled (only for C++) and the C compiler does not recognize it;
#Currently building a C file (upb/table.c), these settings cause errors again
#Add compilation conditions Coverage conditions C++ plug-in Coverage options
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
## 🌟🌟 2.4、Configuring external dependencies
Many package links cannot be downloaded. Manually download them to local and import them into WORKSPACE
Generate the corresponding sha256 checksum

```bash
http_archive(
    name = "org_tensorflow",
    sha256 = "a99890443df024e52d9c7b075e9916250c6cc6b778d62c384b7dcd1903d8f4f1", # Keep the original SHA256 checksum
    strip_prefix = "tensorflow-d250676d7776cfbca38e8690b75e1376afecf58d",
    urls = ["file:///home/tensorflow-d250676d7776cfbca38e8690b75e1376afecf58d.tar.gz"]
)
http_archive(
    name = "org_tensorflow_runtime",
    sha256 = "c554b8c9ed2e34363f9366d4be0ea4fb905dbd824ea5deb5dfe23b42b8eb432a",  # Use the checksum calculated in the previous step
    strip_prefix = "runtime-1a28370b26c23d9d7c9399896ea5eba23bec029f",
    urls = ["file:///home/runtime-1a28370b26c23d9d7c9399896ea5eba23bec029f.tar.gz"]
)
http_archive(
    name = "org_llvm_llvm_project",
    sha256 = "4138bb3f83bfb822f3ce795cd7e3e7a7018709d8af4e7b54194cbd81931cc93c",  # Use the checksum calculated in the previous step
    strip_prefix = "llvm-project-4821508d4db75a535d02b8938f81fac6de66cc26",
    urls = ["file:///home/llvm-project-4821508d4db75a535d02b8938f81fac6de66cc26.tar.gz"]
)
```

## 🌟🌟 2.5、Compile
```bash
#Clear the cache
bazel clean --expunge
#Run the compilation
 bazel build   --compilation_mode=opt   --define=enable_coverage=true   --define=android=false   //build:build_wheel
```
Compilation successful
#  🌟 三、Install Jax
## 🌟🌟 3.1、Generate Wheel Package

```bash
# Create output directory
# Execute script to generate wheel package
./bazel-bin/build/build_wheel \
  --output_path=/tmp/jax_wheel \
  --cpu=$(uname -m)  # Automatically detect CPU architecture (e.g. x86_64)
# View the generated wheel file
ls -l /tmp/jax_wheel/*.whl
```
## 🌟🌟 3.2、Installing the Wheel Package

```bash
# Install the generated wheel package Jaxlib
pip install /tmp/jax_wheel/*.whl
#Set up the environment
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
#Install the corresponding package
pip install "numpy<2" "scipy==1.5.4"
# Verify installation
python -c "import jax; print(jax.__version__)"
# Should output 0.3.14
#Install JAX. No C++ plug-in is required. Only Python installation is required.
pip install -e .  # installs jax
```

#  🌟 4、Find the source path of JAX and JAXlib
## 🌟🌟 4.1、JAXlib

```powershell
python -c "import jaxlib; print(jaxlib.__file__)"
#/root/miniconda3/envs/Jax/lib/python3.9/site-packages/jaxlib/__init__.py
```
The corresponding C++ dynamic library (such as .so file) is usually located at:

> /path/to/site-packages/jaxlib/_lib/