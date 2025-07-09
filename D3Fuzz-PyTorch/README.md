



# 🌟 1、System environment


> ubuntu20.04
> pytorch1.11.0
> cuda 11.3
> Anaconda_conda 4.10.3

#  🌟 2、Source code compilation

## 🌟🌟 2.1、Install common dependencies

```bash
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```
## 🌟🌟 2.2、GPU option (requires GPU)
If you select the GPU option, you need this dependency. The key point is that the hardware GPU must be used during the compilation process. Failure to use it may cause compilation errors
```bash
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda113  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```


## 🌟🌟 2.3、Get PyTorch source code

```bash
#Use git clone to clone the pytorch source code
git clone https://github.com/pytorch/pytorch
# Enter the pytorch directory
cd pytorch
# if you are updating an existing checkout
git submodule sync
# Update third-party libraries
git submodule update --init --recursive --jobs 0
```
## 🌟🌟 2.4、Configure the compilation environment and compile

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#Install and compile
python setup.py install
```
#  🌟 三、Pytorch framework source code python line coverage calculation
## 🌟🌟 3.1、Environment and tools

```bash
#python tools
coverage
#Install compiled pytorch source code with developer options
python setup.py develop
```
## 🌟🌟 3.2、Calculating python line coverage

> Configuring coverage files

```powershell
[run]
#Calculate the target path for coverage
source = /home/pytorch/torch
```

> Write an automated script to execute each Python statement to calculate coverage

```python
import os
import subprocess

# Define the root directory
root_dir = "/home/NablaFuzz/NablaFuzz-PyTorch-Jax/output-ad/torch/union"
# Traverse all "all" directories in root_dir
for dirpath, dirnames, filenames in os.walk(root_dir):
if os.path.basename(dirpath) == "all": # Only process the "all" directory
for filename in filenames:
if filename.endswith(".py"): # Only process Python files
filepath = os.path.join(dirpath, filename)
print(f"Running {filepath}...")
# Run the current test file using coverage
subprocess.run(["coverage", "run", "-a", filepath])
```

> Generate coverage report

**Generate command line report**：

```bash
coverage report -m
```
**Generate HTML report**：

```bash
coverage html
```

**Generate JSON report**：

```bash
coverage json -o coverage.json
```
**Clear last record**：

```bash
coverage erase
```

#  🌟 4、Pytorch framework source code C++ line coverage calculation
## 🌟🌟 4.1、Pre-tools

```powershell
sudo apt-get update
#Install gcc tool
sudo apt-get install gcc

sudo apt-get update
#Install C++ coverage tool
sudo apt-get install lcov
```
## 🌟🌟 4.2、Configure the basic environment

```powershell
#Before compiling, set environment variables to enable coverage support
export CFLAGS="--coverage"
export CXXFLAGS="--coverage"
export LDFLAGS="--coverage"

#Make sure to clean the previous build cache to avoid interference
python setup.py clean

#Recompile PyTorch with coverage options
python setup.py build develop
```

> Verify that the compilation was successful

```powershell
#Use the following command to find the .gcno file in the PyTorch build directory
find /home/pytorch/build -name "*.gcno"
```
If there is a file of this type under this file, it can be judged that it is compiled correctly.

## 🌟🌟 4.3、

> Calculating C++ Line Coverage

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
During the run, PyTorch's C++ backend generates `.gcda` files, which record the coverage information of the run time. Make sure that after the test is run, the `.gcda` files have been generated. They are usually located in the PyTorch C++ object file directory (such as build/)

> Collecting coverage data using lcov

```bash
lcov --directory /home/pytorch/build --capture --output-file coverage.info
```

- directory specifies the directory containing .gcda and .gcno files.
- capture indicates capturing coverage information.
- output-file specifies the generated coverage data file

> Generate coverage report

```bash
genhtml coverage.info --output-directory coverage-report
```

- coverage.info is the coverage data file generated in the previous step
- coverage-report is the directory for the output report

> Cleaning coverage data

Clean up the generated .gcda files for the next test:

```bash
lcov --directory /path/to/pytorch/build --zerocounters
```

## 🌟🌟 4.4、Precautions
**Test scope**: Python test code needs to call PyTorch's C++ backend function, otherwise C++ coverage data cannot be generated
**File path filtering**: If you need to only count the coverage of a specific module (such as torch.add), you can add filtering options in lcov:

```bash
lcov --directory /home/pytorch/build --capture --output-file coverage.info --no-external
lcov --extract coverage.info "/home/pytorch/src/torch/add/*" --output-file filtered_coverage.info
```
**Performance Overhead**: C++ code may run slower when coverage support is enabled