# D3-Fuzz
D3Fuzz first executes an official test suite to collect coverage and identify low-coverage files and critical lines, then assigns weights to these files and employs a dynamic priority scoring function to calculate their coverage score, finally prompts LLM to generate high-quality test cases to explore unknown input spaces.
# 🔬 About

D³Fuzz is a novel fuzzing framework that integrates **Large Language Models (LLMs)** and **coverage feedback mechanisms** to discover deep bugs in deep learning libraries like **PyTorch** and **JAX**. It is designed to overcome limitations of existing fuzzers such as poor test diversity, stagnating code coverage, and lack of direction.

## 🚀 Getting Started

### Step1 : Requirements

1. Our testing framework leverages [MongoDB](https://www.mongodb.com/) so you need to [install and run MongoDB](https://docs.mongodb.com/manual/installation/) first
   - After installing MongoDB and before loading the database, run the command to adjust the limit that the system resources a process may use. You can see this [document](https://docs.mongodb.com/manual/reference/ulimit/) for more details.`ulimit -n 640000`

2. Python version >= 3.9.0 (It must support.)

   - highly recommend to use Python 3.9

3. Check our dependent python libraries in and install with:`requirements.txt`

   ```
   pip install -r requirements.txt
   ```

4. Pytorch Source code compilation
   - Refer to README.md under [D3-Fuzz\D3Fuzz-PyTorch](https://github.com/djy-lqz-hyd/D3-Fuzz/blob/main/D3Fuzz-PyTorch/README.md)

5. Jax Source code compilation

   - Refer to README.md under [D3-Fuzz\D3Fuzz-Jax](https://github.com/djy-lqz-hyd/D3-Fuzz/blob/main/D3Fuzz-Jax/README.md)

   For more information about JAX installation, please refer to the [official website](https://jax.readthedocs.io/en/latest/installation.html#installing-jax).

### Step2 : Set up Database

Run the following commands in current directory (`D3-Fuzz`) to load the database.

```
mongorestore D3-Fuzz/PyTorch-Jax-data/dump
```

### Step2 : Run

Before running, you can set the number of mutants for each API (which is 1000 by default):

```
NUM_MUTANT=100
```

Also, you can set the number of APIs you want to test (which is -1 by default, meaning all APIs will be tested)

```
NUM_API=100
```

#### PyTorch

First go into the `D3-Fuzz/D3Fuzz-PyTorch` directory,Run the following command to start D^3^Fuzz to test PyTorch

```
python torch_test.py --num $NUM_MUTANT --max_api $NUM_API
```

The output will be put in `D3Fuzz-PyTorch/output-ad/torch/union` directory by default.

#### Jax

First go into the `D3-Fuzz/D3Fuzz-Jax` directory,Run the following command to start D^3^Fuzz to test Jax

```
python jax_test.py --num $NUM_MUTANT --max_api $NUM_API
```

The output will be put in `D3Fuzz-Jax/output-ad/jax/union` directory by default.

## 📚 Architecture Overview

```text
           ┌────────────┐
           │   LLM Gen  │ ◀───── prompt-based test seed generation
           └────┬───────┘
                ↓
      ┌────────────────────┐
      │ Mutation Engine    │ ◀──── intelligent mutation + hierarchy
      └────────────────────┘
                ↓
      ┌────────────────────┐
      │ Execution Engine   │ ◀──── safe DL execution + error capture
      └────────────────────┘
                ↓
      ┌────────────────────┐
      │ Coverage Analyzer  │ ◀──── dynamic tracking: lines, funcs
      └────────────────────┘
                ↓
      ┌────────────────────┐
      │ Prioritization     │ ◀──── decay factor + random selection
      └────────────────────┘

```
