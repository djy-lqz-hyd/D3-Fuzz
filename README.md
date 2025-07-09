# D-Fuzz
D3Fuzz first executes an official test suite to collect coverage and identify low-coverage files and critical lines, then assigns weights to these files and employs a dynamic priority scoring function to calculate their coverage score, finally prompts LLM to generate high-quality test cases to explore unknown input spaces.
# 🔬 D³Fuzz: Dynamic, Directed and Decaying LLM-Driven Fuzzing for Deep-Learning Libraries

D³Fuzz is a novel fuzzing framework that integrates **Large Language Models (LLMs)** and **coverage feedback mechanisms** to discover deep bugs in deep learning libraries like **PyTorch** and **JAX**. It is designed to overcome limitations of existing fuzzers such as poor test diversity, stagnating code coverage, and lack of direction.

## 🚀 Features

- 🤖 **LLM-based Test Case Generation**  
  Generate high-quality, diverse test inputs for DL APIs with GPT-4-based reasoning and mutation.

- 📊 **Coverage-Guided Mutation and Prioritization**  
  Introduces a dynamic scoring strategy combining coverage-driven decaying factors and random perturbation.

- 🔍 **Multi-Layer Coverage Feedback**  
  Supports **Python** and **C++** level coverage monitoring.

- 🐞 **Rich Bug Detection Categories**  
  - `CRASH`: Fatal errors or runtime crashes  
  - `NaN`: Numerical instability (`nan`, `inf`, overflow)  
  - `ND_FAIL`: Nondeterministic outputs under fixed inputs

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
