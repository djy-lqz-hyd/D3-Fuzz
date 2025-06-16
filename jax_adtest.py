import jax
from helper_jax import DirectInv, RevInv, FwdInv, NDCheck, allow_error, Grad
from classes.jax_library import JaxLibrary
from classes.jax_api import JaxAPI
from classes.database import JaxDatabase
from constant.returntypes import ResType
from utils.printer import dump_data
from pathlib import Path
from random import choice
import os
import argparse
import coverage
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import re
import time
import random
import subprocess as sp

jax.config.update("jax_enable_x64", True)
JaxDatabase.database_config("127.0.0.1", 27017, "jax-test-unique")

# 全局状态维护（可通过闭包实现，这里简化使用全局变量）
processed_files = set()
#动态衰减因子
#decay_factors = {}
#前一段覆盖率
before_coverage = 0
DECAY_FILE = "decay_factors.json"

# 读取衰减因子文件
def load_decay_factors():
    if os.path.exists(DECAY_FILE):
        with open(DECAY_FILE, "r") as f:
            return json.load(f)
    return {}  # 如果文件不存在，返回空字典

#更新衰减因子文件    
def update_decay_factor(decay_factors):
    with open(DECAY_FILE, "w") as f:
        json.dump(decay_factors, f, indent=4)  # 格式化存储，方便查看
        
#更新处理过的文件
def update_processed_files(file_name,before_coverage):
    code_status = 0
    if os.path.exists('coverage_Python_report_stage.csv'):
       code_status = compare_last_two_percentages('coverage_Python_report_stage.csv')
    if(code_status == 1):
        processed_files.add(file_name)

def compare_last_two_percentages(file_path):
    # 读取CSV文件（无表头，制表符分隔）
    df = pd.read_csv(file_path, sep='\t', header=None)

    # 检查至少需要两行数据
    if len(df) < 2:
        print("错误：文件需要至少包含两行数据")
        return

    # 获取最后两行
    last_two = df.tail(2).reset_index(drop=True)
    list_fron = last_two[0][0]
    list_back = last_two[0][1]
    percent_str_fron = list_fron.split(',')[1].strip('%')
    percent_str_back = list_back.split(',')[1].strip('%')
    fron = float(percent_str_fron)
    back = float(percent_str_back)
    if(fron == back):
        return 1
    else:
        return 0

# 递归查找函数
def find_numbers(data, target):
    for item in data:
        if isinstance(item, tuple):
            path, content = item
            if path == target:
                for sub in content:
                    if isinstance(sub, tuple) and sub[0] == target:
                        return sub[1]
            elif isinstance(content, list):
                result = find_numbers(content, target)
                if result:
                    return result
    return []

##解析覆盖率数据JAX
def analyze_coverage():
    with open('coverage.json') as f:
        data = json.load(f)

    # 计算各文件覆盖率
    coverage_stats = {}
    for file, info in data['files'].items():
        if 'jax' not in file:  # 仅关注JAX源码
            continue
        total_lines = info['summary']['num_statements']
        covered_lines = info['summary']['covered_lines']
        missing_lines = info['missing_lines']
        coverage_pct = (covered_lines / total_lines) * 100 if total_lines > 0 else 0

        coverage_stats[file] = {
            'pct': coverage_pct,
            'missing': missing_lines
        }

    return coverage_stats

#识别低覆盖率区域定位
def find_low_coverage_files(stats, threshold=70):
    return {
        f: data for f, data in stats.items()
        if data['pct'] < threshold
    }

#提取关键未覆盖代码JAX
def get_critical_lines(file_path, missing_lines):
    """多维度关键代码行识别 - 针对JAX框架"""
    critical = []
    line_total = []
    line_fileName = []
    line_target = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line_no in missing_lines:
        line = lines[line_no - 1].strip()

        # 1. 控制流相关
        control_flow_keywords = {'if', 'elif', 'else', 'for', 'while',
                                 'try', 'except', 'finally', 'raise',
                                 'assert', 'return', 'yield'}
        if any(kw in line for kw in control_flow_keywords):
            critical.append((file_path, line_no, line, 'control_flow'))
            line_total.append(line_no)
            line_target.append((line_no, line))
            continue

        # 2. JAX 核心计算函数（如 jax.numpy, jax.lax, jax.scipy）
        if re.search(r'\bjax\.(numpy|lax|scipy|random)\.', line):
            critical.append((file_path, line_no, line, 'jax_math_core'))
            line_total.append(line_no)
            line_target.append((line_no, line))
            continue

        # 3. JIT / grad / vmap / pmap 等变换函数
        if re.search(r'\bjax\.(jit|grad|vmap|pmap|scan|remat)\b', line):
            critical.append((file_path, line_no, line, 'transform_primitive'))
            line_total.append(line_no)
            line_target.append((line_no, line))
            continue

        # 4. 设备放置和跨设备执行（XLA/TPU/GPU等）
        if re.search(r'\bjax\.(device_put|devices|local_devices|get_device_platform)\b', line):
            critical.append((file_path, line_no, line, 'device_handling'))
            line_total.append(line_no)
            line_target.append((line_no, line))
            continue

        # 5. XLA相关后端调用（如xla、compilation、lowering等）
        if re.search(r'(xla|lowering|backend)', line, re.IGNORECASE):
            critical.append((file_path, line_no, line, 'xla_backend'))
            line_total.append(line_no)
            line_target.append((line_no, line))
            continue

        # 6. 编译/缓存行为检测
        if 'cache' in line or 'compilation_cache' in line:
            critical.append((file_path, line_no, line, 'compile_cache'))
            line_total.append(line_no)
            line_target.append((line_no, line))
            continue

    line_fileName.append((file_path, line_total))
    return critical, line_fileName, line_target

#JAX代码
def prioritize_files(low_coverage, top_k=5, shuffle_window=3):
    """基于JAX测试套件特性的优先级权重分配"""
    priority_rules = {
        # 主模块类别: (路径关键词, 基础权重, 子模块额外权重)
        'autodiff': (['autodiff', 'grad'], 4.0, {
            'higher_order': 0.3,    # 高阶梯度测试
            'custom_vjp': 0.3,      # 自定义反向规则
            'holomorphic': 0.2      # 全纯函数支持
	    }),
        'jit': (['jit', 'xla'], 4.2, {          # JIT编译与XLA集成
            'cache_miss': 0.4,      # 编译缓存失效场景
            'device_placement': 0.3,# 设备放置逻辑
            'shape_poly': 0.3       # 多态形状支持
	    }),
        'pmap': (['pmap', 'shard'], 3.8, {       # 数据并行
            'axis_groups': 0.4,      # 设备分组策略
            'collective_ops': 0.3,  # 集合通信操作
            'memory_optimize': 0.3
	    }),
        'linalg': (['linalg'], 3.5, {           # 线性代数核心
            'eigen': 0.3,         # 特征值计算
            'svd': 0.3,           # 奇异值分解
            'matrix_exp': 0.2
	    }),
	    'random': (['random'], 3.0, {           # 随机数生成
            'prng': 0.4,          # 并行随机数生成器
            'threefry': 0.3,      # 底层算法实现
            'distribution': 0.3
	    }),
        'device_management': (['device'], 3.2, {# 设备管理
            'multi_device': 0.5,   # 多设备协调
            'gpu_specific': 0.3,  # GPU专属特性
            'tpu_special': 0.2    # TPU特殊处理
	    }),
	    'custom_derivatives': (['custom_derivatives'], 3.6, {
            'defjvp': 0.4,        # 前向模式定义
            'defvjp': 0.4,        # 反向模式定义
            'jaxpr_analysis': 0.2 # 中间表示验证
	    }),
        'dynamic_shape': (['dynamic_shape'], 3.3, {# 动态形状支持
            'symbolic_dim': 0.5,
            'lazy_expr': 0.5
	    }),
        'xmap': (['xmap'], 3.4, { # 多维并行
            'axis_resources': 0.4,
            'vectorization': 0.4,
            'hardware_mapping': 0.2
	    }),
        'sparse': (['sparse'], 2.8, {           # 稀疏计算
            'bcoo': 0.6,          # BCOO格式支持
            'spmv': 0.4
        })
    }
    prioritized = []
    line_fileNames = []
    critical_lines_all = []
    line_target = []
    score_ = 0
    #critical = []
    for file_path, data in low_coverage.items():
        # 基础权重计算
        base_weight = 1.0
        sub_weight = 0.0
        for category, (keywords, weight, submodules) in priority_rules.items():
            if any(kw in file_path for kw in keywords):
                base_weight = weight
                # 子模块加权
                for sub_key, sub_val in submodules.items():
                    if sub_key in file_path:
                        sub_weight += sub_val
                break

        # 关键行权重加成
        critical_lines,line_fileName,line_target_ = get_critical_lines(file_path, data['missing'])
        critical_lines_all.append(critical_lines)
        line_fileNames.append((file_path,line_fileName))
        #critical.append((file_path,critical_lines))
        critical_factor = 1 + 0.1 * len(critical_lines)  # 每关键行+10%权重
        #拿到所有文件的动态衰减因子
        decay_factors = load_decay_factors()
        #应用动态衰减因子
        decay = decay_factors.get(file_path, 1.0)
        # 最终得分 = (100-覆盖率) * 权重 * 关键因子
        score = (100 - data['pct']) * (base_weight + sub_weight) * critical_factor * decay
        if(score > score_):
            score_ = score
            line_target = line_target_
            
        prioritized.append((file_path, score))  
         
    sorted_prioritized = sorted(prioritized, key=lambda x: -x[1])

    # 动态调整Top-K 随机扰动
    top_candidates = sorted_prioritized[:top_k*2]  # 取双倍候选
    
    # 随机扰动前N名
    if len(top_candidates) >= shuffle_window:
        shuffle_part = list(top_candidates[:shuffle_window])
        random.shuffle(shuffle_part)
        top_candidates = shuffle_part + top_candidates[shuffle_window:]
        
    # 最终选择并更新衰减因子
    final_selection = []
    code_status = 0
    for file_info in top_candidates[:top_k]:
        file_path, score = file_info
        if os.path.exists('coverage_Python_report_stage.csv'):
            code_status = compare_last_two_percentages('coverage_Python_report_stage.csv')
        if(code_status == 1):
            decay_factors[file_path] = decay_factors.get(file_path, 1.0) * 0.7
        else:
            decay_factors[file_path] = decay_factors.get(file_path, 1.0)
        final_selection.append((file_path, score))
    
    update_decay_factor(decay_factors)


    return final_selection,line_fileNames,line_target
    
#转换list
def convert_str_list(string):
    # 提取第一行
    first_line = string.splitlines()[0]
    # 正则提取方括号内部的内容
    match = re.search(r'\[(.*?)\]', first_line)
    result_list=[]
    if match:
        result = match.group(1)
        # 按逗号分割并去除空格
        result_list = [item.strip() for item in result.split(',')]
        #print(result_list)
    else:
        print("未找到匹配内容")
    return result_list

#提取相关代码
def extract_sections(content):
    lines = content.strip().split('\n')
    sections = {
        'fn_code': '',
        'inv_code': '',
        'input_list': ''
    }
    current_section = None
    code_buffer = []
    in_code_block = False

    for line in lines:
        stripped_line = line.strip()
        if line.startswith('###'):
            if '函数逻辑模板' in line:
                current_section = 'fn_code'
            elif '输入初始化模板' in line:
                current_section = 'inv_code'
            elif '输入列表' in line:
                current_section = 'input_list'
            else:
                current_section = None
            in_code_block = False
            code_buffer = []
        elif current_section and not in_code_block and stripped_line == '```python':
            in_code_block = True
            code_buffer = []
        elif current_section and in_code_block:
            if stripped_line == '```':
                sections[current_section] = '\n'.join(code_buffer)
                current_section = None
                in_code_block = False
            else:
                code_buffer.append(line)
    return sections['fn_code'], sections['inv_code'], sections['input_list']

def gen_prompt(file_name,target_api,code_context_):
    code_context = code_context_
    mutation_prompt = f"""
    **角色**：你是一个精通JAX测试开发的高级工程师，需要为指定代码段生成符合工业级测试标准的用例。
    **目标上下文**：
    - 文件(参考JAX源码文件)：{file_name}
    - 关键未覆盖行（部分）：{code_context}
    - 目标API：{target_api}
    
    **生成要求**：
    1. 必须执行到上述代码行，验证正常/异常路径
    2. 输入需包含：
        - 至少两种设备类型（CPU/GPU/TPU）和两种数值精度（float32/bfloat16）
        - 边界值（如0, inf, NaN）
    3. 必须包含两类异常测试：
        a) 无效形状或类型（如不符合JAX的严格类型要求）
        b) 非法JAXPR转换（如包含动态形状或非法控制流）
    
	**缩进规范**：
	    必须使用4个空格缩进，禁止使用制表符
	    多级嵌套保持严格缩进对齐
    **生成约束**：
    1. 禁止使用experimental模块（除非目标API明确要求）
    2. 必须包含JIT兼容性检查（使用@jax.jit装饰器）
    3. 梯度计算测试需符合JAX自动微分规范：
         ```python
        def loss_fn():
            return result.sum()
        
        grad_fn = jax.value_and_grad(loss_fn)
        _, grads = grad_fn()
        assert not jax.tree_util.tree_all(jax.tree_map(jnp.isnan, grads))
        ```
	
    **代码模板**（严格遵循JAX规范）：
    ### 函数逻辑模板 (fn_code)
    ```python
    # TEST CASE FOR: {file_name}
    import jax
    import jax.numpy as jnp
    from jax import config, jit 
    from jax import config
    config.update("jax_enable_x64", True)
    def get_fn():
        # 固定随机种子
        def fn({{input_vars}}):
            """ '测试函数核心逻辑' """
            # 输入初始化
            {{input_initialization}}        
            # 目标API调用
            {{target_api_call}}	        
            # 结果验证
            {{validation_logic}}	        
            # 设备迁移测试（可选）
            {{device_migration_test}}	        
            return {{result_vars}}
        return jax.jit(fn)
    fn = get_fn()
    ```
    ###输入初始化模板 (inv_code)
    # 输入张量构造（至少两种类型），根据input_vars参数确定输入张量构造
    ```python
	# 基础输入构造（至少两种精度）
	{input_var} = jax.random.normal(subkeys[0], {shape}, dtype={dtype})
	special_{input_var} = jnp.array({edge_values}, dtype={alt_dtype})
    ```
    # 设备放置（显式指定）
    ```python
	{input_var} = jax.device_put({input_var}, jax.devices('cpu')[0])
    ```
    ###输入列表 (input_list),根据input_vars参数确定输入列表，input_list只能是参数名，不能涉及操作
    ```python
    input_list = [({{input_var}},),(special_{{input_var}},)]
    ```
    请满足上述要求生成代码(保证缩进格式正确，不含有无用空格),覆盖关键未覆盖行，最后只输出函数逻辑 (fn_code)、输入初始化 (inv_code)、输入列表 (input_list)三个部分生成的代码，无需解释说明
    """

    return mutation_prompt
    
#生成代码
def gen(
    engine,
    prompt,
    temperature,
    max_tokens,
    n_samples,
    top_p,
    frequency_penalty,
    presence_penalty,
):
    t_start = time.time()
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-NS7xaopX9KliOJhqsNRptJw2ujVajN5S385l7i9zc8jVhGYI",
        base_url="https://api.chatanywhere.tech/v1"
    )

    completion = client.chat.completions.create(
        model=engine,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=temperature,  # 控制生成文本的创意程度，0.0 到 1.0 之间的值
        max_tokens=max_tokens,  # 限制生成文本的最大长度
        top_p=top_p,  # 通过采样来控制多样性（top-p 采样）
        frequency_penalty=frequency_penalty,  # 减少模型重复使用某些词汇的频率
        presence_penalty=presence_penalty,  # 控制模型是否倾向于引入新的话题
        n=1,  # 生成的回答数量，默认是1
        stop=None,  # 指定停止符，生成文本时达到该符号时停止
    )
    # 提取生成的代码内容
    codes = completion.choices[0].message.content
    t_end = time.time()
    return codes, t_end - t_start


def test(fn, inputs):
    inputs = tuple(inputs)

    direct_status, direct_value, direct_err = DirectInv(fn, inputs)

    for _ in range(9):
        direct_status_, direct_value_, direct_err_ = DirectInv(fn, inputs)
        if direct_status != direct_status_ or not JaxLibrary.is_equal(
            direct_value, direct_value_
        ):
            return ResType.RANDOM

    rev_status, rev_value, rev_grad, rev_err = RevInv(fn, inputs)
    fwd_status, fwd_value, fwd_grad, fwd_err = FwdInv(fn, inputs)
    rev_restype = ResType.PASS
    fwd_restype = ResType.PASS

    if rev_status != direct_status and not allow_error(rev_err):
        rev_restype = ResType.REV_STATUS
    elif rev_status == "fail":
        rev_restype = ResType.SKIP
    elif not JaxLibrary.is_equal(direct_value, rev_value):
        rev_restype = ResType.REV_VALUE

    if fwd_status != direct_status and not allow_error(fwd_err):
        fwd_restype = ResType.FWD_STATUS
    elif fwd_status == "fail":
        fwd_restype = ResType.SKIP
    elif not JaxLibrary.is_equal(direct_value, fwd_value):
        fwd_restype = ResType.FWD_VALUE

    if (
        rev_restype == ResType.PASS
        and fwd_restype == ResType.PASS
        and not JaxLibrary.is_equal(rev_grad, fwd_grad)
    ):
        return ResType.REV_FWD_GRAD

    if rev_restype == ResType.PASS:
        nd_status, nd_err = NDCheck(fn, inputs, "rev")
        if nd_status == "fail":
            rev_restype = ResType.ND_GRAD
    if fwd_restype == ResType.PASS:
        nd_status, nd_err = NDCheck(fn, inputs, "fwd")
        if nd_status == "fail":
            fwd_restype = ResType.ND_GRAD
    return (rev_restype, fwd_restype)


def testAPI(
    api_name,
    num=1000,
    output_dir: Path = Path("../output-ad/jax"),
    mutate=True,
):
    def get_clean_counts(counts):
        clean_counts = dict()
        for key, value in counts.items():
            if value > 0:
                clean_counts[str(key).replace("ResType.", "")] = value
        return clean_counts

    apiout_dir = output_dir / api_name
    all_dir = apiout_dir / "all"
    os.makedirs(all_dir, exist_ok=True)
    first_dirs = {
        ResType.RANDOM: apiout_dir / "random",
        ResType.STATUS: apiout_dir / "status",
        ResType.VALUE: apiout_dir / "value",
        ResType.REV_FWD_GRAD: apiout_dir / "grad-rev-fwd",
        ResType.ND_GRAD: apiout_dir / "grad-nd",
        ResType.REV_STATUS: apiout_dir / "status-rev",
        ResType.REV_VALUE: apiout_dir / "value-rev",
        ResType.FWD_STATUS: apiout_dir / "status-fwd",
        ResType.FWD_VALUE: apiout_dir / "value-fwd",
        ResType.PASS: apiout_dir / "pass",
        # ResType.SKIP: apiout_dir / "skip",
        ResType.DIRECT_CRASH: apiout_dir / "crash-direct",
        ResType.REV_CRASH: apiout_dir / "crash-rev",
        ResType.FWD_CRASH: apiout_dir / "crash-fwd",
        ResType.ND_CRASH: apiout_dir / "crash-nd",
        ResType.NAN: apiout_dir / "nan",
    }

    second_out_dir = apiout_dir / "grad"
    os.makedirs(second_out_dir, exist_ok=True)
    second_dirs = {
        ResType.RANDOM: second_out_dir / "random",
        ResType.STATUS: second_out_dir / "status",
        ResType.VALUE: second_out_dir / "value",
        ResType.REV_FWD_GRAD: second_out_dir / "grad-rev-fwd",
        ResType.ND_GRAD: second_out_dir / "grad-nd",
        ResType.REV_STATUS: second_out_dir / "status-rev",
        ResType.REV_VALUE: second_out_dir / "value-rev",
        ResType.FWD_STATUS: second_out_dir / "status-fwd",
        ResType.FWD_VALUE: second_out_dir / "value-fwd",
        ResType.PASS: second_out_dir / "pass",
        ResType.SKIP: second_out_dir / "skip",
        ResType.DIRECT_CRASH: second_out_dir / "crash-direct",
        ResType.REV_CRASH: second_out_dir / "crash-rev",
        ResType.FWD_CRASH: second_out_dir / "crash-fwd",
        ResType.ND_CRASH: second_out_dir / "crash-nd",
        ResType.NAN: second_out_dir / "nan",
    }

    api = JaxAPI(api_name)
    records = JaxDatabase.get_all_records(api_name)

    first_counts = {t: 0 for t in ResType}
    second_counts = {t: 0 for t in ResType}

    for k in range(num):
        if mutate:
            api.new_record(choice(records))
            api.mutate()
        else:
            if k < len(records):
                api.new_record(records[k])
            else:
                break

        first_ret = testrun(api, first_dirs, all_dir, output_dir, first_counts)
        if first_ret == (ResType.PASS, ResType.PASS):
            testrun(api, second_dirs, None, output_dir, second_counts, True)

    first_clean_counts = get_clean_counts(first_counts)
    second_clean_counts = get_clean_counts(second_counts)
    print(first_clean_counts)
    print(second_clean_counts)

    log_file = output_dir / "log.txt"
    dump_data(
        f"{api_name}\n{first_clean_counts}\n{second_clean_counts}\n",
        log_file,
        "a",
    )

    log_csv_file = output_dir / "log.csv"
    first_str_list = [str(i) for i in first_counts.values()]
    second_str_list = [str(i) for i in second_counts.values()]
    dump_data(
        f"{api_name}, {', '.join(first_str_list)}, {', '.join(second_str_list)}\n",
        log_csv_file,
        "a",
    )


def testrun(
    api: JaxAPI, dirs, all_dir, output_dir, counts: dict, use_grad=False
):
    def get_log_code():
        log_code = "import jax\n"
        log_code += "jax.config.update('jax_enable_x64', True)\n"
        log_code += fn_code
        log_code += inv_code
        log_code += str(input_list)
        fn_output = "fn("
        if len(input_list) != 0:            
	        for arg in input_list:
	            fn_output += arg
	            fn_output += ','
	        fn_output = fn_output[:-1]
        fn_output += ')'
        log_code += '\n'
        log_code += fn_output
        return log_code

    def log(restype: ResType):
        log_code = get_log_code()
        if restype in dirs.keys():
            out_dir = dirs[restype]
            os.makedirs(out_dir, exist_ok=True)
            JaxLibrary.write_to_dir(out_dir, log_code)
        counts[restype] += 1
        if all_dir is not None:
            JaxLibrary.write_to_dir(all_dir, log_code)

    fn_code, inv_code, input_list = api.to_differential_fn_code(
        limit_max_value=128
    )
    dump_data(get_log_code(), output_dir / "temp.py")
    if len(input_list):
        exec(fn_code)
        exec(inv_code)
        inputs_str = f"({', '.join(input_list)},)"
        if use_grad:
            ret = eval(f"test(Grad(fn, {inputs_str}), {inputs_str})")
        else:
            ret = eval(f"test(fn, {inputs_str})")
    else:
        ret = ResType.SKIP

    if isinstance(ret, tuple):
        # Merge the restype
        if ResType.REV_STATUS in ret and ResType.FWD_STATUS in ret:
            ret = ResType.STATUS
            log(ret)
        elif ResType.REV_STATUS in ret and ResType.FWD_STATUS in ret:
            ret = ResType.VALUE
            log(ret)
        elif ret[0] == ret[1]:
            log(ret[0])
        else:
            for t in ret:
                if t != ResType.SKIP:
                    log(t)
    else:
        log(ret)
    return ret

def testAPI_LLM(
    api_name,
    num=1000,
    output_dir: Path = Path("../output-ad/jax"),
    mutate=True,
):
    def get_clean_counts(counts):
        clean_counts = dict()
        for key, value in counts.items():
            if value > 0:
                clean_counts[str(key).replace("ResType.", "")] = value
        return clean_counts

    apiout_dir = output_dir / api_name
    all_dir = apiout_dir / "all"
    os.makedirs(all_dir, exist_ok=True)
    first_dirs = {
        ResType.RANDOM: apiout_dir / "random",
        ResType.STATUS: apiout_dir / "status",
        ResType.VALUE: apiout_dir / "value",
        ResType.REV_FWD_GRAD: apiout_dir / "grad-rev-fwd",
        ResType.ND_GRAD: apiout_dir / "grad-nd",
        ResType.REV_STATUS: apiout_dir / "status-rev",
        ResType.REV_VALUE: apiout_dir / "value-rev",
        ResType.FWD_STATUS: apiout_dir / "status-fwd",
        ResType.FWD_VALUE: apiout_dir / "value-fwd",
        ResType.PASS: apiout_dir / "pass",
        # ResType.SKIP: apiout_dir / "skip",
        ResType.DIRECT_CRASH: apiout_dir / "crash-direct",
        ResType.REV_CRASH: apiout_dir / "crash-rev",
        ResType.FWD_CRASH: apiout_dir / "crash-fwd",
        ResType.ND_CRASH: apiout_dir / "crash-nd",
        ResType.NAN: apiout_dir / "nan",
    }

    second_out_dir = apiout_dir / "grad"
    os.makedirs(second_out_dir, exist_ok=True)
    second_dirs = {
        ResType.RANDOM: second_out_dir / "random",
        ResType.STATUS: second_out_dir / "status",
        ResType.VALUE: second_out_dir / "value",
        ResType.REV_FWD_GRAD: second_out_dir / "grad-rev-fwd",
        ResType.ND_GRAD: second_out_dir / "grad-nd",
        ResType.REV_STATUS: second_out_dir / "status-rev",
        ResType.REV_VALUE: second_out_dir / "value-rev",
        ResType.FWD_STATUS: second_out_dir / "status-fwd",
        ResType.FWD_VALUE: second_out_dir / "value-fwd",
        ResType.PASS: second_out_dir / "pass",
        ResType.SKIP: second_out_dir / "skip",
        ResType.DIRECT_CRASH: second_out_dir / "crash-direct",
        ResType.REV_CRASH: second_out_dir / "crash-rev",
        ResType.FWD_CRASH: second_out_dir / "crash-fwd",
        ResType.ND_CRASH: second_out_dir / "crash-nd",
        ResType.NAN: second_out_dir / "nan",
    }

    api = JaxAPI(api_name)
    records = JaxDatabase.get_all_records(api_name)

    first_counts = {t: 0 for t in ResType}
    second_counts = {t: 0 for t in ResType}

    for k in range(num):
        if mutate:
            api.new_record(choice(records))
            api.mutate()
        else:
            if k < len(records):
                api.new_record(records[k])
            else:
                break

        first_ret = testrun_LLM(api, first_dirs, all_dir, output_dir, first_counts)
        if first_ret == (ResType.PASS, ResType.PASS):
            testrun_LLM(api, second_dirs, None, output_dir, second_counts, True)

    first_clean_counts = get_clean_counts(first_counts)
    second_clean_counts = get_clean_counts(second_counts)
    print(first_clean_counts)
    print(second_clean_counts)

    log_file = output_dir / "log.txt"
    dump_data(
        f"{api_name}\n{first_clean_counts}\n{second_clean_counts}\n",
        log_file,
        "a",
    )

    log_csv_file = output_dir / "log.csv"
    first_str_list = [str(i) for i in first_counts.values()]
    second_str_list = [str(i) for i in second_counts.values()]
    dump_data(
        f"{api_name}, {', '.join(first_str_list)}, {', '.join(second_str_list)}\n",
        log_csv_file,
        "a",
    )

def testrun_LLM(
    api: JaxAPI, dirs, all_dir, output_dir, counts: dict, use_grad=False
):
    def get_log_code():
        log_code = "import jax\n"
        log_code += "jax.config.update('jax_enable_x64', True)\n"
        log_code += fn_code
        log_code += inv_code
        log_code += str(input_list)
        fn_output = "fn("
        if len(input_list) != 0:            
	        for arg in input_list:
	            fn_output += arg
	            fn_output += ','
	        fn_output = fn_output[:-1]
        fn_output += ')'
        log_code += '\n'
        log_code += fn_output
        return log_code

    def log(restype: ResType):
        log_code = get_log_code()
        if restype in dirs.keys():
            out_dir = dirs[restype]
            os.makedirs(out_dir, exist_ok=True)
            JaxLibrary.write_to_dir(out_dir, log_code)
        counts[restype] += 1
        if all_dir is not None:
            JaxLibrary.write_to_dir(all_dir, log_code)

    fn_code, inv_code, input_list = api.to_differential_fn_code(
        limit_max_value=128
    )
    target_api = api.api

    coverage_stats = analyze_coverage()
    stats = find_low_coverage_files(coverage_stats)
    files,line_fileNames,line_target = prioritize_files(stats)
    code_context = "\n".join(f"Line {ln}: {code}" for ln, code in line_target[:20])
    file_name = files[0][0]
    ## 显式标记已处理文件
    update_processed_files(file_name,before_coverage)
    
    prompt = gen_prompt(file_name,target_api,code_context)

    max_patience = 5
    num_limit_reached = 0
    codes = ''
    wait_seconds = 0.0
    tcosts = []
    ret = {}
    if not os.path.exists(output_dir / target_api):
        os.makedirs(output_dir / target_api, exist_ok=True)

    if wait_seconds > 0.0:
        time.sleep(wait_seconds)
    try:
        codes, tc = gen(
            'gpt-4o-mini',
            prompt,
            0.9,
            4096,
            25,
            0.9,
            0.7,
            0.3,
        )
        tcosts.append(tc)
        ret[target_api] = {"code": codes, "g_time": tc}
    except Exception as e:
        if wait_seconds < 3.0:
            if num_limit_reached >= max_patience:
                wait_seconds += 0.5
                num_limit_reached = 0
            time.sleep(30.0)
        else:
            time.sleep(30.0)
    with open(output_dir / "outputs.json", "w") as f:
        json.dump(ret, f)
    if tcosts:
        avg_cost = np.average(tcosts)
    else:
        avg_cost = 0.0  # 或者设置为一个默认值，根据你的应用决定
    with open(output_dir / "log.txt", "w") as f:
        f.write(
            "{avg_cost}\n\nAverage cost in seconds for each Codex API call"
        )
    #生成新代码
    fn_code, inv_code, input_list_ = extract_sections(codes)
    input_list = convert_str_list(input_list_)
    
    fn_code += '\n'
    inv_code += '\n'
            
    dump_data(get_log_code(), output_dir / "temp.py")
    if len(input_list):
        exec(fn_code)
        exec(inv_code)
        inputs_str = f"({', '.join(input_list)},)"
        if use_grad:
            ret = eval(f"test(Grad(fn, {inputs_str}), {inputs_str})")
        else:
            ret = eval(f"test(fn, {inputs_str})")
    else:
        ret = ResType.SKIP

    if isinstance(ret, tuple):
        # Merge the restype
        if ResType.REV_STATUS in ret and ResType.FWD_STATUS in ret:
            ret = ResType.STATUS
            log(ret)
        elif ResType.REV_STATUS in ret and ResType.FWD_STATUS in ret:
            ret = ResType.VALUE
            log(ret)
        elif ret[0] == ret[1]:
            log(ret[0])
        else:
            for t in ret:
                if t != ResType.SKIP:
                    log(t)
    else:
        log(ret)
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jax Autodiff Unit Test")
    parser.add_argument(
        "--api",
        type=str,
        help="The name of API to be test, e.g., jax.numpy.sum",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="The number of mutants for each API (default: 1000)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="../output-ad/jax",
        help="The output dir",
    )
    args = parser.parse_args()
    if not os.path.exists(DECAY_FILE):
        with open(DECAY_FILE, "w") as f:
            json.dump({}, f, indent=4)  # 创建空 JSON 文件    


    testAPI(args.api, args.num, Path(args.dir))
    code_status = 0
    if os.path.exists('coverage_Python_report_stage.csv'):
       code_status = compare_last_two_percentages('coverage_Python_report_stage.csv')
    if(code_status == 1):
       testAPI_LLM(args.api, args.num, Path(args.dir))

