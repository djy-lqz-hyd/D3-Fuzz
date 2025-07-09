from pathlib import Path
from random import choice
import numpy as np
import pandas as pd
import os
import argparse
import torch
from openai import OpenAI
import re
import time
import json
import random
import subprocess as sp

from helper_torch import (
    DirectInv,
    RevInv,
    FwdInv,
    NDCheck,
    Grad,
    allow_error,
    is_crash,
)
from classes.torch_library import TorchLibrary
from classes.torch_api import TorchAPI, TorchArgument, Argument
from classes.database import TorchDatabase
from constant.returntypes import ResType
from utils.printer import dump_data

TorchDatabase.database_config("127.0.0.1", 27017, "pytorch")


processed_files = set()
before_coverage = 0
DECAY_FILE = "decay_factors.json"
def load_decay_factors():
    if os.path.exists(DECAY_FILE):
        with open(DECAY_FILE, "r") as f:
            return json.load(f)
    return {} 
   
def update_decay_factor(decay_factors):
    with open(DECAY_FILE, "w") as f:
        json.dump(decay_factors, f, indent=4) 
        

def update_processed_files(file_name,before_coverage):
    code_status = 0
    if os.path.exists('coverage_Python_report_stage.csv'):
       code_status = compare_last_two_percentages('coverage_Python_report_stage.csv')
    if(code_status == 1):
        processed_files.add(file_name)

def compare_last_two_percentages(file_path):
   
    df = pd.read_csv(file_path, sep='\t', header=None)
    if len(df) < 2:
        print("错误：文件需要至少包含两行数据")
        return


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



def analyze_coverage():
    with open('coverage.json') as f:
        data = json.load(f)


    coverage_stats = {}
    for file, info in data['files'].items():
        if 'torch' not in file:  
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


def find_low_coverage_files(stats, threshold=70):
    return {
        f: data for f, data in stats.items()
        if data['pct'] < threshold
    }




def get_critical_lines(file_path, missing_lines):
    critical = []
    line_total = []
    line_fileName = []
    line_target = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line_no in missing_lines:
        line = lines[line_no - 1].strip()

     
        control_flow_keywords = {'if', 'elif', 'else', 'for', 'while',
                                 'try', 'except', 'finally', 'raise',
                                 'assert', 'return', 'yield'}
        if any(kw in line for kw in control_flow_keywords):
            critical.append((file_path,line_no, line, 'control_flow'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        math_ops = re.compile(r'(torch\.(_?[a-z]+_ops|native))|(ATen\/)')
        if math_ops.search(line):
            critical.append((file_path,line_no, line, 'math_kernel'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue


        cuda_patterns = [
            r'\.cuda\(', r'to\(["\']cuda',
            r'CUDAGuard\(', r'cudaStream'
        ]
        if any(re.search(p, line) for p in cuda_patterns):
            critical.append((file_path,line_no, line, 'cuda_related'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue
        memory_ops = {'malloc', 'free', 'resize_', 'storage',
                      'untyped_storage', 'data_ptr'}
        if any(op in line for op in memory_ops):
            critical.append((file_path,line_no, line, 'memory_ops'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

    
        if re.search(r'\bvirtual\b.*\boverride\b', line):
            critical.append((file_path,line_no, line, 'virtual_function'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        loop_pattern = r'(for|while)\s*\(.*\)\s*{'
        if re.search(loop_pattern, line) and '// PERF:' in line:
            critical.append((file_path,line_no, line, 'performance_loop'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue
            
    line_fileName.append((file_path,line_total))
    return critical,line_fileName,line_target



def prioritize_files(low_coverage, top_k=5, shuffle_window=3):

    priority_rules = {
        
        'autograd': (['autograd'], 3.5, {
            'anomaly_mode': 0.2,
            'gradcheck': 0.3
        }),
        'nn': (['nn'], 3.2, {
            'modules': 0.4, 
            'functional': 0.4,
            'parallel': 0.2
        }),
        'distributed': (['distributed'], 3.0, {
            'rpc': 0.3,
            'c10d': 0.3
        }),
        'cuda': (['cuda'], 2.8, {
            'amp': 0.3,
            'nccl': 0.2
        }),
        'jit': (['jit', 'torchscript'], 2.5, {
            'frontend': 0.3,
            'ir': 0.2
        }),
        'ops': (['ops'], 2.3, {}),
        'core': (['_tensor', 'serialization'], 2.0, {})
    }

    prioritized = []
    line_fileNames = []
    critical_lines_all = []
    line_target = []
    score_ = 0
    
    for file_path, data in low_coverage.items():

        base_weight = 1.0
        sub_weight = 0.0
        for category, (keywords, weight, submodules) in priority_rules.items():
            if any(kw in file_path for kw in keywords):
                base_weight = weight
                for sub_key, sub_val in submodules.items():
                    if sub_key in file_path:
                        sub_weight += sub_val
                break


        critical_lines,line_fileName,line_target_ = get_critical_lines(file_path, data['missing'])
        #critical_factor = 1 + 0.1 * len(critical_lines)  
        decay_factors = load_decay_factors()
        score = (100 - data['pct']) * (base_weight + sub_weight) * critical_factor * decay
        if(score > score_):
            score_ = score
            line_target = line_target_
         
    sorted_prioritized = sorted(prioritized, key=lambda x: -x[1])

    top_candidates = sorted_prioritized[:top_k*2]  
    

    if len(top_candidates) >= shuffle_window:
        shuffle_part = list(top_candidates[:shuffle_window])
        random.shuffle(shuffle_part)
        top_candidates = shuffle_part + top_candidates[shuffle_window:]
    

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


def convert_str_list(string):
  
    first_line = string.splitlines()[0]
    #
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
    - 文件(参考pytorch源码文件)：{file_name}
    - 关键未覆盖行（部分）：{code_context}
    - 目标API：{target_api}
    # TEST CASE FOR: {{file_path}}
    import torch
    torch.manual_seed({{seed}})
    def get_fn():
        def fn({{input_vars}}):
            """ '测试函数核心逻辑' """
            # [在此添加核心测试逻辑]
            # 必须调用目标代码段的API
            {targetAPI}
            # [在此添加断言/异常触发逻辑]  
	    #在此处梯度计算测试（可选）
            return {{result_vars}}
        return fn
    fn = get_fn()
    input_list = [{input_var}]
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
        api_key="key",
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
        presence_penalty=presence_penalty, 
        n=1,  
        stop=None,  
    )
    codes = completion.choices[0].message.content
    t_end = time.time()
    return codes, t_end - t_start

def test(fn, inputs):
    inputs = tuple(inputs)

    direct_status, direct_value, direct_err = DirectInv(fn, inputs)
    if is_crash(direct_err):
        return ResType.DIRECT_CRASH

    for _ in range(9):
        direct_status_, direct_value_, direct_err_ = DirectInv(fn, inputs)
        if direct_status != direct_status_ or not TorchLibrary.is_equal(
            direct_value, direct_value_, equal_nan=True
        ):
            return ResType.RANDOM
        elif not TorchLibrary.is_equal(direct_value, direct_value_):
            return ResType.NAN

    rev_status, rev_value, rev_grad, rev_err = RevInv(fn, inputs)
    fwd_status, fwd_value, fwd_grad, fwd_err = FwdInv(fn, inputs)
    rev_restype = ResType.PASS
    fwd_restype = ResType.PASS

    if rev_status != direct_status and not allow_error(rev_err):
        rev_restype = ResType.REV_STATUS
    elif is_crash(rev_err):
        rev_restype = ResType.REV_CRASH
    elif rev_status == "fail":
        # print(rev_err)
        rev_restype = ResType.SKIP
    elif not TorchLibrary.is_equal(direct_value, rev_value):
        rev_restype = ResType.REV_VALUE

    if fwd_status != direct_status and not allow_error(fwd_err):
        fwd_restype = ResType.FWD_STATUS
    elif is_crash(fwd_err):
        fwd_restype = ResType.FWD_CRASH
    elif fwd_status == "fail":
        fwd_restype = ResType.SKIP
    elif not TorchLibrary.is_equal(direct_value, fwd_value):
        fwd_restype = ResType.FWD_VALUE

    if (
        rev_restype == ResType.PASS
        and fwd_restype == ResType.PASS
        and not TorchLibrary.is_equal(rev_grad, fwd_grad)
    ):
        # print(rev_grad)
        # print(fwd_grad)
        return ResType.REV_FWD_GRAD

    if rev_restype == ResType.PASS:
        nd_status, nd_err = NDCheck(fn, inputs, "rev")
        if is_crash(nd_err):
            rev_restype = ResType.ND_CRASH
        elif nd_status == "fail":
            if allow_error(nd_err):
                rev_restype = ResType.SKIP
            elif "Jacobian" in nd_err:
                rev_restype = ResType.ND_GRAD
            else:
                rev_restype = ResType.ND_FAIL
    if fwd_restype == ResType.PASS:
        nd_status, nd_err = NDCheck(fn, inputs, "fwd")
        if is_crash(nd_err):
            fwd_restype = ResType.ND_CRASH
        elif nd_status == "fail":
            if allow_error(nd_err):
                fwd_restype = ResType.SKIP
            elif "Jacobian" in nd_err:
                fwd_restype = ResType.ND_GRAD
            else:
                fwd_restype = ResType.ND_FAIL
    return (rev_restype, fwd_restype)


def testAPI(
    api_name,
    num=1000,
    output_dir: Path = Path("../output-ad/torch"),
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
        ResType.CRASH: apiout_dir / "crash",
        ResType.DIRECT_CRASH: apiout_dir / "crash-direct",
        ResType.REV_CRASH: apiout_dir / "crash-rev",
        ResType.FWD_CRASH: apiout_dir / "crash-fwd",
        ResType.ND_CRASH: apiout_dir / "crash-nd",
        ResType.NAN: apiout_dir / "nan",
        ResType.ND_FAIL: apiout_dir / "nd-fail",
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
        ResType.CRASH: second_out_dir / "crash",
        ResType.DIRECT_CRASH: second_out_dir / "crash-direct",
        ResType.REV_CRASH: second_out_dir / "crash-rev",
        ResType.FWD_CRASH: second_out_dir / "crash-fwd",
        ResType.ND_CRASH: second_out_dir / "crash-nd",
        ResType.NAN: second_out_dir / "nan",
        ResType.ND_FAIL: second_out_dir / "nd-fail",
    }

    # set the tensor_size_limit as 1024 to reduce memory and time cost
    TorchArgument._tensor_size_limit = 1024
    # gradcheck should avoid large number, which can cause false positive
    temp_values = []
    for v in Argument._float_values:
        if abs(v) < 1024:
            temp_values.append(v)
    Argument._float_values = temp_values

    api = TorchAPI(api_name)
    records = TorchDatabase.get_all_records(api_name)

    first_counts = {t: 0 for t in ResType}
    second_counts = {t: 0 for t in ResType}

    for k in range(num):
        if mutate:
            api.get_invocation(choice(records))
            api.mutate()
        else:
            if k < len(records):
                api.get_invocation(records[k])
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
    api: TorchAPI, dirs, all_dir, output_dir, counts: dict, use_grad=False
):
    def get_log_code():
        log_code = "import torch\n"
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
            TorchLibrary.write_to_dir(out_dir, log_code)
        counts[restype] += 1
        if all_dir is not None:
            TorchLibrary.write_to_dir(all_dir, log_code)

    fn_code, inv_code, input_list = api.to_differential_fn_code()
    dump_data(get_log_code(), output_dir / "temp.py")
    if len(input_list):
        try:
            exec(fn_code)
            exec(inv_code)
        except Exception:
            ret = ResType.SKIP
        else:
            try:
                inputs_str = f"({', '.join(input_list)},)"
                if use_grad:
                    ret = eval(f"test(Grad(fn, {inputs_str}), {inputs_str})")
                else:
                    ret = eval(f"test(fn, {inputs_str})")
            except Exception:
                ret = ResType.CRASH
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
    output_dir: Path = Path("../output-ad/torch"),
    mutate=True,
):
    def get_clean_counts(counts):
        clean_counts = dict()
        for key, value in counts.items():
            if value > 0:
                clean_counts[str(key).replace("ResType.", "")] = value
        return clean_counts

    apiout_dir = output_dir / api_name / "LLM"
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
        ResType.CRASH: apiout_dir / "crash",
        ResType.DIRECT_CRASH: apiout_dir / "crash-direct",
        ResType.REV_CRASH: apiout_dir / "crash-rev",
        ResType.FWD_CRASH: apiout_dir / "crash-fwd",
        ResType.ND_CRASH: apiout_dir / "crash-nd",
        ResType.NAN: apiout_dir / "nan",
        ResType.ND_FAIL: apiout_dir / "nd-fail",
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
        ResType.CRASH: second_out_dir / "crash",
        ResType.DIRECT_CRASH: second_out_dir / "crash-direct",
        ResType.REV_CRASH: second_out_dir / "crash-rev",
        ResType.FWD_CRASH: second_out_dir / "crash-fwd",
        ResType.ND_CRASH: second_out_dir / "crash-nd",
        ResType.NAN: second_out_dir / "nan",
        ResType.ND_FAIL: second_out_dir / "nd-fail",
    }

    # set the tensor_size_limit as 1024 to reduce memory and time cost
    TorchArgument._tensor_size_limit = 1024
    # gradcheck should avoid large number, which can cause false positive
    temp_values = []
    for v in Argument._float_values:
        if abs(v) < 1024:
            temp_values.append(v)
    Argument._float_values = temp_values

    api = TorchAPI(api_name)
    records = TorchDatabase.get_all_records(api_name)

    first_counts = {t: 0 for t in ResType}
    second_counts = {t: 0 for t in ResType}

    for k in range(num):
        if mutate:
            api.get_invocation(choice(records))
            api.mutate()
        else:
            if k < len(records):
                api.get_invocation(records[k])
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


#LLM大模型指导
def testrun_LLM(
    api: TorchAPI, dirs, all_dir, output_dir, counts: dict, use_grad=False
):
    def get_log_code():
        log_code = "import torch\n"
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
            TorchLibrary.write_to_dir(out_dir, log_code)
        counts[restype] += 1
        if all_dir is not None:
            TorchLibrary.write_to_dir(all_dir, log_code)

    fn_code, inv_code, input_list = api.to_differential_fn_code()
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
        try:
            exec(fn_code)
            exec(inv_code)
        except Exception:
            ret = ResType.SKIP
        else:
            try:
                inputs_str = f"({', '.join(input_list)},)"
                if use_grad:
                    ret = eval(f"test(Grad(fn, {inputs_str}), {inputs_str})")
                else:
                    ret = eval(f"test(fn, {inputs_str})")
            except Exception:
                ret = ResType.CRASH
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
    parser = argparse.ArgumentParser(description="PyTorch Autodiff Unit Test")
    parser.add_argument(
        "--api",
        type=str,
        help="The name of API to be test, e.g., torch.sum",
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
        default="../output-ad/torch",
        help="The output dir",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        default=False,
        help="Use gradgradcheck to test forward-over-rev mode (default: False)",
    )
    if not os.path.exists(DECAY_FILE):
        with open(DECAY_FILE, "w") as f:
            json.dump({}, f, indent=4)  # 创建空 JSON 文件    

    args = parser.parse_args()
    os.makedirs(Path(args.dir), exist_ok=True)

    mutate = not args.db
    testAPI(args.api, args.num, Path(args.dir), mutate)
    code_status = 0
    if os.path.exists('coverage_Python_report_stage.csv'):
       code_status = compare_last_two_percentages('coverage_Python_report_stage.csv')
    if(code_status == 1):
       testAPI_LLM(args.api, args.num, Path(args.dir), mutate)
    
