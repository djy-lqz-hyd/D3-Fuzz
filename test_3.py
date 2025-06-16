import json
import re


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




#解析覆盖率数据
def analyze_coverage():
    with open('coverage.json') as f:
        data = json.load(f)

    # 计算各文件覆盖率
    coverage_stats = {}
    for file, info in data['files'].items():
        if 'torch' not in file:  # 仅关注PyTorch源码
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



#提取关键未覆盖代码
def get_critical_lines(file_path, missing_lines):
    """多维度关键代码行识别"""
    critical = []
    line_total = []
    line_fileName = []
    line_target = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line_no in missing_lines:
        line = lines[line_no - 1].strip()

        # 1. 基础控制流检测
        control_flow_keywords = {'if', 'elif', 'else', 'for', 'while',
                                 'try', 'except', 'finally', 'raise',
                                 'assert', 'return', 'yield'}
        if any(kw in line for kw in control_flow_keywords):
            critical.append((file_path,line_no, line, 'control_flow'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        # 2. 数学核心运算检测
        math_ops = re.compile(r'(torch\.(_?[a-z]+_ops|native))|(ATen\/)')
        if math_ops.search(line):
            critical.append((file_path,line_no, line, 'math_kernel'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        # 3. CUDA相关代码检测
        cuda_patterns = [
            r'\.cuda\(', r'to\(["\']cuda',
            r'CUDAGuard\(', r'cudaStream'
        ]
        if any(re.search(p, line) for p in cuda_patterns):
            critical.append((file_path,line_no, line, 'cuda_related'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        # 4. 内存管理关键代码
        memory_ops = {'malloc', 'free', 'resize_', 'storage',
                      'untyped_storage', 'data_ptr'}
        if any(op in line for op in memory_ops):
            critical.append((file_path,line_no, line, 'memory_ops'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        # 5. 虚函数/接口实现检测
        if re.search(r'\bvirtual\b.*\boverride\b', line):
            critical.append((file_path,line_no, line, 'virtual_function'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue

        # 6. 性能关键循环
        loop_pattern = r'(for|while)\s*\(.*\)\s*{'
        if re.search(loop_pattern, line) and '// PERF:' in line:
            critical.append((file_path,line_no, line, 'performance_loop'))
            line_total.append(line_no)
            line_target.append((line_no,line))
            continue
            
    line_fileName.append((file_path,line_total))
    return critical,line_fileName,line_target



def prioritize_files(low_coverage):
    """基于PyTorch测试套件特征的优先级权重分配"""
    # 权重规则（根据test/目录下的测试文件分布调整）
    priority_rules = {
        # 模块类别: (路径关键词, 权重, 子模块额外权重)
        'autograd': (['autograd'], 3.5, {
            'anomaly_mode': 0.2,
            'gradcheck': 0.3
        }),
        'nn': (['nn'], 3.2, {
            'modules': 0.4,  # test_nn.py覆盖的模块
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
        # 最终得分 = (100-覆盖率) * 权重 * 关键因子
        score = (100 - data['pct']) * (base_weight + sub_weight) * critical_factor
        if(score > score_):
            score_ = score
            line_target = line_target_
            
        prioritized.append((file_path, score))  
         
    sorted_prioritized = sorted(prioritized, key=lambda x: -x[1])
    return sorted_prioritized,line_fileNames,line_target





if __name__ == "__main__":
    coverage_stats = analyze_coverage()
    stats = find_low_coverage_files(coverage_stats)
    files,line_fileNames,line_target = prioritize_files(stats)
    code_context = "\n".join(f"Line {ln}: {code}" for ln, code in line_target[:20])
    print(code_context)
    #print(files)
    #print(line_fileNames)
    # 获取并转换数字
    numbers = find_numbers(line_fileNames, files[0][0])
    result = ",".join(map(str, numbers))
    print(files[0][0])
    print(files[0][1])
    print(result)
    #get_critical_lines()