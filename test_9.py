import sys
import re

def fix_indentation_in_file(file_path, spaces=4):
    """
    读取指定的 Python 文件，仅在每行开头替换 Tab 为指定数量的空格，
    从而解决因混用 Tab 和空格导致的缩进问题。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        sys.exit(1)

    fixed_lines = []
    for line in lines:
        # 只处理行首缩进部分，防止修改字符串中的 Tab 字符
        match = re.match(r'^(\s*)', line)
        if match:
            leading_whitespace = match.group(1)
            # 将所有 Tab 替换为指定数量的空格
            fixed_whitespace = leading_whitespace.replace('\t', ' ' * spaces)
            fixed_lines.append(fixed_whitespace + line[len(leading_whitespace):])
        else:
            fixed_lines.append(line)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(fixed_lines))
        print(f"文件 '{file_path}' 的缩进已成功修正。")
    except IOError as e:
        print(f"写入文件 {file_path} 时出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    file_path = '/home/NablaFuzz/NablaFuzz-PyTorch-Jax/output-ad/torch/union/torch.Tensor.cos_/LLM/all/3.py'
    fix_indentation_in_file(file_path)


