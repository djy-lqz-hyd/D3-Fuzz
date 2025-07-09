import argparse
from pathlib import Path
import subprocess as sp
import os
import shutil
import time
from classes.database import JaxDatabase
from utils.printer import dump_data
from utils.loader import load_data
import coverage
import json
import re
import pandas as pd
JaxDatabase.database_config("127.0.0.1", 27017, "jax-test-unique")

def parse_lcov_summary():
    # 运行 lcov --summary 获取覆盖率报告
#    result = sp.run(
#        ["lcov","--gcov-tool", "gcov-11", "--summary", "coverage.info"],
#        capture_output=True,
#        text=True,
#        check=True
#    )
    reresult = ''
    if os.path.getsize("coverage.info") == 0:
        print("coverage.info 文件为空，跳过 lcov summary")
        output = "No coverage data available."
    else:
        result = sp.run(
            ["lcov", "--gcov-tool", "gcov-11", "--summary", "coverage.info"],
            capture_output=True,
            text=True,
            check=True
        )
    output = result.stdout

    # 正则表达式提取关键数据
    line_pattern = r"lines.*?: ([\d.]+%) \((\d+) of (\d+) lines\)"
    func_pattern = r"functions.*?: ([\d.]+%) \((\d+) of (\d+) functions\)"

    line_match = re.search(line_pattern, output)
    func_match = re.search(func_pattern, output)

    # 组织数据结构
    return {
        "lines": f"{line_match.group(1)} ({line_match.group(2)}/{line_match.group(3)})",
        "functions": f"{func_match.group(1)} ({func_match.group(2)}/{func_match.group(3)})"
    }


def load_log_and_put(output_dir: Path, target_dir: Path, api_name):
    data = load_data(output_dir / "temp.py")
    dump_data(data, target_dir / f"{api_name}.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NablaFuzz for Jax")

    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="The number of mutants for each API or API pair (default: 1000)",
    )
    parser.add_argument(
        "--max_api",
        type=int,
        default=-1,
        help="The number of API that will be tested (default: -1, which means all API)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the output dir (default: False)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output-ad",
        help="The output directory (default: 'output-ad')",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device (default: 'cpu')"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="The suffix of the output dir (default: '')",
    )

    args = parser.parse_args()
    print(args)

    if args.suffix != "":
        suffix = f"-{args.suffix}"
    else:
        suffix = ""

    output_dir = Path("..", args.output, "jax", f"union{suffix}")
    if args.clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    timeout_dir = output_dir / "run-timeout"
    crash_dir = output_dir / "run-crash"
    os.makedirs(timeout_dir, exist_ok=True)
    os.makedirs(crash_dir, exist_ok=True)

    # TEST
    timeout = 600
    max_api_number = args.max_api

    time_file = output_dir / "time.csv"
    api_list = JaxDatabase.get_api_list()

    i = 0
    for api_name in api_list:
        if api_name in [
            "jax._src.custom_derivatives.custom_jvp",
            "jax._src.custom_derivatives.custom_vjp",
        ]:
            continue

        i += 1
        if max_api_number != -1 and i > max_api_number:
            break
        print(f"{api_name}, {i} / {len(api_list)}")
        st_time = time.time()
        try:
            ret = sp.run(
                [
                    "python",
                    "jax_adtest.py",
                    "--api",
                    api_name,
                    "--num",
                    str(args.num),
                    "--dir",
                    output_dir,
                ],
                timeout=timeout,
                shell=False,
                capture_output=True,
            )
        except sp.TimeoutExpired:
            dump_data(
                f"{api_name}\n", output_dir / "test-run-timeout.txt", mode="a"
            )
            load_log_and_put(output_dir, timeout_dir, api_name)
            print("TIMEOUT\n")
        else:
            if ret.returncode != 0:
                dump_data(
                    f"{api_name}\n",
                    output_dir / "test-run-crash.txt",
                    mode="a",
                )
                error_msg = ret.stdout.decode("utf-8") + ret.stderr.decode(
                    "utf-8"
                )
                print(error_msg)
                dump_data(
                    f"{api_name}\n{error_msg}\n\n",
                    output_dir / "crash.log",
                    "a",
                )
                print("ERROR\n")
                load_log_and_put(output_dir, crash_dir, api_name)
            else:
                print(ret.stdout.decode("utf-8"))
        running_time = time.time() - st_time
        dump_data(f"{api_name}, {running_time}\n", time_file, "a")
        if (i % 1 == 0):
            #实时输出python覆盖率
            sp.run(["coverage", "json", "-o", "coverage.json"], check=True)
            with open("coverage.json", "r") as f:
                data = json.load(f)
                total_coverage = data['totals']['percent_covered']
                print(f"Total coverage: {total_coverage:.1f}%")
                coverage_data = pd.DataFrame({'api_num': [i],'Total Coverage': [f"{total_coverage:.1f}%"]})
                coverage_data.to_csv('coverage_Python_report_stage.csv', mode='a', header=False, index=False)
#        if (i % 100 == 0 or i == 2 or i == 1094):      
#            #实时输出C++覆盖率
#            sp.run(["lcov", "--capture", "--directory", "/home/jax/bazel-out/k8-opt/bin/jaxlib/", "--output-file", "coverage.info"])
#            coverage_C_data = parse_lcov_summary()
#            coverage_CSV_data = pd.DataFrame({'api_num': [i],'lines':coverage_C_data["lines"],'functions':coverage_C_data["functions"]})
#            coverage_CSV_data.to_csv('coverage_C++_report_stage.csv', mode='a', header=False, index=False)
#        
