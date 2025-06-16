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

