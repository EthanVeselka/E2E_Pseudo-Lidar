import os
import subprocess

def run_script(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File {file_path} not found.")
    
    result = subprocess.run(["python", file_path], capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Error: {result.stderr.decode()}")
    print(result.stdout.decode())
    
    