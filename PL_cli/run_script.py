import os
import subprocess

def run_script(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # need to cd into the directory of the script to run it
    script_dir = os.path.dirname(os.path.abspath(file_path))
    os.chdir(script_dir)
    new_file_path = os.path.basename(file_path)

    # check last 3 chars of file_path to determine how to run the script
    if file_path[-3:] == ".py":
        result = subprocess.run(["python", new_file_path], capture_output=True)
    elif file_path[-3:] == ".sh":
        result = subprocess.run(["bash", new_file_path], capture_output=True)
    elif file_path[-4:] == ".bat":
        result = subprocess.run([new_file_path], capture_output=True, shell=True)
    else:
        raise Exception(f"{new_file_path} is an unknown type. Expected .py, .sh, or .bat")
    
    if result.returncode != 0:
        raise Exception(f"{result.stderr.decode()}")
    print(result.stdout.decode())

    # return to the original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    