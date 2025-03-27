import argparse
import subprocess
import os
import shutil

def submit_script(script, args, debug_id=None):
    if args.debug:
        print(f"[DEBUG] {script}")
        return debug_id
    else:
        with open("s.sh", "w") as script_file:
            script_file.write(script.lstrip("\n"))
        submit_command = ["sbatch", "s.sh"]
        
        extract_process = subprocess.run(submit_command, stdout=subprocess.PIPE, text=True, check=True)
        return extract_process.stdout.strip().split()[-1] # get SLURM job ID 