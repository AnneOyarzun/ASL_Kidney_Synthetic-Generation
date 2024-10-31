import subprocess
import os


def calculate_fid(real_path, fake_path):
    # Ensure the paths are absolute
    real_path = os.path.abspath(real_path)
    fake_path = os.path.abspath(fake_path)

    # Construct the command to execute
    command = f"python -m pytorch_fid {real_path} {fake_path}"

    # Execute the command
    os.system(command)
