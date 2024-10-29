import subprocess

def load_data():
    url = "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betas_norm_modela.mat"
    target_directory = "utils_ex7/data_ctx"
    
    # Run the wget command using subprocess
    subprocess.run(["wget", "-P", target_directory, url])