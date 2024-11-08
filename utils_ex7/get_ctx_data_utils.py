import subprocess

def get_ctx_data():
    urls = ["https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betas_norm_modela.mat", 
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betas_norm_modelb.mat", 
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betas_norm_modelc.mat", 
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betas_norm_modeld.mat", 
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat1_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat1_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat1_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat1_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat2_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat2_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat2_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bmat2_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname1_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname1_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname1_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname1_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname2_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname2_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname2_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/bname2_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/pcaResp_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/pcaResp_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/pcaResp_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/pcaResp_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/per_var_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/per_var_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/per_var_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/per_var_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Regmod_units_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Regmod_units_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Regmod_units_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Regmod_units_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modela.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modelb.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modelc.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/Rtot_units_modeld.mat",
            "https://github.com/ManteLab/Iton_notebooks_public/raw/refs/heads/main/utils_ex7/data_ctx/task_conditions.mat"]
    
    for url in urls:
        target_directory = "utils_ex7/data_ctx"
        #print('Loading:', url)
        # Run the wget command using subprocess
        subprocess.run(["wget", "-P", target_directory, url])

    print('Data loaded.')

    