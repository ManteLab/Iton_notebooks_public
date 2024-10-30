import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.io import loadmat
from PIL import Image
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.ticker as ticker

def load_context_dependent_models(model_id):
    # common parameters
    data_variables = loadmat('utils_ex7/data_ctx/task_conditions.mat')
    task_vars = data_variables['task_conditions']

    cd_vars = {}
    cd_vars['kjcor'] = task_vars[:,0]
    cd_vars['ktdir'] = task_vars[:,1]
    cd_vars['ksdir'] = task_vars[:,2]
    cd_vars['kscol'] = task_vars[:,3]
    cd_vars['kjmot'] = task_vars[:,4]

    if model_id == 'model_a':
        betas = loadmat('utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modela.mat')
        betResp_ch_inm_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modela_v2.mat')
        betResp_inm_ch_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modela_v2.mat')
        betResp_inm_incol_ch = betas['betResp']

        data = loadmat('utils_ex7/data_ctx/Regmod_units_modela.mat') # XXw
        behavior = data['Regmod_units']

        data = loadmat('utils_ex7/data_ctx/Rtot_units_modela.mat') # XXw
        neural = data['Rtot_units']

        data = loadmat('utils_ex7/data_ctx/bmat1_modela.mat') # XXw
        bmat1 = data['bmat1']

        data = loadmat('utils_ex7/data_ctx/bmat2_modela.mat') # XXw
        bmat2 = data['bmat2']

        data = loadmat('utils_ex7/data_ctx/betas_norm_modela.mat') # XXw
        bnorm = data['betas_norm']

        data = loadmat('utils_ex7/data_ctx/pcaResp_modela.mat') # XXw
        pcaResp = data['pcaResp']

        data = loadmat('utils_ex7/data_ctx/per_var_modela.mat') # XXw
        per_var_pca = data['percent_explained']

    if model_id == 'model_b':
        betas = loadmat('utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modelb.mat')
        betResp_ch_inm_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modelb_v2.mat')
        betResp_inm_ch_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modelb_v2.mat')
        betResp_inm_incol_ch = betas['betResp']

        data = loadmat('utils_ex7/data_ctx/Regmod_units_modelb.mat') # XXw
        behavior = data['Regmod_units']

        data = loadmat('utils_ex7/data_ctx/Rtot_units_modelb.mat') # XXw
        neural = data['Rtot_units']

        data = loadmat('utils_ex7/data_ctx/bmat1_modelb.mat') # XXw
        bmat1 = data['bmat1']

        data = loadmat('utils_ex7/data_ctx/bmat2_modelb.mat') # XXw
        bmat2 = data['bmat2']

        data = loadmat('utils_ex7/data_ctx/betas_norm_modelb.mat') # XXw
        bnorm = data['betas_norm']

        data = loadmat('utils_ex7/data_ctx/pcaResp_modelb.mat') # XXw
        pcaResp = data['pcaResp']

        data = loadmat('utils_ex7/data_ctx/per_var_modelb.mat') # XXw
        per_var_pca = data['percent_explained']

    if model_id == 'model_c':
        betas = loadmat('utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modelc.mat')
        betResp_ch_inm_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modelc_v2.mat')
        betResp_inm_ch_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modelc_v2.mat')
        betResp_inm_incol_ch = betas['betResp']

        data = loadmat('utils_ex7/data_ctx/Regmod_units_modelc.mat') # XXw
        behavior = data['Regmod_units']

        data = loadmat('utils_ex7/data_ctx/Rtot_units_modelc.mat') # XXw
        neural = data['Rtot_units']

        data = loadmat('utils_ex7/data_ctx/bmat1_modelc.mat') # XXw
        bmat1 = data['bmat1']

        data = loadmat('utils_ex7/data_ctx/bmat2_modelc.mat') # XXw
        bmat2 = data['bmat2']

        data = loadmat('utils_ex7/data_ctx/betas_norm_modelc.mat') # XXw
        bnorm = data['betas_norm']

        data = loadmat('utils_ex7/data_ctx/pcaResp_modelc.mat') # XXw
        pcaResp = data['pcaResp']

        data = loadmat('utils_ex7/data_ctx/per_var_modelc.mat') # XXw
        per_var_pca = data['percent_explained']

    if model_id == 'model_d':
        betas = loadmat('utils_ex7/data_ctx/betResp_choice_inputmot_inputcol_modeld.mat')
        betResp_ch_inm_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_choice_inputcol_modeld_v2.mat')
        betResp_inm_ch_incol = betas['betResp']

        betas = loadmat('utils_ex7/data_ctx/betResp_inputmot_inputcol_choice_modeld_v2.mat')
        betResp_inm_incol_ch = betas['betResp']

        data = loadmat('utils_ex7/data_ctx/Regmod_units_modeld.mat') # XXw
        behavior = data['Regmod_units']

        data = loadmat('utils_ex7/data_ctx/Rtot_units_modeld.mat') # XXw
        neural = data['Rtot_units']

        data = loadmat('utils_ex7/data_ctx/bmat1_modeld.mat') # XXw
        bmat1 = data['bmat1']

        data = loadmat('utils_ex7/data_ctx/bmat2_modeld.mat') # XXw
        bmat2 = data['bmat2']

        data = loadmat('utils_ex7/data_ctx/betas_norm_modeld.mat') # XXw
        bnorm = data['betas_norm']

        data = loadmat('utils_ex7/data_ctx/pcaResp_modeld.mat') # XXw
        pcaResp = data['pcaResp']

        data = loadmat('utils_ex7/data_ctx/per_var_modeld.mat') # XXw
        per_var_pca = data['percent_explained']



    bname1 = ['choice','choice','motion','choice (motion)','motion (motion)','color (motion)']
    bname2 = ['motion','color' ,'color' ,'choice (color)' ,'motion (color)' ,'color (color)' ]

    units, trials, time = neural.shape

    model = {'neural': neural, 
            'regmod': behavior,
            'choice': behavior[:, :, 1], 
            'context': behavior[:, :, 4], 
            'motion_coherence': behavior[:, :, 2], 
            'color_coherence': behavior[:, :, 3]}

    choice_context = np.zeros((units, trials))
    choice_context[(model['context'] == 1) & (model['choice'] == 1)] = 1
    choice_context[(model['context'] == -1) & (model['choice'] == 1)] = 2
    choice_context[(model['context'] == 1) & (model['choice'] == -1)] = 3
    choice_context[(model['context'] == -1) & (model['choice'] == -1)] = 4

    model['choice_x_context'] = choice_context
    model['task_vars'] = task_vars
    model['cd_vars'] = cd_vars
    model['betas_ch_inm_incol'] = betResp_ch_inm_incol
    model['betas_inm_ch_incol'] = betResp_inm_ch_incol
    model['betas_inm_incol_ch'] = betResp_inm_incol_ch
    model['type'] = model_id
    model['bmat1'] = bmat1
    model['bmat2'] = bmat2
    model['bname1'] = bname1
    model['bname2'] = bname2
    model['bnorm'] = bnorm
    model['pca'] = pcaResp
    model['per_var_pca'] = per_var_pca

    return model


def show_task():
    # Load the image
    image = Image.open('utils_ex7/task.png') 

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(15, 12))  # Adjust the values as needed for size

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  
    plt.show()


def show_models_geometry():
    # Load the image
    image = Image.open('utils_ex7/models_v2.png') 

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(15, 12))  # Adjust the values as needed for size

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  
    plt.show()


def plot_psths(model):

    unit_ids = np.random.choice(model['neural'].shape[0], 6, replace=False)  # Select 6 random indices

    print('**** Sort by choice (all trials) ****')
    behavior_variable = 'choice'
    plot_psths_row(model, behavior_variable, None, unit_ids)
    plt.show()

    print('**** Sort by motion and choice (motion context) ****')
    behavior_variable = 'motion_coherence'
    context = 1
    plot_psths_row(model, behavior_variable, context, unit_ids)
    plt.show()

    print('**** Sort by color and choice (color context) ****')
    behavior_variable = 'color_coherence'
    context = -1
    plot_psths_row(model, behavior_variable, context, unit_ids)
    plt.show()

    print('**** Sort by context and choice (all trials) ****')
    behavior_variable = 'choice_x_context'
    plot_psths_row(model, behavior_variable, None, unit_ids)
    plt.show()

def plot_psths_row(model, behavior_variable, context, random_idx):
    #regressors = [model[behavior_variable] for behavior_variable in behavior_variables]
    unique_values = np.unique(model[behavior_variable])


    # Create a figure with 6 subplots side by side
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))  # 1 row, 6 columns

    if behavior_variable == 'choice':
        color_values = ['red', 'red']
        line_styles = ['solid', 'dashed']


    if behavior_variable == 'color_coherence': #context = -1
        color_values = ['#00008B', '#1E90FF', '#ADD8E6', '#ADD8E6', '#1E90FF', '#00008B']
        line_styles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed']

    if behavior_variable == 'motion_coherence': #context = 1
        color_values = ['#0D0D0D', '#4D4D4D', '#A6A6A6', '#A6A6A6', '#4D4D4D', '#0D0D0D']
        line_styles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed']

    if behavior_variable == 'choice_x_context':
        # motion +1 / color -1 / motion / color
        color_values = ['#0D0D0D', '#00008B', '#0D0D0D', '#00008B']
        line_styles = ['solid', 'solid', 'dashed', 'dashed']


    for i, ax in enumerate(axes):
        for j, condition_value in enumerate(unique_values):

            if context is None:
                filter_condition = model[behavior_variable][random_idx[i], :] == condition_value
            else:
                filter_condition = (model[behavior_variable][random_idx[i], :] == condition_value) & (model['context'][random_idx[i], :] == context)

            psth = np.mean(model['neural'][:, filter_condition, :], axis = 1)

            #ax.plot(psth[random_idx[i], :], label=f'{behavior_variable} {condition_value}')
            ax.plot(psth[random_idx[i], :], color=color_values[j], linestyle=line_styles[j], label=f'{behavior_variable} {condition_value}')
            if behavior_variable == 'color_coherence' or behavior_variable == 'motion_coherence':
                if j == 0:
                    ax.text(15, psth[random_idx[i], 15], 'strong', color=color_values[j], fontsize=12)
                if j == 2:
                    ax.text(15, psth[random_idx[i], 15], 'weak', color=color_values[j], fontsize=12)
            if behavior_variable == 'choice_x_context':
                if j == 0:
                    ax.text(15, psth[random_idx[i], 15], 'motion', color=color_values[j], fontsize=12)
                if j == 1:
                    ax.text(15, psth[random_idx[i], 15], 'color', color=color_values[j], fontsize=12)
            if behavior_variable == 'choice':
                if j == 0:
                    ax.text(15, psth[random_idx[i], 15], 'choice 1', color=color_values[j], fontsize=12)
                if j == 1:
                    ax.text(15, psth[random_idx[i], 15], 'choice -1', color=color_values[j], fontsize=12)

            ax.set_title(f'neuron {random_idx[i]}')
            #ax.legend(loc='best', fontsize='small', framealpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Response')

    plt.tight_layout()  # Adjusts the spacing between plots
    plt.show()

def plot_beta_norm(model_a, model_b, model_c, model_d):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6)) 
    models = [model_a, model_b, model_c, model_d]

    for id in range(4):
        model = models[id]
    
        axes[id].plot(model['bnorm'][0, :], label="choice direction")
        axes[id].plot(model['bnorm'][1, :], label="motion input")
        axes[id].plot(model['bnorm'][2, :], label="color input")
        axes[id].plot(model['bnorm'][3, :], label="context")
        axes[id].set_xlabel("Time")
        axes[id].set_ylabel("Beta Norm")
        axes[id].set_title(model['type'])
        # Add a legend for the 3 lines
        axes[id].legend()

def plot_pca_var_explained(model_a, model_b, model_c, model_d):
    # Create a figure with 4 subplots side by side
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    models = [model_a, model_b, model_c, model_d]

    for i, ax in enumerate(axs):
        model = models[i]

        # Create the primary (host) and secondary (parasite) axes
        host = ax
        parasite = host.twinx()
        axes = [host, parasite]
        
        # Plot on the primary axis
        axes[0].plot(np.cumsum(model['per_var_pca']), color='C0', label='cumulative variance explained')
        axes[0].set_xlabel('# component')

        # Plot on the secondary axis
        axes[1].plot(model['per_var_pca'] / sum(model_a['per_var_pca']), color='C2', label='variance explained')  
        axes[1].set_yscale('log')
        axes[0].set_title(model['type'])
        axes[0].legend()
        axes[1].legend()
        
        # Only add legend to the first subplot for clarity
        if i == 0:
            
            axes[0].set_xlabel('# component')
            axes[0].set_ylabel('cumulative variance explained', color='C0')
            axes[1].set_ylabel('variance explained', color='C2')
            
    plt.show()



def plot_tunings(model):
    # Adjust spacing with 'wspace' for columns and 'hspace' for rows
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # Increase height to accommodate more row space
    plt.subplots_adjust(wspace=0, hspace=0.5)  # Adjust the space between columns and rows

    # Loop over the subplots and plot the data
    for i, ax in enumerate(axes.flat):
        idx = i  # index for each subplot
        ax.scatter(model['bmat1'][:,idx],  model['bmat2'][:,idx], edgecolor='white', color='black', s=20)
        ax.set_xlabel(model['bname1'][idx])
        ax.set_ylabel(model['bname2'][idx])

        # Set the aspect ratio dynamically based on the data ratio for each subplot
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  # Individual aspect ratio

    # Title for the figure
    plt.suptitle(model['type'], fontsize=16)

    # Show the plot
    plt.show()

def get_pdata():
    pdata = {
        (0, 0): {
            "pdesc": "relevant motion",
            "pjcor": [2, 2, 2, 2, 2, 2],
            "ptdir": [1, 1, 1, 2, 2, 2],
            "psdir": [1, 2, 3, 4, 5, 6],
            "pscol": [0, 0, 0, 0, 0, 0],
            "pjmot": [2, 2, 2, 2, 2, 2],
            "pstyl": [3, 2, 1, 1, 2, 3],
            "pcmap": ['#0D0D0D', '#4D4D4D', '#A6A6A6', '#A6A6A6', '#4D4D4D', '#0D0D0D'], #['blue', 'blue', 'blue', 'blue', 'blue', 'blue'],#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "pdash": [2, 2, 2, 1, 1, 1],
            "pwdth": [2, 2, 2, 1, 1, 1],
            "pordr": 1
        },
        (0, 1): {
            "pdesc": "irrelevant color",
            "pjcor": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "ptdir": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "psdir": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "pscol": [6, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1],
            "pjmot": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "pstyl": [3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3],
            "pcmap": ['#0D0D0D', '#4D4D4D', '#A6A6A6', '#A6A6A6', '#4D4D4D', '#0D0D0D','#0D0D0D', '#4D4D4D', '#A6A6A6', '#A6A6A6', '#4D4D4D', '#0D0D0D'], #['black', 'black', 'black','black', 'black', 'black'],#[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "pdash": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "pwdth": [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
            "pordr": 1
        },
        (1, 0): {
            "pdesc": "irrelevant motion",
            "pjcor": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "ptdir": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "psdir": [6, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1],
            "pscol": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "pjmot": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "pstyl": [3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3],
            "pcmap": ['#00008B', '#1E90FF', '#ADD8E6', '#ADD8E6', '#1E90FF', '#00008B','#00008B', '#1E90FF', '#ADD8E6', '#ADD8E6', '#1E90FF', '#00008B'],#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "pdash": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "pwdth": [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
            "pordr": 2
        },
        (1, 1): {
            "pdesc": "relevant color",
            "pjcor": [2, 2, 2, 2, 2, 2],
            "ptdir": [1, 1, 1, 2, 2, 2],
            "psdir": [0, 0, 0, 0, 0, 0],
            "pscol": [1, 2, 3, 4, 5, 6],
            "pjmot": [1, 1, 1, 1, 1, 1],
            "pstyl": [3, 2, 1, 1, 2, 3],
            "pcmap": ['#00008B', '#1E90FF', '#ADD8E6', '#ADD8E6', '#1E90FF', '#00008B'], #['black', 'black', 'black','black', 'black', 'black'], #[4, 4, 4, 4, 4, 4],
            "pdash": [2, 2, 2, 1, 1, 1],
            "pwdth": [2, 2, 2, 1, 1, 1],
            "pordr": 2
        }
    }

    return pdata

def plot_projections_2d(model, method,order_orth=[], context='motion'):
    if method == 'pca':
        plot_projections_2d_pca(model, context)
    else:
        plot_projections_2d_tdr(model, order_orth)


def plot_projections_2d_pca(model, context):
    pcaResp = model['pca']
    cd_vars = model['cd_vars']
    ndim, ncond, time = pcaResp.shape
    jtt = np.ones((time,), dtype=bool)

    pdata = get_pdata()
    #icol, irow = 0, 0
    if context == 'motion':
        irow = 0
    else:
        irow = 1

    for icol in range(2): #gives the input (relevant / irrelevant)
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))

        if irow == 0:
            if icol == 0:
                axes[0,0].text(-0.3, 1.1, 'Motion Context\nMotion Input', transform=axes[0, 0].transAxes, fontsize=16, va='center', ha='center')
            else:
                axes[0,0].text(-0.3, 1.1, 'Motion Context\nColor Input', transform=axes[0, 0].transAxes, fontsize=16, va='center', ha='center')
        else:
            if icol == 0:
                axes[0,0].text(-0.3, 1.1, 'Color Context\nMotion Input', transform=axes[0, 0].transAxes, fontsize=16, va='center', ha='center')
            else:
                axes[0,0].text(-0.3, 1.1, 'Color Context\nColor Input', transform=axes[0, 0].transAxes, fontsize=16, va='center', ha='center')

        for ipca1 in range(4):
            for ipca2 in range(4):
                if ipca1 > ipca2:
                    np_len = len(pdata[(icol, irow)]['pjcor'])
                    ax = axes[ipca1-1, ipca2]  # Get the current subplot

                    # Loop over conditions and plot
                    for ip in range(np_len):
                        # Find condition
                        jj1 = (cd_vars['kjcor'] == pdata[(icol, irow)]['pjcor'][ip])
                        jj2 = (cd_vars['ktdir'] == pdata[(icol, irow)]['ptdir'][ip])
                        jj3 = (cd_vars['ksdir'] == pdata[(icol, irow)]['psdir'][ip])
                        jj4 = (cd_vars['kscol'] == pdata[(icol, irow)]['pscol'][ip])
                        jj5 = (cd_vars['kjmot'] == pdata[(icol, irow)]['pjmot'][ip])

                        # Indices to use
                        jj = jj1 & jj2 & jj3 & jj4 & jj5

                        index = np.where(jj)[0]
                        if np.sum(jj) > 1:
                            raise ValueError('More than one condition')


                        # Plot before dots offset
                        ax.plot(pcaResp[ipca1, jj, jtt], pcaResp[ipca2, jj, jtt], color = pdata[(icol, irow)]['pcmap'][ip])
                        ax.set_xlabel(f'pca {ipca1+1}')
                        ax.set_ylabel(f'pca {ipca2+1}')
                            

                        # Adjust marker style
                        if pdata[(icol, irow)]['pdash'][ip] == 1:
                            ax.scatter(pcaResp[ipca1, jj, jtt], pcaResp[ipca2, jj, jtt], marker='o', 
                                    c=pdata[(icol, irow)]['pcmap'][ip], s=20)  # or c='#000000'
                        elif pdata[(icol, irow)]['pdash'][ip] == 2:
                            ax.scatter(pcaResp[ipca1, jj, jtt], pcaResp[ipca2, jj, jtt],  marker='o',
                                    facecolors='none', edgecolor=pdata[(icol, irow)]['pcmap'][ip], s=20)  # or c='#000000'

                    ax.scatter(pcaResp[ipca1, jj, 0], pcaResp[ipca2, jj, 0], color= 'purple', s = 30)  # or c='#000000'
 
        plt.tight_layout()
        plt.show()


def plot_projections_2d_tdr(model, order_orth):
    axes_limits_models = {}
    cd_vars = model['cd_vars']
    if order_orth[0] == 'choice' and order_orth[1] == 'input_motion' and order_orth[2] == 'input_color':
        betResp = model['betas_ch_inm_incol']



    if order_orth[0] == 'input_motion' and order_orth[1] == 'choice' and order_orth[2] == 'input_color':
        betResp = model['betas_inm_ch_incol']



    if order_orth[0] == 'input_motion' and order_orth[1] == 'input_color' and order_orth[2] == 'choice':
        betResp = model['betas_inm_incol_ch']


    ndim, ncond, time = betResp.shape
    pdata = get_pdata()

    ncol = 2
    nrow = 2
    nsub = 4

    ha = np.zeros((ncol, nrow), dtype=object)
    axlims = [np.inf, -np.inf]
    aylims = [np.inf, -np.inf]

    jtt = np.ones((time,), dtype=bool)

    if model['type'] == 'model_a':
        pdims = [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
            ]
    else:
        pdims = [
            [[0, 1], [0, 2]],
            [[0, 1], [0, 2]]
            ]

    xlabels = ['choice (motion)', 'choice (motion)', 'choice (color)', 'choice (color)']
    ylabels = ['motion (motion)', 'color (motion)', 'motion (color)', 'color (color)']

    # Loop through subplots
    for isub in range(1, nsub + 1):
        irow, icol = np.unravel_index(isub - 1, (nrow, ncol))  

        # Create subplot
        ha[icol, irow] = plt.subplot(ncol, nrow, isub)

        plt.subplots_adjust(hspace=0.5)  
        plt.subplots_adjust(wspace=0.5)  

        # Dimensions to plot
        id1 = pdims[icol][irow][0]
        id2 = pdims[icol][irow][1]

        np_len = len(pdata[(icol, irow)]['pjcor'])
    
        ron = np.zeros((np_len, 2))

        for ip in range(np_len):
            # Find condition
            jj1 = (cd_vars['kjcor'] == pdata[(icol, irow)]['pjcor'][ip])
            jj2 = (cd_vars['ktdir'] == pdata[(icol, irow)]['ptdir'][ip])
            jj3 = (cd_vars['ksdir'] == pdata[(icol, irow)]['psdir'][ip])
            jj4 = (cd_vars['kscol'] == pdata[(icol, irow)]['pscol'][ip])
            jj5 = (cd_vars['kjmot'] == pdata[(icol, irow)]['pjmot'][ip])

            # Indices to use
            jj = jj1 & jj2 & jj3 & jj4 & jj5

            if np.sum(jj) > 1:
                raise ValueError('More than one condition')

            # Response on first sample
            ron[ip, 0] = betResp[id1, jj, 0]
            ron[ip, 1] = betResp[id2, jj, 0]


        # Loop over conditions and plot
        for ip in range(np_len):
            # Find condition
            jj1 = (cd_vars['kjcor'] == pdata[(icol, irow)]['pjcor'][ip])
            jj2 = (cd_vars['ktdir'] == pdata[(icol, irow)]['ptdir'][ip])
            jj3 = (cd_vars['ksdir'] == pdata[(icol, irow)]['psdir'][ip])
            jj4 = (cd_vars['kscol'] == pdata[(icol, irow)]['pscol'][ip])
            jj5 = (cd_vars['kjmot'] == pdata[(icol, irow)]['pjmot'][ip])

            # Indices to use
            jj = jj1 & jj2 & jj3 & jj4 & jj5

            index = np.where(jj)[0]
            if np.sum(jj) > 1:
                raise ValueError('More than one condition')


            # Plot before dots offset
            plt.plot(betResp[id1, jj, jtt] - np.mean(ron[:, 0]), betResp[id2, jj, jtt] - np.mean(ron[:, 1]), color = pdata[(icol, irow)]['pcmap'][ip])
        
            # Plot with markers
            plt.plot(betResp[id1, jj, jtt] - np.mean(ron[:, 0]), betResp[id2, jj, jtt] - np.mean(ron[:, 1]), color = pdata[(icol, irow)]['pcmap'][ip])

            # Adjust marker style
            if pdata[(icol, irow)]['pdash'][ip] == 1:
                plt.scatter(betResp[id1, jj, jtt] - np.mean(ron[:, 0]), betResp[id2, jj, jtt] - np.mean(ron[:, 1]),  marker='o', 
                        c=pdata[(icol, irow)]['pcmap'][ip], s=20)  # or c='#000000'
            elif pdata[(icol, irow)]['pdash'][ip] == 2:
                plt.scatter(betResp[id1, jj, jtt] - np.mean(ron[:, 0]), betResp[id2, jj, jtt] - np.mean(ron[:, 1]),  marker='o',
                        facecolors='none', edgecolor=pdata[(icol, irow)]['pcmap'][ip], s=20)  # or c='#000000'

            plt.xlabel(xlabels[isub-1])
            plt.ylabel(ylabels[isub-1])

            if model['type'] == 'model_a':
                if isub < 3:
                    plt.gca().invert_xaxis()
            
            if model['type'] == 'model_b':
                plt.gca().invert_xaxis()

            if model['type'] == 'model_c':
                plt.gca().invert_xaxis()

            if model['type'] == 'model_d':
                plt.gca().invert_xaxis()

            # Retrieve current x and y limits
        x_limits = ha[icol, irow].get_xlim()
        y_limits = ha[icol, irow].get_ylim()

        # Check and overwrite limits if the absolute value is below 25
        if abs(x_limits[0]) < 25 or abs(x_limits[1]) < 25:
            plt.xlim([-55, 55])

        if abs(y_limits[0]) < 25 or abs(y_limits[1]) < 25:
            plt.ylim([-55, 55])
        
        plt.scatter(0, 0, color= 'purple', s = 30)  # or c='#000000'

    plt.suptitle(model['type'])
    plt.show()
    

def plot_projections_1d(model, method, order_orth=[], context='motion'):
    cd_vars = model['cd_vars']

    if method == 'tdr':
        if order_orth[0] == 'choice' and order_orth[1] == 'input_motion' and order_orth[2] == 'input_color':
            betResp = model['betas_ch_inm_incol']

        if order_orth[0] == 'input_motion' and order_orth[1] == 'choice' and order_orth[2] == 'input_color':
            betResp = model['betas_inm_ch_incol']


        if order_orth[0] == 'input_motion' and order_orth[1] == 'input_color' and order_orth[2] == 'choice':
            betResp = model['betas_inm_incol_ch']

        number_axes = 3
    else:
        betResp = model['pca']
        number_axes = 4

    ndim, ncond, time = betResp.shape

    pdata = get_pdata()
    #icol, irow = 0, 0
    if context == 'motion':
        irow = 0
        
    else:
        irow = 1
        model_a_tdr_labels = ['color input', 'choice', 'context']

    tdr_labels = ['choice', 'motion input', 'color input']

    # Set up 2x3 grid of subplots
    if number_axes == 3:
        fig, axes = plt.subplots(2, number_axes, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(2, number_axes, figsize=(16, 8))

    for icol in range(2): 
        if icol == 0:
            model_a_tdr_labels = ['motion input', 'choice', 'context']
        else:
            model_a_tdr_labels = ['color input', 'choice', 'context']
            
        np_len = len(pdata[(icol, irow)]['pjcor'])
        jtt = np.ones((time,), dtype=bool)

        for id1 in range(number_axes):
            ax = axes[icol,id1]  # Get the current subplot
            ax.set_title(f'{method} {id1+1}')  # Individual panel titles

            # Loop over conditions and plot
            for ip in range(np_len):
                # Find condition
                jj1 = (cd_vars['kjcor'] == pdata[(icol, irow)]['pjcor'][ip])
                jj2 = (cd_vars['ktdir'] == pdata[(icol, irow)]['ptdir'][ip])
                jj3 = (cd_vars['ksdir'] == pdata[(icol, irow)]['psdir'][ip])
                jj4 = (cd_vars['kscol'] == pdata[(icol, irow)]['pscol'][ip])
                jj5 = (cd_vars['kjmot'] == pdata[(icol, irow)]['pjmot'][ip])

                # Indices to use
                jj = jj1 & jj2 & jj3 & jj4 & jj5

                index = np.where(jj)[0]
                if np.sum(jj) > 1:
                    raise ValueError('More than one condition')


                # Plot before dots offset
                ax.plot(betResp[id1, jj, jtt], color = pdata[(icol, irow)]['pcmap'][ip])

                # Adjust marker style
                if pdata[(icol, irow)]['pdash'][ip] == 1:
                    ax.scatter(np.arange(0, time), betResp[id1, jj, jtt],  marker='o', 
                            c=pdata[(icol, irow)]['pcmap'][ip], s=20)  # or c='#000000'
                elif pdata[(icol, irow)]['pdash'][ip] == 2:
                    ax.scatter(np.arange(0, time), betResp[id1, jj, jtt],  marker='o',
                            facecolors='none', edgecolor=pdata[(icol, irow)]['pcmap'][ip], s=20)  # or c='#000000'

                ax.set_xlabel('time')
                if method == 'tdr':
                    if model['type'] == 'model_a':
                        if context == 'motion':
                            ax.set_ylabel(f'{model_a_tdr_labels[id1]} projection')      
                        else:
                            ax.set_ylabel(f'{model_a_tdr_labels[id1]} projection')             
                    else:
                        ax.set_ylabel(f'{tdr_labels[id1]} projection')   
                else:
                    ax.set_ylabel(f'{method} projection')

    
    if context == 'motion':
        # Add a title for the first row
        axes[0, 0].text(-0.3, 1.1, 'Motion Context\nMotion Input', transform=axes[0, 0].transAxes, fontsize=16, va='center', ha='center')

        # Add a title for the second row
        axes[1, 0].text(-0.3, 1.1, 'Motion Context\nColor Input', transform=axes[1, 0].transAxes, fontsize=16, va='center', ha='center')
    else:
                # Add a title for the first row
        axes[0, 0].text(-0.3, 1.1, 'Color Context\nMotion Input', transform=axes[0, 0].transAxes, fontsize=16, va='center', ha='center')

        # Add a title for the second row
        axes[1, 0].text(-0.3, 1.1, 'Color Context\nColor Input', transform=axes[1, 0].transAxes, fontsize=16, va='center', ha='center')


    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()