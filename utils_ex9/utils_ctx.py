import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipywidgets import interact, widgets
from PIL import Image

from ipywidgets import FloatSlider, Button, VBox, Output, Layout
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.cm as cm
from ipywidgets import interact, widgets

import h5py

import warnings

warnings.filterwarnings("ignore")


def show_network():
    # Load the image
    image = Image.open('utils_ex9/ctx_rnn.png') 

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(15, 12))  # Adjust the values as needed for size

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  
    plt.show()

def load_rnn():

    n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1 = np.load('rnn_data/rnn_weights.npz').values()
    coherencies_trial, conditionIds, inputs, targets = np.load('rnn_data/trial_data.npz').values()

    trialId = 8
    gaussian_inputs = inputs[:,:,trialId]    

    pulse_inputs = np.zeros(inputs.shape)
    vector = np.zeros((1, 1400), dtype=float)
    pulse_inputs[1,:,trialId] = vector
    random_indices = np.random.choice(1400, size=5, replace=False)
    vector[0, random_indices] = 1
    pulse_inputs[0,:,trialId] = vector
    

    step_inputs = np.zeros(inputs.shape)
    vector = np.zeros((1, 1400), dtype=float)
    step_inputs[1,:,trialId] = vector
    vector[0, 100: 1100] = 1
    step_inputs[0,:,trialId] = vector
    

    pulse_inputs = pulse_inputs[:,:,trialId]
    step_inputs = step_inputs[:,:,trialId]

    step_inputs = step_inputs.reshape(5, 1400, 1)
    pulse_inputs = pulse_inputs.reshape(5, 1400, 1)
    gaussian_inputs = gaussian_inputs.reshape(5, 1400, 1)

    return n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, gaussian_inputs, pulse_inputs, step_inputs, conditionIds[:,:1]


def run_rnn(n_Wru_v, n_Wrr_n, m_Wzr_n, n_bx0_c, n_bx_1, m_bz_1, dt, tau,
            noise_sigma, inputs, conditionIds):
    # n_Wru_v: [n_units, n_inputs], input weights
    # n_Wrr_n: [n_units, n_units], recurrent weights
    # m_Wzr_n: [n_outputs, n_inputs], output weights
    # n_x0_c: [n_units, n_contexts], initial conditions per context
    # n_bx_1: [n_units, 1], bias of hidden units
    # m_bz_1: [n_outputs, 1], bias of output units
    # dt: (float), simulation time step
    # tau: (float), time constant
    # noise_sigma: (float), input noise sigma
    # inputs: [n_units, n_timesteps, n_trials], inputs to network u_t
    # conditionIds: [n_trials], condition id per trial (ctxt 1 or 2)

    # Outputs:
    # n_r_t: [n_units, n_timesteps, n_trials], activities of recurrent units (including transfer function)
    # m_z_t: [n_outputs, n_timesteps, n_trials], activities of readout unit
    # n_r0_1: [n_initial_conditions, n_trials], initial condition(s)
    # n_x_t: [n_units, n_timesteps, n_trials], membrane potentials of recurrent units (excluding transfer function)
    # n_x0_1: [n_units, n_trials], initial condition(s)

    [_, n_timesteps, n_trials] = np.shape(inputs)
    [n_outputs, n_units] = np.shape(m_Wzr_n)


    n_x_t = np.zeros([n_units, n_timesteps, n_trials])
    n_r_t = np.zeros([n_units, n_timesteps, n_trials])
    m_z_t = np.zeros([n_outputs, n_timesteps, n_trials])
    n_x0_1 = np.zeros([n_units, n_trials])
    n_r0_1 = np.zeros([n_units, n_trials])


    for trial_nr in range(n_trials):
        n_x0_1[:, trial_nr] = n_bx0_c[:, int(conditionIds[0, trial_nr] - 1)]
        n_r0_1[:, trial_nr] = np.tanh(n_x0_1[:, trial_nr])
        n_x_1 = n_x0_1[:, trial_nr]
        n_r_1 = n_r0_1[:, trial_nr]
        n_Wu_t = np.matmul(n_Wru_v, inputs[:, :, trial_nr])
        n_nnoise_t = noise_sigma * np.random.normal(size=[n_units, n_timesteps])
        for t in range(n_timesteps):
            n_x_1 = (1.0 - (dt / tau)) * n_x_1 + (dt / tau) * (n_Wu_t[:, t]
                        + np.matmul(n_Wrr_n, n_r_1) + np.squeeze(n_bx_1) + n_nnoise_t[:, t])
            n_r_1 = np.tanh(n_x_1)
            n_x_t[:, t, trial_nr] = n_x_1
            n_r_t[:, t, trial_nr] = n_r_1
            m_z_t[:, t, trial_nr] = np.matmul(m_Wzr_n, n_r_t[:, t, trial_nr]) + m_bz_1

    return n_r_t, m_z_t, n_r0_1, n_x_t, n_x0_1


def load_weights(path_to_weights, net_id):
        # load network weights
        # path_to_weights: (str), path to hdf5-file with stored weight matrices
        # net_id: (integer), number of network in hdf5-file to load

        # n_Wru_v: [n_units, n_inputs], input weights
        # n_Wrr_n: [n_units, n_units], recurrent weights
        # m_Wzr_n: [n_outputs, n_inputs], output weights
        # n_x0_c: [n_units, n_contexts], initial conditions per context
        # n_bx_1: [n_units, 1], bias of hidden units
        # m_bz_1: [n_outputs, 1], bias of output units

        name_dataset = '/NetNr' + str(net_id) + '/final'

        f = h5py.File(path_to_weights, 'r')
        n_Wru_v = np.asarray(f[name_dataset + '/n_Wru_v']).T
        n_Wrr_n = np.asarray(f[name_dataset + '/n_Wrr_n']).T
        m_Wzr_n = np.asarray(f[name_dataset + '/m_Wzr_n']).T
        n_x0_c = np.asarray(f[name_dataset + '/n_x0_c']).T
        n_bx_1 = np.asarray(f[name_dataset + '/n_bx_1']).T
        m_bz_1 = np.asarray(f[name_dataset + '/m_bz_1']).T
        f.close()

        return n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1




def run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1,
                        inputs, conditionIds, seed_run, net_noise):
    # run network for one forward pass (several trials)
    # n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1: network weights
    # inputs: [2*nIntegrators+1 x T x N], sensory and context input over time and trials
    # conditionIds = [1 x N], context ID per trial
    # seed_run: (integer), random seed to get frozen noise
    # net_noise: (bool), if true: add gaussin noise to hidden unit activity

    # 'forwardPass'
    # n_x0_1: [n_units, n_trials], initial condition(s)
    # n_x_t: [n_units, n_timesteps, n_trials], membrane potentials of recurrent units (linear)
    # n_r0_1: [n_initial_conditions, n_trials], tanh(initial condition(s))
    # n_r_t: [n_units, n_timesteps, n_trials], activities of recurrent units (incl. tanh)
    # m_z_t: [n_outputs, n_timesteps, n_trials], activities of readout unit

    if not (seed_run is None):
        np.random.seed(seed_run)

    net = {}
    net["tau"] = 0.01
    net["dt"] = 0.001
    net["noiseSigma"] = net_noise

    if not (net_noise == 'default'):
        net["noiseSigma"] = net_noise

    forwardPass = {}
    forwardPass['n_r_t'], forwardPass['m_z_t'], forwardPass['n_r0_1'], \
    forwardPass['n_x_t'], forwardPass['n_x0_1'] = run_rnn(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c,
                                                          n_bx_1, m_bz_1, net["dt"], net["tau"],
                                                          net["noiseSigma"], inputs, conditionIds)

    return forwardPass


def generate_rnn_data():

    # Output widget for displaying the plot
    output_plot = Output()

    n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, gaussian_inputs, pulse_inputs, step_inputs, conditionId = load_rnn()
    

    # Update the plot (triggered by the button)
    def update_plot(input_type="gaussian input", amplitude = '+1 choice 1', context = 'motion context', add_noise = False, change_trial = False):  

        #n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, gaussian_inputs, pulse_inputs, step_inputs, conditionId = load_rnn(change_trial)

        if amplitude == '+1 choice 1':
            amp = +1
        else:
            amp = -1

        # create input
        if input_type == 'gaussian input':
            inputs = gaussian_inputs
        elif input_type == 'pulse input':
            inputs = pulse_inputs
        elif input_type == 'step input':
            inputs = step_inputs


        new_inputs = inputs
        if (np.sum(inputs[0,:,0]) < 0 and amp == 1) or (np.sum(inputs[0,:,0]) > 0 and amp == -1):
            new_inputs[0,:,0] = inputs[0,:,0] * (-1)
            new_inputs[1,:,0] = inputs[1,:,0] * (-1)
 
            
        # context signals
        #if context >= 0:
        #    new_inputs[2,:,0] = 0
        #    new_inputs[3,:,0] = context
        #else:
        #    new_inputs[2,:,0] = (-1) * context
        #    new_inputs[3,:,0] = 0
        if context == 'motion context':
            new_inputs[2,:,0] = 1
            new_inputs[3,:,0] = 0
        else:
            new_inputs[2,:,0] = 0
            new_inputs[3,:,0] = 1
        
        # generate data
        if add_noise == True:
            net_noise = 0.1
        else:
            net_noise = 0

        forward_pass_data = run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1, new_inputs, conditionId, 0, net_noise)

        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)


            trial_id = 0

            f = plt.figure(figsize=(16, 8))

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.4, hspace=0.6)

            ax = plt.subplot(2, 4, 1)
            ax.plot(inputs[0, :, trial_id], label='motion input', color = 'black')
            ax.set_title('Motion input')
            ax.set_xlabel('Time')


            ax = plt.subplot(2, 4, 2)
            ax.plot(inputs[1, :, trial_id], label='color input', color='blue')
            ax.set_title('Color input')
            ax.set_xlabel('Time')


            ax = plt.subplot(2, 4, 3)
            ax.plot(inputs[2, :, trial_id], label='motion context (-1)', color = 'black')
            ax.plot(inputs[3, :, trial_id], label='color context (+1)', color='blue')
            plt.legend()
            ax.set_title('Context signals')
            ax.set_ylim([-0.1, 1.1])
            ax.set_xlabel('Time')

            ax = plt.subplot(2, 4, 4)
            ax.imshow(forward_pass_data['n_r_t'][:, :, trial_id],  aspect='auto')
            ax.set_title('Neuronal Activity ($X$)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Neurons')

            max_value = np.max(np.abs(forward_pass_data['n_r_t'][:, :, trial_id].T @  n_Wru_v[:, 0]))
            ax = plt.subplot(2, 4, 5)
            ax.plot(forward_pass_data['n_r_t'][:, :, trial_id].T @  n_Wru_v[:, 0], color = 'black')
            ax.set_title('projection onto \n motion-input weights')
            ax.set_ylim([-max_value, +max_value])
            ax.set_xlabel('Time')

            max_value = np.max(np.abs(forward_pass_data['n_r_t'][:, :, trial_id].T @  n_Wru_v[:, 1]))
            ax = plt.subplot(2, 4, 6)
            ax.plot(forward_pass_data['n_r_t'][:, :, trial_id].T @  n_Wru_v[:, 1],color='blue')
            ax.set_title('projection onto \n color-input weights')
            ax.set_ylim([-max_value, +max_value])
            ax.set_xlabel('Time')

            max_value = np.max(np.abs(forward_pass_data['n_r_t'][:, :, trial_id].T @  m_Wzr_n.T))
            ax = plt.subplot(2, 4, 7)
            ax.plot(forward_pass_data['n_r_t'][:, :, trial_id].T @  m_Wzr_n.T, color = 'red')
            ax.set_title('projection onto \n output weights')
            ax.set_ylim([-max_value, +max_value])
            ax.set_xlabel('Time')

            context_weights = n_Wru_v[:, 2] - n_Wru_v[:, 3]

            ax = plt.subplot(2, 4, 8)
            max_value2 = np.max(np.abs(forward_pass_data['n_r_t'][:, :, trial_id].T @  context_weights.T))
            ax.plot(forward_pass_data['n_r_t'][:, :, trial_id].T @  context_weights.T, forward_pass_data['n_r_t'][:, :, trial_id].T @  m_Wzr_n.T)
            ax.set_title('2-d projections')
            ax.set_ylabel('projection onto output weights')
            ax.set_xlabel('projection onto context weights')
            ax.set_ylim([-max_value, +max_value])
            ax.set_xlim([-max_value2, max_value2])
            ax.scatter(0, 0, color = 'purple')


            plt.tight_layout()
            plt.show()

    # Create sliders for synaptic weight, frequency, and length
    style = {'description_width': 'initial'}
    context1_slider = widgets.FloatSlider(min=-1, max=1, step=0.1, value=1, description='Context',continuous_update=False)
    context_dropdown = widgets.Dropdown(options=["motion context", "color context"], value="motion context", description="Context")
    #context2_slider = widgets.FloatSlider(min=0, max=1, step=0.1, value=0, description='Context 2',continuous_update=False)
    add_noise_checkbox = widgets.Checkbox(value=False, description='Add noise') 
    input_type_dropdown = widgets.Dropdown(options=["gaussian input", "pulse input"], value="gaussian input", description="Input type")
    amplitude_dropdown = widgets.Dropdown(options=["+1 choice 1", "-1 choice 2"], value="+1 choice 1", description="Choice")


    # Create the button to trigger the plot update
    plot_button = Button(description="Update Plot", button_style='success')

    # Callback function for the checkbox to reset it after checking
    #def on_checkbox_change(change):
    #    if change['new']:
    #        update_plot(
    #            input_type=input_type_dropdown.value,
    #            amplitude=amplitude_dropdown.value,
    #            context=context_dropdown.value,
    #            add_noise=add_noise_checkbox.value
    #            change_trial=change_trial_checkbox.value
    #        )
    #        # Reset the checkbox back to False
    #        change_trial_checkbox.value = False

    # Attach the event handler to the checkbox
    #change_trial_checkbox.observe(on_checkbox_change, names='value')

    # Callback function to update plot when the button is clicked
    def on_button_click(b):
        update_plot(input_type=input_type_dropdown.value, amplitude = amplitude_dropdown.value, context=context_dropdown.value,  add_noise = add_noise_checkbox.value)

    # Link button to callback function
    plot_button.on_click(on_button_click)

    # Create a vertical box to hold sliders, the button, and the output plot
    ui = VBox([input_type_dropdown, amplitude_dropdown, context_dropdown,  add_noise_checkbox, plot_button, output_plot])

    display(ui)

    update_plot(input_type=input_type_dropdown.value, amplitude = amplitude_dropdown.value, context=context_dropdown.value, add_noise = add_noise_checkbox)