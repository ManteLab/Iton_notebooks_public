import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Button, FloatSlider, IntSlider, Layout, Output, VBox


def iplot_bcm_model():
    output_plot = Output()

    def simulate(eta,
                 tau,
                 epsilon,
                 simulation_time,
                 p,
                 freq_1,
                 freq_2,
                 amp_1,
                 amp_2,
                 phase_shift,
                 w1_init,
                 w2_init,
                 use_decay):

        # initialize synaptic weights and inputs:
        w = np.array([w1_init, w2_init])
        x1 = np.cos(2 * np.pi * freq_1 * np.linspace(0, 1, simulation_time)) * amp_1
        x2 = np.cos(2 * np.pi * freq_2 * np.linspace(0, 1, simulation_time) + phase_shift) * amp_2

        x1[x1 < 0] = 0
        x2[x2 < 0] = 0

        inputs = np.vstack((x1, x2))

        # initialize variables for storing results:
        y = np.zeros(simulation_time)
        theta_M = np.zeros(simulation_time)
        avg_y = 0  # initial average postsynaptic activity
        w_history = np.zeros((simulation_time, 2))  # to store synaptic weights over time

        # simulation loop:
        for t in range(simulation_time):
            # compute postsynaptic activity:
            y[t] = np.dot(w, inputs[:, t])

            # update average postsynaptic activity:
            avg_y = avg_y + (y[t] - avg_y) / tau

            # update the sliding threshold:
            theta_M[t] = avg_y ** p

            # update synaptic weights according to the BCM rule:
            if use_decay:
                delta_w = eta * y[t] * (y[t] - theta_M[t]) * inputs[:, t] - epsilon * w
            else:
                delta_w = eta * y[t] * (y[t] - theta_M[t]) * inputs[:, t]

            w += delta_w

            # ensure weights remain within a reasonable range:
            w = np.clip(w, 0, 1)

            # store synaptic weights:
            w_history[t] = w

        return x1, x2, y, theta_M, w_history

    def update_plot(eta=0.01,  # learning rate
                    tau=100.0,  # time constant for averaging postsynaptic activity
                    epsilon=0.001,  # decay rate (only used if decay term is included)
                    simulation_time=500,  # total simulation time in ms
                    p=2,  # exponent for the sliding threshold function
                    freq_1=5,  # frequency of input 1
                    freq_2=5,  # frequency of input 2
                    amp_1=0.8,  # amplitude of input 1
                    amp_2=0.8,  # amplitude of input 2
                    phase_shift=np.pi / 4,  # phase shift between inputs
                    w1_init=0.5,  # Initial synaptic weight 1
                    w2_init=0.5,  # Initial synaptic weight 2
                    use_decay=False  # whether to include weight decay
                    ):

        x1, x2, y, theta_M, w_history = simulate(
            tau=tau,
            eta=eta,
            epsilon=epsilon,
            simulation_time=simulation_time,
            p=p,
            freq_1=freq_1,
            freq_2=freq_2,
            amp_1=amp_1,
            amp_2=amp_2,
            phase_shift=phase_shift,
            w1_init=w1_init,
            w2_init=w2_init,
            use_decay=use_decay
        )

        with output_plot:
            output_plot.clear_output(wait=True)

            # Improved plots:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(15, 10))

            # Plot Presynaptic Inputs
            ax1.plot(x1, label='Input 1', alpha=0.7, color='purple')
            ax1.plot(x2, label='Input 2', alpha=0.7, color='teal')
            ax1.set_ylabel('Input Signal')
            ax1.set_title('Presynaptic Input Signals')
            ax1.legend()
            ax1.grid()

            # Plot Postsynaptic Activity and Sliding Threshold
            ax2.plot(y, label='Postsynaptic Activity (y)', color='blue')
            ax2.plot(theta_M, label='Sliding Threshold (\u03B8M)', linestyle='--', color='orange')
            ax2.set_ylabel('Activity / Threshold')
            ax2.set_title('Postsynaptic Activity vs Sliding Threshold')
            ax2.legend()
            ax2.grid()

            # Plot Synaptic Weights
            ax3.plot(w_history[:, 0], label='Weight 1', color='purple')
            ax3.plot(w_history[:, 1], label='Weight 2', color='teal')
            ax3.set_ylabel('Synaptic Weights')
            ax3.set_title('Evolution of Synaptic Weights')
            ax3.legend()
            ax3.grid()

            # Show Input-Weight Relation
            ax4.plot(np.cumsum(x1), label='Cumulative Input 1', color='purple')
            ax4.plot(np.cumsum(x2), label='Cumulative Input 2', color='teal')
            # ax4.plot(np.cumsum(w_history[:, 0]), label='Cumulative Weight 1', linestyle='-')
            # ax4.plot(np.cumsum(w_history[:, 1]), label='Cumulative Weight 2', linestyle='--')
            ax4.set_ylabel('Cumulative Effect')
            ax4.set_title('Cumulative Inputs')
            ax4.legend()
            ax4.grid()

            plt.tight_layout()
            plt.show()

    style = {'description_width': 'initial'}

    eta_slider = FloatSlider(value=0.04, min=0.0, max=0.1, step=0.001, description='eta', style=style,
                             layout=Layout(width='500px'))
    tau_slider = FloatSlider(value=50.0, min=0.0, max=200.0, step=10.0, description='tau', style=style,
                             layout=Layout(width='500px'))
    epsilon_slider = FloatSlider(value=0.001, min=0.0, max=0.01, step=0.001, description='epsilon', style=style,
                                 layout=Layout(width='500px'))
    simulation_time_slider = IntSlider(value=1000, min=10, max=1000, step=10, description='simulation time',
                                       style=style, layout=Layout(width='500px'))
    p_slider = FloatSlider(value=2, min=1, max=5, step=1, description='p', style=style, layout=Layout(width='500px'))
    freq_1_slider = FloatSlider(value=0, min=0.0, max=10, step=.1, description='Input 1 frequency', style=style,
                                layout=Layout(width='500px'))
    freq_2_slider = FloatSlider(value=0, min=0.0, max=10, step=.1, description='Input 2 frequency', style=style,
                                layout=Layout(width='500px'))
    amp_1_slider = FloatSlider(value=0.4, min=0.0, max=1.0, step=0.1, description='Input 1 amplitude.', style=style,
                               layout=Layout(width='500px'))
    amp_2_slider = FloatSlider(value=0.4, min=0.0, max=1.0, step=0.1, description='Input 2 amplitude', style=style,
                               layout=Layout(width='500px'))
    phase_shift_slider = FloatSlider(value=0, min=0, max=2 * np.pi, step=np.pi / 4, description='Phase shift',
                                     style=style, layout=Layout(width='500px'))
    w1_init_slider = FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description='w1 initial', style=style,
                                 layout=Layout(width='500px'))
    w2_init_slider = FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description='w2 initial', style=style,
                                 layout=Layout(width='500px'))
    use_decay_checkbox = IntSlider(value=0, min=0, max=1, description='Use weight decay', style=style, layout=Layout(width='500px'))
    update_button = Button(description="Update", button_style='success')

    update_button.on_click(lambda x: update_plot(eta=eta_slider.value,
                                                 tau=tau_slider.value,
                                                 epsilon=epsilon_slider.value,
                                                 simulation_time=simulation_time_slider.value,
                                                 p=p_slider.value,
                                                 freq_1=freq_1_slider.value,
                                                 freq_2=freq_2_slider.value,
                                                 amp_1=amp_1_slider.value,
                                                 amp_2=amp_2_slider.value,
                                                 phase_shift=phase_shift_slider.value,
                                                 w1_init=w1_init_slider.value,
                                                 w2_init=w2_init_slider.value,
                                                 use_decay=bool(use_decay_checkbox.value)
                                                 ))

    # Setup the tabs
    tab = widgets.Tab()
    tab.children = [
        VBox([freq_1_slider, freq_2_slider, amp_1_slider, amp_2_slider, phase_shift_slider,
              w1_init_slider, w2_init_slider, use_decay_checkbox, update_button, output_plot]),
        VBox([eta_slider, tau_slider, epsilon_slider, simulation_time_slider, p_slider, output_plot])]
    tab.titles = ('Simulation', 'Configuration')

    display(tab)
    update_button.click()
