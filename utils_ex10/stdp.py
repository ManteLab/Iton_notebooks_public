import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Button, FloatSlider, Layout, Output, VBox
from scipy.interpolate import interp1d


def iplot_stdp_model():
    output_plot = Output()
    submitted_values = {'t_pos': [], 'weight_pos': [], 't_neg': [], 'weight_neg': [], 'last_update': None}
    config = {}

    def reset_values():
        submitted_values['t_pos'] = []
        submitted_values['weight_pos'] = []
        submitted_values['t_neg'] = []
        submitted_values['weight_neg'] = []
        submitted_values['last_update'] = None

    def update_config(A_plus, A_minus, tau_pos, tau_neg):
        config['A_plus'] = A_plus
        config['A_minus'] = A_minus
        config['tau_pos'] = tau_pos
        config['tau_neg'] = tau_neg
        reset_values()

    def update_plot(
            t: float,
    ):
        """
        Update the plot (triggered by the button)

        Args:
            t: Time of spike of neuron i relative to neuron j
        """

        def calc_weight_change(A, t, tau):
            """
            Calculate weight change using exponential decay fuanction
            Args:
                A: Max. Magnitude of weight change
                t: time difference between pre- and post-synaptic spike
                tau: Time constant

            Returns:

            """
            return A * np.exp(t / tau)

        if t is not None:
            t_delta = t
            pos_weight_change = calc_weight_change(config['A_plus'], t_delta, config['tau_pos'])
            neg_weight_change = calc_weight_change(config['A_minus'], -t_delta, config['tau_neg'])

            if t_delta < 0:
                submitted_values['t_pos'].append(t_delta)
                submitted_values['weight_pos'].append(pos_weight_change)
                submitted_values['last_update'] = 'pos'
            else:
                submitted_values['t_neg'].append(t_delta)
                submitted_values['weight_neg'].append(-neg_weight_change)
                submitted_values['last_update'] = 'neg'

        with output_plot:
            output_plot.clear_output(wait=True)

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=False)

            # Plot 1: Formula for STDP
            ax1.set_title("Calculation of STDP")
            if submitted_values['last_update'] == 'pos':
                calcs = f"${config['A_plus']} \cdot e^" + r"{" + f"{t_delta / config['tau_pos']:.2f}" + r"}" + (f"={submitted_values['weight_pos'][-1]:.2f}$")
                ax1.text(.5, .5, r"$\Delta w_{ij} = A_+ \cdot e^{\frac{t}{\tau_{+}}}=$" + calcs,
                         horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            elif submitted_values['last_update'] == 'neg':
                calcs = f"${config['A_minus']} \cdot - e^" + r"{" + f"{-t_delta / config['tau_neg']:.2f}" + r"}" + (f"= {submitted_values['weight_neg'][-1]:.2f}$")
                ax1.text(.5, .5, r"$\Delta w_{ij} = A_- \cdot - e^{\frac{t}{\tau_{-}}}=$" + calcs,
                         horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            ax1.grid(False)
            ax1.axis('off')

            # Plot 2: Neuron Spike Times
            time = np.linspace(-1, 1, 2000)

            if t is not None:
                spikes = np.zeros_like(time)
                spikes[np.abs(time) < 0.01] = 1
                ax2.plot(time, spikes, color='black', label='Spike neuron j')

                if t > 0:
                    spikes = np.zeros_like(time)
                    spikes[np.abs(time-t) < 0.01] = 1
                    ax2.plot(time, spikes, color='blue', label='Spike neuron i (i before j)')

                else:
                    spikes = np.zeros_like(time)
                    spikes[np.abs(time-t) < 0.01] = 1
                    ax2.plot(time, spikes, color='red', label='Spike neuron i (j before i)')
            else:
                ax2.plot(time, np.zeros_like(time), color='black', label='No spikes yet...')

            ax2.set_title("Neuron Spike Times")
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Spike")
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlim(-1.1, 1.1)
            ax2.grid(True)
            ax2.legend()

            # Plot 3: Weight change based on time
            ax3.scatter(submitted_values['t_pos'], submitted_values['weight_pos'], color='red', marker="o",
                        label='Pre-synaptic spike')
            ax3.scatter(submitted_values['t_neg'], submitted_values['weight_neg'], color='blue', marker="o",
                        label='Post-synaptic spike')

            # Interpolation for pre-synaptic spikes (red)
            if len(submitted_values['t_pos']) > 1:  # Ensure there are enough points for interpolation
                interp_pos = interp1d(submitted_values['t_pos'], submitted_values['weight_pos'], kind='linear')
                t_pos_smooth = np.linspace(min(submitted_values['t_pos']), max(submitted_values['t_pos']), 300)
                weight_pos_smooth = interp_pos(t_pos_smooth)
                ax3.plot(t_pos_smooth, weight_pos_smooth, color='red', linestyle='--',
                         label='Pre-synaptic interpolation')

            # Interpolation for post-synaptic spikes (blue)
            if len(submitted_values['t_neg']) > 1:
                interp_neg = interp1d(submitted_values['t_neg'], submitted_values['weight_neg'], kind='linear')
                t_neg_smooth = np.linspace(min(submitted_values['t_neg']), max(submitted_values['t_neg']), 300)
                weight_neg_smooth = interp_neg(t_neg_smooth)
                ax3.plot(t_neg_smooth, weight_neg_smooth, color='blue', linestyle='--',
                         label='Post-synaptic interpolation')

            if submitted_values['last_update'] == 'pos':
                ax3.scatter(submitted_values['t_pos'][-1], submitted_values['weight_pos'][-1], color='black',
                            marker="x", label='Last spike')
            elif submitted_values['last_update'] == 'neg':
                ax3.scatter(submitted_values['t_neg'][-1], submitted_values['weight_neg'][-1], color='black',
                            marker="x", label='Last spike')

            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-config['A_minus'], config['A_plus'])

            ax3.set_title("STDP")
            ax3.set_ylabel("Weight change")
            ax3.set_xlabel("Time difference (pre-post)")

            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            plt.show()

    style = {'description_width': 'initial'}

    # First tab contains the configuration
    A_pos_slider = FloatSlider(min=0, max=1, step=0.01, value=0.1, description='A_+:', style=style,
                               layout=Layout(width='500px'))
    A_neg_slider = FloatSlider(min=0, max=1, step=0.01, value=0.1, description='A_-:', style=style,
                               layout=Layout(width='500px'))
    tau_pos_slider = FloatSlider(min=0.01, max=3, step=0.01, value=0.3, description='tau_pos:', style=style,
                                 layout=Layout(width='500px'))
    tau_neg_slider = FloatSlider(min=0.01, max=3, step=0.01, value=0.3, description='tau_neg:', style=style,
                                 layout=Layout(width='500px'))
    config_update_button = Button(description="Update Config (will reset plot)", button_style='danger')

    # Second tab contains the input values
    t_slider = FloatSlider(min=-1, max=1, step=0.01, value=0.0, description='Firing time neuron i (relative to neuron j)', style=style,
                               layout=Layout(width='500px'))
    submit_button = Button(description="Submit", button_style='success')

    # Setup the tabs
    tab = widgets.Tab()
    tab.children = [
        VBox([A_pos_slider, A_neg_slider, tau_pos_slider, tau_neg_slider, config_update_button, output_plot]),
        VBox([ t_slider, submit_button, output_plot])]
    tab.titles = ('Configuration', 'Add value')

    # set the default configuration
    A_pos_slider.value = 0.1
    A_neg_slider.value = 0.1
    tau_pos_slider.value = 0.3
    tau_neg_slider.value = 0.3

    # set some default values for the input
    t_slider.value = -0.1

    # store the configuration
    update_config(A_pos_slider.value, A_neg_slider.value, tau_pos_slider.value, tau_neg_slider.value)

    # Button triggers
    def button_config_on_click_action(_=None):
        update_config(A_pos_slider.value, A_neg_slider.value, tau_pos_slider.value, tau_neg_slider.value)
        update_plot(None)

    def button_submit_on_click_action(_=None):
        # print("Submit button triggered")
        update_plot(t_slider.value)

    config_update_button.on_click(button_config_on_click_action)
    submit_button.on_click(button_submit_on_click_action)

    display(tab)

    update_plot(None)
