import ipywidgets as widgets
from ipywidgets import Layout, interact

style = {'description_width': 'initial'}


def simulate_lif_model():
    # Define parameters for the LIF neuron
    V_th = -55.  # Spike threshold [mV]
    V_reset = -75.  # Reset potential [mV]
    tau_m = 10.  # Membrane time constant [ms]
    g_L = 10.  # Leak conductance [nS]
    V_init = -65.  # Initial potential [mV]
    V_L = -75.  # Leak reversal potential [mV]
    dt = 0.1  # Time step [ms]
    T = 100.  # Simulation duration [ms]
    tref = 2.  # Refractory period [ms]

    # Generate time range and input current
    range_t = np.arange(0, T, dt)
    I = 300. * np.ones(range_t.size)  # Constant input current [pA]

    # Define the LIF simulation function
    def simulate(V_th, V_reset, tau_m, g_L, V_init, V_L, dt, range_t, tref, I):
        Lt = range_t.size

        # Initialize voltage and spike times
        v = np.zeros(Lt)
        v[0] = V_reset  # V_init # TODO: Optionally start from a different initial potential
        tr = 0.  # Refractory counter
        rec_spikes = []  # Record spike times

        # Simulate the LIF dynamics
        for it in range(Lt - 1):
            if tr > 0:
                # Refractory period: voltage clamped to reset potential
                v[it] = V_reset
                tr -= 1
            elif v[it] >= V_th:
                # Spike condition: reset voltage and record spike
                rec_spikes.append(it)
                v[it] = V_reset
                tr = tref / dt  # Start refractory period

            # Update membrane potential
            dv = (-(v[it] - V_L) + I[it] / g_L) * (dt / tau_m)
            v[it + 1] = v[it] + dv

        rec_spikes = np.array(rec_spikes) * dt
        return v, rec_spikes

    def lif_interactive(V_th=-55., V_reset=-75., tau_m=10., I_amplitude=300., tref=2.):
        I = I_amplitude * np.ones(range_t.size)
        v, rec_spikes = simulate(V_th, V_reset, tau_m, g_L, V_init, V_L, dt, range_t, tref, I)

        plt.figure(figsize=(12, 6))
        plt.plot(range_t, v, label="Membrane Potential (V)")
        plt.plot(rec_spikes, np.ones_like(rec_spikes) * V_th, 'o', color='r', label="Spikes")
        plt.axhline(V_th, color='gray', linestyle='--', label="Threshold")
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.title("LIF Neuron Simulation Without Adaptation")
        plt.legend()
        plt.grid()
        plt.show()

    # Create sliders for interactivity
    interact(
        lif_interactive,
        V_th=widgets.FloatSlider(min=-70, max=-40, step=1, value=-55, description="Threshold (V_th)",
                                 style=style, layout=Layout(width='500px')),
        V_reset=widgets.FloatSlider(min=-80, max=-60, step=1, value=-75, description="Reset Potential (V_reset)",
                                    style=style, layout=Layout(width='500px')),
        tau_m=widgets.FloatSlider(min=1, max=20, step=1, value=10, description="Membrane Time Constant (tau_m)",
                                  style=style, layout=Layout(width='500px')),
        I_amplitude=widgets.FloatSlider(min=100, max=800, step=50, value=300, description="Input Current (I)",
                                        style=style, layout=Layout(width='500px')),
        tref=widgets.FloatSlider(min=1, max=5, step=1, value=2, description="Refractory Period (tref)",
                                 style=style, layout=Layout(width='500px'))
    )


from ipywidgets import FloatSlider, VBox
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Output


def adaptation_model():
    output_plot = Output()
    output_image = Output()

    def simulate(V_th, V_reset, tau_m, g_L, V_init, V_L, dt, range_t, tau_w, b, tref, I, use_adaptation):
        Lt = range_t.size

        # Initialize voltage, adaptation current, and spike times
        v = np.zeros(Lt)
        w = np.zeros(Lt)
        v[0] = V_init
        tr = 0.
        rec_spikes = []  # record spike times

        # simulate the LIF dynamics
        for it in range(Lt - 1):
            if tr > 0:
                v[it] = V_reset
                tr -= 1
            elif v[it] >= V_th:  # reset voltage and record spike event
                rec_spikes.append(it)
                v[it] = V_reset
                w[it] += b  # Increment adaptation current
                tr = tref / dt

            if use_adaptation:
                # Calculate the increment of the membrane potential
                dw = (-w[it] + (b if v[it] >= V_th else 0)) * (dt / tau_w)
                w[it + 1] = w[it] + dw  # Update adaptation current

                dv = (-(v[it] - V_L) + (I[it] - w[it]) / g_L) * (dt / tau_m)
                v[it + 1] = v[it] + dv  # Update membrane potential
            else:
                # calculate the increment of the membrane potential
                dv = (-(v[it] - V_L) + I[it] / g_L) * (dt / tau_m)

                # update the membrane potential
                v[it + 1] = v[it] + dv

        rec_spikes = np.array(rec_spikes) * dt
        return v, rec_spikes, w

    def generate_gray_pattern(duration, range_t, dt, min_gray=0, max_gray=2000):
        gray_pattern = np.zeros(range_t.size)

        # Generate continuous gray values using a sinusoidal pattern
        for i, t in enumerate(range_t):
            gray_value = min_gray + (max_gray - min_gray) * (0.5 * np.sin(2 * np.pi * t / duration) + 0.5)
            gray_pattern[i] = gray_value

        return gray_pattern

    def update_plot(V_th=-60.,  # spike threshold [mV]
                    V_reset=-75.,  # reset potential [mV]
                    tau_m=15.,  # membrane time constant [ms]
                    g_L=8.,  # leak conductance [nS]
                    V_init=-65.,  # initial potential [mV]
                    V_L=-75.,  # leak reversal potential [mV]
                    dt=0.02,  # simulation time step [ms]
                    T=250.,  # total duration of simulation [ms]
                    tau_w=40.,  # adaptation time constant [ms]
                    b=120,  # increment to adaptation current at each spike [pA]
                    tref=2.,  # refractory time (ms)
                    duration=100.,
                    input_current_factor=1.0):  # Slider factor for input current
        range_t = np.arange(0, T, dt)

        # Adjust input current based on the slider value
        I = generate_gray_pattern(duration, range_t, dt, min_gray=0,
                                  max_gray=input_current_factor * 1500)  # Pass dt here

        v, rec_spikes, _ = simulate(V_th, V_reset, tau_m, g_L, V_init, V_L, dt, range_t, tau_w, b, tref, I,
                                    use_adaptation=False)
        v_adapted, rec_spikes_adapted, w_adapted = simulate(V_th, V_reset, tau_m, g_L, V_init, V_L, dt, range_t, tau_w,
                                                            b, tref, I, use_adaptation=True)

        with output_plot:
            output_plot.clear_output(wait=True)
            fig, axs = plt.subplots(3, 1, figsize=(8, 4), sharex=True)

            axs[0].plot(range_t, I, label='Input current (pA)')
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('I (pA)')
            axs[0].legend()
            axs[0].grid()

            axs[1].plot(rec_spikes, np.ones_like(rec_spikes), '|', label='spikes without adaptation')
            axs[1].plot(rec_spikes_adapted, np.ones_like(rec_spikes_adapted) * 2, '|', label='spikes with adaptation')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Output')
            axs[1].set_yticks([1, 2])
            axs[1].set_yticklabels(['Without Adaptation', 'With Adaptation'])
            axs[1].set_ylim([0.1, 2.9])
            axs[1].grid()

            axs[2].plot(range_t, w_adapted, label='Adaptation Current (pA)')
            axs[2].set_xlabel('Time (ms)')
            axs[2].set_ylabel('w (pA)')
            axs[2].grid()

            plt.tight_layout()
            plt.show()

    style = {'description_width': 'initial'}

    duration_slider = FloatSlider(value=35., min=5., max=50., step=10., description='Pattern Duration (ms)',
                                  style=style, layout=Layout(width='500px'))

    input_current_slider = FloatSlider(value=1.3, min=0.5, max=2.0, step=0.1, description='Input Current Factor',
                                       style=style, layout=Layout(width='500px'))

    b_slider = FloatSlider(value=190., min=40., max=200., step=5., description='Adaptation Increment b (pA)',
                           style=style, layout=Layout(width='500px'))

    tau_w_slider = FloatSlider(value=90., min=5., max=100., step=5., description='Adaptation Time Constant tau_w (ms)',
                               style=style, layout=Layout(width='500px'))

    # Create interactive plot updates
    def on_slider_change(change):
        update_plot(
            duration=duration_slider.value,
            input_current_factor=input_current_slider.value,
            b=b_slider.value,
            tau_w=tau_w_slider.value
        )

    duration_slider.observe(on_slider_change, names='value')
    input_current_slider.observe(on_slider_change, names='value')
    b_slider.observe(on_slider_change, names='value')
    tau_w_slider.observe(on_slider_change, names='value')

    box = VBox([duration_slider, input_current_slider, b_slider, tau_w_slider, output_plot])
    output_plot.layout.height = '350px'
    display(box)
    on_slider_change(None)
