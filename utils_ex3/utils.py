import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, VBox, Output
import ipywidgets as widgets

def iplot_InoF_model():
    # Simulation parameters
    simtime = 100  # Total simulation time in milliseconds
    dt = 1         # Time step in milliseconds
    time = np.arange(0, simtime + dt, dt)

    # Function to generate input times based on frequency
    def generate_input_times(frequency):
        period = 1000 / frequency  # Converting frequency (Hz) to period (ms)
        input_times = np.arange(0, simtime, period)
        input_times = input_times.astype(int)
        return input_times.tolist()

    # Output widget for displaying the plot
    output_plot = Output()

    # Update the plot (triggered by the button)
    def update_plot(frequency=10, synaptic_weight=5, length=1):
        rise_time = 1
        # Membrane potential starting at -70 mV
        V = np.full(len(time), -70.0)

        # Input raster plot data
        raster = np.zeros(len(time))

        # Generate input times based on frequency
        input_times = generate_input_times(frequency)

        # Compute synaptic delay based on length
        # Assuming propagation velocity v = 0.1 mm/ms (100 mm/s)
        v = 0.1
        delay = length / v  # Delay in milliseconds
        delay = int(round(delay))  # Convert to integer milliseconds

        # Simulation loop
        for t_idx, t in enumerate(time):
            if t_idx == 0:
                continue  # Skip the first time step

            # Carry over membrane potential from previous timestep
            V[t_idx] = V[t_idx - 1]

            # Add contribution of input with its weight and delay over rise_time
            for spike_time in input_times:
                input_time = spike_time + delay
                if input_time <= t < input_time + rise_time:
                    # Calculate the fraction of synaptic weight to add at this time
                    fraction = (t - input_time + dt) / rise_time
                    delta_V = synaptic_weight * (fraction)
                    V[t_idx] = V[t_idx - 1] + delta_V
                elif t == input_time + rise_time:
                    # After rise_time, ensure the total synaptic weight has been added
                    V[t_idx] = V[t_idx - 1] + synaptic_weight - synaptic_weight * (rise_time - dt) / rise_time
                # Mark delayed spike in raster plot at input_time
                if t == input_time:
                    raster[t_idx] = 1

        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)

            # Plot the results
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            # Plot membrane potential
            ax1.plot(time, V, color='b')
            ax1.set_title("Integrate-and-No-Fire Neuron Model")
            ax1.set_ylabel("Membrane Potential (mV)")
            ax1.set_ylim(-70, 100)
            ax1.set_yticks(np.arange(-70, 110, 20))
            ax1.grid(True)

            # Plot input spikes as raster
            # Original spike times in black
            for spike_time in input_times:
                ax2.scatter(spike_time, 0, marker='|', color='black', s=100, label='Original input time' if spike_time == input_times[0] else "")
            # Delayed input times in grey
            for spike_time in input_times:
                delayed_time = spike_time + delay
                if delayed_time <= simtime:
                    ax2.scatter(delayed_time, 0, marker='|', color='silver', s=100, label='Delayed input at soma' if spike_time == input_times[0] else "")

            ax2.set_title("Synaptic Input Raster Plot")
            ax2.set_xlabel("Time (ms)")
            ax2.set_xticks(np.arange(0, simtime + 1, 10))
            ax2.set_yticks([0])
            ax2.set_yticklabels(['Synapse'])
            ax2.set_ylim(-1, 1)
            ax2.grid(True)

            # Adding legend
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys())

            plt.tight_layout()
            plt.show()

    # Create sliders for synaptic weight, frequency, and length
    style = {'description_width': 'initial'}
    synaptic_weight_slider = FloatSlider(min=1, max=100, step=1, value=5, description='Synaptic Weight (mV):', style=style)
    frequency_slider = FloatSlider(min=1, max=200, step=1, value=10, description='Input Frequency (Hz):', style=style)
    length_slider = FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Synaptic Length (mm):', style=style)

    # Create the button to trigger the plot update
    plot_button = Button(description="Update Plot", button_style='success')

    # Callback function to update plot when the button is clicked
    def on_button_click(b):
        update_plot(frequency=frequency_slider.value, synaptic_weight=synaptic_weight_slider.value, length=length_slider.value)

    # Link button to callback function
    plot_button.on_click(on_button_click)

    # Create a vertical box to hold sliders, the button, and the output plot
    ui = VBox([frequency_slider, synaptic_weight_slider, length_slider, plot_button, output_plot])

    display(ui)

    update_plot(frequency=frequency_slider.value, synaptic_weight=synaptic_weight_slider.value, length=length_slider.value)

def iplot_Integrate_and_Fire_model():
    # Simulation parameters
    simtime = 100  # Total simulation time in milliseconds
    dt = 1         # Time step in milliseconds
    time = np.arange(0, simtime, dt)

    # Function to generate input times based on frequency
    def generate_input_times(frequency):
        period = 1000 / frequency  # Convert frequency (Hz) to period (ms)
        input_times = np.arange(0, simtime, period)
        input_times = input_times.astype(int)
        return input_times.tolist()

    # Output widget for displaying the plot
    output_plot = Output()

    # Function to update the plot
    def update_plot(frequency=10, synaptic_weight=5, length=1, threshold=20, refractory_scale=0.1):
        # Membrane potential, starting from -70 mV
        V = np.full(len(time), -70.0)

        # Spike train of the neuron
        spikes = np.zeros(len(time))

        # Refractory period variables
        refractory_time_remaining = 0  # Time remaining in refractory period
        refractory_regions = []  # List to store refractory start and end times

        # Generate input times based on frequency
        input_times = generate_input_times(frequency)

        # Compute synaptic delay based on length
        # Assume propagation velocity v = 0.1 mm/ms (100 mm/s)
        v = 0.1  # Propagation velocity in mm/ms
        delay = length / v  # Delay in milliseconds
        delay = int(round(delay))  # Convert to integer milliseconds

        # Simulation loop
        for t_idx, t in enumerate(time):
            if t_idx == 0:
                continue  # Skip the first time step

            # Check if neuron is in refractory period
            if refractory_time_remaining > 0:
                refractory_time_remaining -= dt
                V[t_idx] = -70  # Membrane potential remains at resting value
                continue  # Skip to next time step

            # Carry over membrane potential from previous timestep
            V[t_idx] = V[t_idx - 1]

            # Add contribution of input with its weight and delay
            for spike_time in input_times:
                input_time = spike_time + delay
                if t == input_time:
                    V[t_idx] += synaptic_weight

            # Check for firing
            if V[t_idx] >= threshold:
                spikes[t_idx] = 1  # Record spike

                # **Record peak membrane potential before reset**:
                V_peak = V[t_idx]

                # Adaptive refractory period: longer period for higher V_peak
                refractory_time_remaining = refractory_scale * V_peak
                refractory_regions.append((t, t + refractory_time_remaining))  # Store start and end of refractory period

                # Reset after spike (but after overshooting the threshold)
                V[t_idx] = V_peak  # Show overshoot at the threshold crossing

                # In the next time step, the potential will be reset to resting value (-70 mV)
                if t_idx + 1 < len(time):
                    V[t_idx + 1] = -70

        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)

            # Plot the results
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Plot membrane potential with fixed y-axis limits
            axs[0].plot(time, V, color='b')
            axs[0].set_title("Integrate-and-Fire Neuron Model with Adaptive Refractory Period")
            axs[0].set_ylabel("Membrane Potential (mV)")
            axs[0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            axs[0].set_ylim(-70, 100)  # Set static y-axis limits
            axs[0].set_yticks(np.arange(-70, 110, 20))  # Explicitly set y-ticks including -70
            axs[0].grid(True)

            # Highlight refractory period with shading
            for (start, end) in refractory_regions:
                axs[0].axvspan(start, end, color='yellow', alpha=0.3, label='Refractory Period' if start == refractory_regions[0][0] else "")

            axs[0].legend()

            # Plot neuron spikes
            axs[1].plot(time, spikes, color='k')
            axs[1].set_title("Neuron Spike Train")
            axs[1].set_ylabel("Spikes")
            axs[1].set_ylim(-0.1, 1.1)
            axs[1].grid(True)

            # Plot input spikes as raster
            # Original spike times in black
            for spike_time in input_times:
                axs[2].scatter(spike_time, 0, marker='|', color='black', s=100, label='Original input time' if spike_time == input_times[0] else "")
            # Delayed input times in grey
            for spike_time in input_times:
                delayed_time = spike_time + delay
                if delayed_time < simtime:
                    axs[2].scatter(delayed_time, 0, marker='|', color='silver', s=100, label='Delayed input at soma' if spike_time == input_times[0] else "")

            axs[2].set_title("Synaptic Input Raster Plot (Black: Original, Grey: Delayed)")
            axs[2].set_xlabel("Time (ms)")
            axs[2].set_xticks(np.arange(0, simtime + 1, 10))
            axs[2].set_yticks([0])
            axs[2].set_yticklabels(['Synapse'])
            axs[2].set_ylim(-1, 1)
            axs[2].grid(True)

            # Adding legend
            handles, labels = axs[2].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[2].legend(by_label.values(), by_label.keys())

            plt.tight_layout()
            plt.show()

    # Create sliders for synaptic weight, frequency, length, threshold, and refractory scale
    style = {'description_width': 'initial'}
    synaptic_weight_slider = FloatSlider(min=0, max=100, step=1, value=5, description='Synaptic Weight (mV):', style=style)
    frequency_slider = FloatSlider(min=1, max=1000, step=1, value=10, description='Input Frequency (Hz):', style=style)
    length_slider = FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Synaptic Length (mm):', style=style)
    threshold_slider = FloatSlider(min=1, max=100, step=1, value=20, description='Threshold (mV):', style=style)
    refractory_scale_slider = FloatSlider(min=0.01, max=1.0, step=0.01, value=0.1, description='Refractory Scale Factor:', style=style)

    # Create the button to trigger the plot update
    plot_button = Button(description="Update Plot", button_style='success')

    # Callback function to update plot when the button is clicked
    def on_button_click(b):
        update_plot(frequency=frequency_slider.value, synaptic_weight=synaptic_weight_slider.value, length=length_slider.value, threshold=threshold_slider.value, refractory_scale=refractory_scale_slider.value)

    # Link button to callback function
    plot_button.on_click(on_button_click)

    # Create a vertical box to hold sliders, the button, and the output plot
    ui = VBox([frequency_slider, synaptic_weight_slider, length_slider, threshold_slider, refractory_scale_slider, plot_button, output_plot])

    display(ui)

    update_plot(frequency=frequency_slider.value, synaptic_weight=synaptic_weight_slider.value, length=length_slider.value, threshold=threshold_slider.value, refractory_scale=refractory_scale_slider.value)


def iplot_Leaky_Integrate_and_Fire_model():
    # Simulation parameters
    simtime = 100  # Total simulation time in milliseconds
    dt = 1         # Time step in milliseconds
    time = np.arange(0, simtime, dt)

    # Function to generate input times based on frequency
    def generate_input_times(frequency):
        period = 1000 / frequency  # Convert frequency (Hz) to period (ms)
        input_times = np.arange(0, simtime, period)
        input_times = input_times.astype(int)
        return input_times.tolist()

    # Create an Output widget for displaying the plot
    output_plot = Output()

    # Function to update the plot
    def update_plot(frequency=10, synaptic_weight=5, length=1, threshold=20, tau=10, refractory_scale=0.1):
        # Membrane potential, starting from -70 mV
        V = np.full(len(time), -70.0)

        # Spike train of the neuron
        spikes = np.zeros(len(time))

        # Refractory period variables
        refractory_time_remaining = 0  # Time remaining in refractory period
        refractory_regions = []  # List to store refractory start and end times

        # Generate input times based on frequency
        input_times = generate_input_times(frequency)

        # Compute synaptic delay based on length
        v = 0.1  # Propagation velocity in mm/ms (100 mm/s)
        delay = length / v  # Delay in milliseconds
        delay = int(round(delay))  # Convert to integer milliseconds

        # Simulation loop
        for t_idx, t in enumerate(time):
            if t_idx == 0:
                continue  # Skip the first time step

            # Check if neuron is in refractory period
            if refractory_time_remaining > 0:
                refractory_time_remaining -= dt
                V[t_idx] = -70  # Membrane potential remains at resting value
                continue  # Skip to next time step

            # Apply leaky decay to membrane potential
            V[t_idx] = V[t_idx - 1] - (V[t_idx - 1] / tau)

            # Add contribution of input with its weight and delay when the delayed spike reaches the soma
            for spike_time in input_times:
                input_time = spike_time + delay
                if t == input_time:
                    # The delayed spike reaches the soma, membrane potential starts increasing
                    V[t_idx] += synaptic_weight  # Add the full synaptic weight when spike reaches soma

            # Check for firing
            if V[t_idx] >= threshold:
                spikes[t_idx] = 1  # Record spike

                # Record peak membrane potential before reset
                V_peak = V[t_idx]

                # Set refractory period as a function of membrane potential (scaling factor)
                refractory_time_remaining = refractory_scale * V_peak
                refractory_regions.append((t, t + refractory_time_remaining))  # Store start and end of refractory period

                # In the next time step, the potential will be reset to -70 mV (resting potential)
                if t_idx + 1 < len(time):
                    V[t_idx + 1] = -70

        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)

            # Plot the results
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Plot membrane potential with fixed y-axis limits from -70 to 100
            axs[0].plot(time, V, color='b')
            axs[0].set_title("Leaky Integrate-and-Fire Neuron Model")
            axs[0].set_ylabel("Membrane Potential (mV)")
            axs[0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            axs[0].set_ylim(-70, 100)  # Set static y-axis limits from -70 to 100 mV
            axs[0].set_yticks(np.arange(-70, 110, 20))  # Explicitly set y-ticks including -70
            axs[0].grid(True)

            # Highlight refractory period with shading
            for (start, end) in refractory_regions:
                axs[0].axvspan(start, end, color='yellow', alpha=0.3, label='Refractory Period' if start == refractory_regions[0][0] else "")

            axs[0].legend()

            # Plot neuron spikes
            axs[1].plot(time, spikes, color='k')
            axs[1].set_title("Neuron Spike Train")
            axs[1].set_ylabel("Spikes")
            axs[1].set_ylim(-0.1, 1.1)
            axs[1].grid(True)

            # Plot input spikes as raster
            # Original spike times in black
            for spike_time in input_times:
                axs[2].scatter(spike_time, 0, marker='|', color='black', s=100, label='Original input time' if spike_time == input_times[0] else "")
            # Delayed input times in grey
            for spike_time in input_times:
                delayed_time = spike_time + delay
                if delayed_time < simtime:
                    axs[2].scatter(delayed_time, 0, marker='|', color='silver', s=100, label='Delayed input at soma' if spike_time == input_times[0] else "")

            axs[2].set_title("Synaptic Input Raster Plot")
            axs[2].set_xlabel("Time (ms)")
            axs[2].set_xticks(np.arange(0, simtime + 1, 10))
            axs[2].set_yticks([0])
            axs[2].set_yticklabels(['Synapse'])
            axs[2].set_ylim(-1, 1)
            axs[2].grid(True)

            # Adding legend
            handles, labels = axs[2].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[2].legend(by_label.values(), by_label.keys())

            plt.tight_layout()
            plt.show()

    # Create sliders for synaptic weight, frequency, length, threshold, tau (time constant), and refractory scale
    style = {'description_width': 'initial'}
    synaptic_weight_slider = FloatSlider(min=0, max=100, step=1, value=5, description='Synaptic Weight (mV):', style=style)
    frequency_slider = FloatSlider(min=1, max=100, step=1, value=10, description='Input Frequency (Hz):', style=style)
    length_slider = FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Synaptic Length (mm):', style=style)
    threshold_slider = FloatSlider(min=1, max=100, step=1, value=20, description='Threshold (mV):', style=style)
    tau_slider = FloatSlider(min=1, max=100, step=1, value=10, description='Time Constant (ms):', style=style)
    refractory_scale_slider = FloatSlider(min=0.01, max=1.0, step=0.01, value=0.1, description='Refractory Scale:', style=style)

    # Create the button to trigger the plot update
    plot_button = Button(description="Update Plot", button_style='success')

    # Callback function to update plot when the button is clicked
    def on_button_click(b):
        update_plot(frequency=frequency_slider.value, synaptic_weight=synaptic_weight_slider.value, length=length_slider.value, threshold=threshold_slider.value, tau=tau_slider.value, refractory_scale=refractory_scale_slider.value)

    # Link button to callback function
    plot_button.on_click(on_button_click)

    # Create a vertical box to hold sliders, the button, and the output plot
    ui = VBox([frequency_slider, synaptic_weight_slider, length_slider, threshold_slider, tau_slider, refractory_scale_slider, plot_button, output_plot])

    display(ui)

    update_plot(frequency=frequency_slider.value, synaptic_weight=synaptic_weight_slider.value, length=length_slider.value, threshold=threshold_slider.value, tau=tau_slider.value, refractory_scale=refractory_scale_slider.value)