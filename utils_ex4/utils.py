import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, VBox, Output, Layout
import ipywidgets as widgets
from IPython.display import display

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
    synaptic_weight_slider = FloatSlider(min=1, max=100, step=1, value=5, description='Synaptic Weight (mV):', style=style, layout=Layout(width='500px'))
    frequency_slider = FloatSlider(min=1, max=200, step=1, value=10, description='Input Frequency (Hz):', style=style, layout=Layout(width='500px'))
    length_slider = FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Distance: Synapse-Soma (mm):', style=style, layout=Layout(width='500px'))

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
        v = 0.1  # Propagation velocity in mm/ms (100 mm/s)
        delay = length / v  # Delay in milliseconds
        delay = int(round(delay))  # Convert to integer 
        just_fired = False

        # Adding functions to make up for the lack of 'goto' — because Python refuses to let us YOLO-jump like it's C++ or Golang :|
        def add_input_contrib():
          for spike_time in input_times:
              input_time = spike_time + delay
              if t == input_time:
                  V[t_idx] += synaptic_weight  # Add the full synaptic weight when spike reaches soma

        # Simulation loop
        for t_idx, t in enumerate(time):
            if t_idx == 0:
                continue  # Skip the first time step

            # Check if neuron is in refractory period
            if refractory_time_remaining > 0:
                refractory_time_remaining -= dt
                if just_fired:
                    V[t_idx] = -70  # Membrane potential reset to resting value
                    just_fired = False
                else:
                    # Carry over membrane potential from previous timestep
                    V[t_idx] = V[t_idx - 1]
                    # Add contribution of input with its weight and delay when the delayed spike reaches the soma
                    add_input_contrib()
                continue  # Skip to next time step

            # Carry over membrane potential from previous timestep
            V[t_idx] = V[t_idx - 1]
            # Add contribution of input with its weight and delay
            add_input_contrib()

            # Check for firing
            if V[t_idx] >= threshold:
                spikes[t_idx] = 1  # Record spike
                just_fired = True

                # **Record peak membrane potential before reset**:
                V_peak = V[t_idx]

                # Adaptive refractory period: longer period for higher V_peak
                refractory_time_remaining = max(refractory_scale * abs(V_peak), 1)  # Ensure non-zero refractory period
                refractory_regions.append((t, t + refractory_time_remaining))  # Store start and end of refractory period


        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)

            # Plot the results
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Plot membrane potential with fixed y-axis limits
            axs[1].plot(time, V, color='b')
            axs[1].set_title("Integrate-and-Fire Neuron Model with Adaptive Refractory Period")
            axs[1].set_ylabel("Membrane Potential (mV)")
            axs[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            axs[1].set_ylim(-70, 100)  # Set static y-axis limits
            axs[1].set_yticks(np.arange(-70, 110, 20))  # Explicitly set y-ticks including -70
            axs[1].grid(True)

            # Highlight refractory period with shading
            for (start, end) in refractory_regions:
                axs[1].axvspan(start, end, color='yellow', alpha=0.3, label='Refractory Period' if start == refractory_regions[0][0] else "")

            axs[1].legend()

            # Plot neuron spikes
            axs[0].plot(time, spikes, color='k')
            axs[0].set_title("Neuron Spike Train")
            axs[0].set_ylabel("Spikes")
            axs[0].set_ylim(-0.1, 1.1)
            axs[0].grid(True)

            # Plot input spikes as raster
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
    synaptic_weight_slider = FloatSlider(min=0, max=100, step=1, value=5, description='Synaptic Weight (mV):', style=style, layout=Layout(width='500px'))
    frequency_slider = FloatSlider(min=1, max=1000, step=1, value=10, description='Input Frequency (Hz):', style=style, layout=Layout(width='500px'))
    length_slider = FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Distance: Synapse-Soma (mm):', style=style, layout=Layout(width='500px'))
    threshold_slider = FloatSlider(min=-60, max=100, step=1, value=20, description='Threshold (mV):', style=style, layout=Layout(width='500px'))
    refractory_scale_slider = FloatSlider(min=0.01, max=1.0, step=0.01, value=0.1, description='Refractory Scale Factor:', style=style, layout=Layout(width='500px'))

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
        
        # Resting potential
        resting_potential = -70.0

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
        just_fired = False

        # Adding functions to make up for the lack of 'goto' — because Python refuses to let us YOLO-jump like it's C++ or Golang :|
        def add_input_contrib():
          for spike_time in input_times:
              input_time = spike_time + delay
              if t == input_time:
                  V[t_idx] += synaptic_weight  # Add the full synaptic weight when spike reaches soma

        def add_mem_decay():
          V[t_idx] = V[t_idx - 1] + ((resting_potential - V[t_idx - 1]) / tau)

        # Simulation loop
        for t_idx, t in enumerate(time):
            if t_idx == 0:
                continue  # Skip the first time step

            # Check if neuron is in refractory period
            if refractory_time_remaining > 0:
                refractory_time_remaining -= dt
                if just_fired:
                  V[t_idx] = resting_potential  # Membrane potential reset to resting value
                  just_fired = False
                else:
                  # Apply leaky decay to membrane potential
                  add_mem_decay()
                  # Add contribution of input with its weight and delay when the delayed spike reaches the soma
                  add_input_contrib()
                continue  # Skip to next time step

            # Apply leaky decay to membrane potential
            add_mem_decay()

            # Add contribution of input with its weight and delay when the delayed spike reaches the soma
            add_input_contrib() 

            # Check for firing
            if V[t_idx] >= threshold:
                spikes[t_idx] = 1  # Record spike
                just_fired = True

                # Record peak membrane potential before reset
                V_peak = V[t_idx]

                # Set refractory period as a function of membrane potential (scaling factor)
                refractory_time_remaining = max(refractory_scale * abs(V_peak), 1)  # Ensure a non-zero refractory period
                refractory_regions.append((t, t + refractory_time_remaining))  # Store start and end of refractory period

                # In the next time step, the potential will be reset to -70 mV (resting potential)
                if t_idx + 1 < len(time):
                    V[t_idx + 1] = resting_potential

        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)

            # Plot the results
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Plot membrane potential with fixed y-axis limits from -70 to 100
            axs[1].plot(time, V, color='b')
            axs[1].set_title("Leaky Integrate-and-Fire Neuron Model")
            axs[1].set_ylabel("Membrane Potential (mV)")
            axs[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            axs[1].set_ylim(-70, 100)  # Set static y-axis limits from -70 to 100 mV
            axs[1].set_yticks(np.arange(-70, 110, 20))  # Explicitly set y-ticks including -70
            axs[1].grid(True)

            # Highlight refractory period with shading
            for (start, end) in refractory_regions:
                axs[1].axvspan(start, end, color='yellow', alpha=0.3, label='Refractory Period' if start == refractory_regions[0][0] else "")

            axs[1].legend()

            # Plot neuron spikes
            axs[0].plot(time, spikes, color='k')
            axs[0].set_title("Neuron Spike Train")
            axs[0].set_ylabel("Spikes")
            axs[0].set_ylim(-0.1, 1.1)
            axs[0].grid(True)

            # Plot input spikes as raster
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

            plt.show()

    # Create sliders for synaptic weight, frequency, length, threshold, tau (time constant), and refractory scale
    style = {'description_width': 'initial'}
    synaptic_weight_slider = FloatSlider(min=0, max=100, step=1, value=5, description='Synaptic Weight (mV):', style=style, layout=Layout(width='500px'))
    frequency_slider = FloatSlider(min=1, max=1000, step=1, value=10, description='Input Frequency (Hz):', style=style, layout=Layout(width='500px'))
    length_slider = FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Distance: Synapse-Soma (mm):', style=style, layout=Layout(width='500px'))
    threshold_slider = FloatSlider(min=-60, max=100, step=1, value=20, description='Threshold (mV):', style=style, layout=Layout(width='500px'))
    tau_slider = FloatSlider(min=1, max=100, step=1, value=10, description='Time Constant (ms):', style=style, layout=Layout(width='500px'))
    refractory_scale_slider = FloatSlider(min=0.01, max=1.0, step=0.01, value=0.1, description='Refractory Scale Factor:', style=style, layout=Layout(width='500px'))

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

def iplot_Leaky_Integrate_and_Fire_with_distributions():
    # Simulation parameters
    simtime = 100  # Total simulation time in milliseconds
    dt = 1         # Time step in milliseconds
    time = np.arange(0, simtime, dt)
    
    # Function to generate input times based on frequency
    def generate_input_times(frequency, num_inputs):
        period = 1000 / frequency  # Convert frequency (Hz) to period (ms)
        input_times = [np.arange(0, simtime, period) for _ in range(num_inputs)]
        return input_times

    # Output widget for plot display
    output_plot = Output()

    # Function to update the plot
    def update_plot(syn_lengths_mean=1, syn_lengths_std=0.5,
                    sync_mean=0.5, sync_std=0.1, rate_mean=10, rate_std=2,
                    weights_mean=5, weights_std=1, tau=10, threshold=20):
        
        num_synapses = 20
        refractory_scale = 0.2
        just_fired = False

        # Adding functions to make up for the lack of 'goto' — because Python refuses to let us YOLO-jump like it's C++ or Golang :|
        def add_mem_decay():
          V[t_idx] = V[t_idx - 1] + ((resting_potential - V[t_idx - 1]) / tau)

        # Adjust standard deviations based on synchronization mean
        syn_lengths_std_adjusted = syn_lengths_std * (1 - sync_mean)
        weights_std_adjusted = weights_std * (1 - sync_mean)
        
        # Ensure standard deviations are not zero
        syn_lengths_std_adjusted = max(syn_lengths_std_adjusted, 0.01)
        weights_std_adjusted = max(weights_std_adjusted, 0.01)
        
        # Sample synchronization factors from Gaussian distribution between 0 and 1
        sync_factor = np.random.normal(sync_mean, sync_std, num_synapses)
        sync_factor = np.clip(sync_factor, 0, 1)  # Ensure values are between 0 and 1

        # Sample synaptic lengths
        synaptic_lengths = np.random.normal(syn_lengths_mean, syn_lengths_std_adjusted, num_synapses)
        synaptic_lengths = np.clip(synaptic_lengths, 0.1, None)  # Ensure positive lengths

        # Sample synaptic weights
        synaptic_weights = np.random.normal(weights_mean, weights_std_adjusted, num_synapses)
        synaptic_weights = np.clip(synaptic_weights, 0.1, None)  # Ensure positive weights

        # Sample input rates
        input_rates = np.random.normal(rate_mean, rate_std, num_synapses)
        input_rates = np.clip(input_rates, 1, None)  # Ensure no negative rates

        # Generate input times for each synapse
        base_input_times = generate_input_times(rate_mean, 1)[0]  # Base input times for synchronization

        input_times = []
        for i in range(num_synapses):
            # Synchronicity affects how much each synapse's input times deviate from the base times
            deviation = np.random.normal(0, (1 - sync_factor[i]) * 10, len(base_input_times))
            syn_input_times = base_input_times + deviation
            input_times.append(syn_input_times)

        # Build synaptic input current over time
        synaptic_current = np.zeros(len(time))
        delayed_input_times = []  # List to store delayed spike times for plotting
        for syn_idx in range(num_synapses):
            delayed_times = []
            for spike_time in input_times[syn_idx]:
                # Apply delay based on synaptic length
                delay = synaptic_lengths[syn_idx] / 0.1  # Adjust delay as needed
                input_time = spike_time + delay
                delayed_times.append(input_time)  # Store delayed time
                input_idx = int(round(input_time / dt))
                if 0 <= input_idx < len(time):
                    synaptic_current[input_idx] += synaptic_weights[syn_idx]
            delayed_input_times.append(delayed_times)  # Add to list for plotting

        # Membrane potential and spike train
        V = np.full(len(time), -70.0)  # Initialize with resting potential of -70 mV
        resting_potential = -70.0
        spikes = np.zeros(len(time))
        refractory_counter = 0  # Use an integer counter for the refractory period
        refractory_regions = []

        # Simulation loop
        for t_idx in range(1, len(time)):
            if refractory_counter > 0:
                # Neuron is in refractory period
                refractory_counter -= 1  # Decrease refractory time remaining
                if just_fired:
                  V[t_idx] = resting_potential  # Membrane potential reset to resting value
                  just_fired = False
                else:
                  # Apply leaky decay to membrane potential
                  add_mem_decay()
                  # Add contribution of input with its weight and delay when the delayed spike reaches the soma
                  V[t_idx] += synaptic_current[t_idx]
                continue  # Skip to next time step
            else:
                # Apply leaky decay to membrane potential
                add_mem_decay()
                
                # Add synaptic current at this time step
                V[t_idx] += synaptic_current[t_idx]

                # Check for firing (only reset after visualization)
                if V[t_idx] >= threshold:
                    spikes[t_idx] = 1
                    just_fired = True
                    V_peak = V[t_idx]  # Record peak membrane potential
                    
                    # Calculate adaptive refractory period as a function of peak potential
                    refractory_period_duration = refractory_scale * abs(V_peak)
                    refractory_counter = int(refractory_period_duration / dt)  # Start refractory period
                    refractory_regions.append((time[t_idx], time[t_idx] + refractory_period_duration))
                    
                    # Reset membrane potential after spike, but only in the next time step
                    if t_idx + 1 < len(time):
                        V[t_idx + 1] = resting_potential

        # Clear the previous plot output before drawing the new plot
        with output_plot:
            output_plot.clear_output(wait=True)

            # Use subplot_mosaic to create a mosaic layout
            fig, axs = plt.subplot_mosaic(
                [["Synaptic Length", "Spike Train"],
                 ["Synchronicity", "Membrane Potential"],
                 ["Input Rate", "Membrane Potential"],
                 ["Synaptic Weight", "Input Raster"]],
                figsize=(14, 10),
                constrained_layout=True
            )

            # Plot Synaptic Length Distribution
            axs["Synaptic Length"].hist(synaptic_lengths, bins=10, color='blue', alpha=0.7)
            axs["Synaptic Length"].set_title("Synaptic Length Distribution")
            axs["Synaptic Length"].set_xlabel("Length (mm)")
            axs["Synaptic Length"].set_ylabel("Count")

            # Plot Synchronization Factor Distribution
            axs["Synchronicity"].hist(sync_factor, bins=10, color='green', alpha=0.7)
            axs["Synchronicity"].set_title("Synchronization Factor Distribution")
            axs["Synchronicity"].set_xlabel("Synchronization Probability")
            axs["Synchronicity"].set_ylabel("Count")

            # Plot Input Rate Distribution
            axs["Input Rate"].hist(input_rates, bins=10, color='red', alpha=0.7)
            axs["Input Rate"].set_title("Input Rate Distribution")
            axs["Input Rate"].set_xlabel("Rate (Hz)")
            axs["Input Rate"].set_ylabel("Count")

            # Plot Synaptic Weight Distribution
            axs["Synaptic Weight"].hist(synaptic_weights, bins=10, color='purple', alpha=0.7)
            axs["Synaptic Weight"].set_title("Synaptic Weight Distribution")
            axs["Synaptic Weight"].set_xlabel("Synaptic Weight")
            axs["Synaptic Weight"].set_ylabel("Count")

            # Plot Membrane Potential
            axs["Membrane Potential"].plot(time, V, color='b')
            axs["Membrane Potential"].set_title("Leaky Integrate-and-Fire Neuron Model")
            axs["Membrane Potential"].set_ylabel("Membrane Potential (mV)")
            axs["Membrane Potential"].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            for idx, (start, end) in enumerate(refractory_regions):
                axs["Membrane Potential"].axvspan(start, end, color='yellow', alpha=0.3, label='Refractory Period' if idx == 0 else "")
            axs["Membrane Potential"].legend()
            axs["Membrane Potential"].grid(True)
            axs["Membrane Potential"].set_xlim(0, simtime)  # Fix x-axis from 0 ms to 100 ms
            axs["Membrane Potential"].set_ylim(-70, 100)  # Static y-axis limits
            axs["Membrane Potential"].set_yticks(np.arange(-70, 101, 10))  # Defined y-ticks

            # Plot Spike Train
            axs["Spike Train"].plot(time, spikes, color='k')
            axs["Spike Train"].set_title("Neuron Spike Train")
            axs["Spike Train"].set_ylabel("Spikes")
            axs["Spike Train"].set_ylim(-0.1, 1.1)
            axs["Spike Train"].grid(True)
            axs["Spike Train"].set_xlim(0, simtime)  # Fix x-axis from 0 ms to 100 ms

            # Plot Input Raster
            for syn_idx in range(num_synapses):
                # Original spike times (before delay)
                axs["Input Raster"].scatter(input_times[syn_idx], [syn_idx]*len(input_times[syn_idx]),
                                            marker='|', color='black', s=100, label='Original Spike' if syn_idx == 0 else "")
                # Delayed spike times (after delay)
                # Filter delayed times within simulation time
                valid_delayed_times = [t for t in delayed_input_times[syn_idx] if 0 <= t <= simtime]
                axs["Input Raster"].scatter(valid_delayed_times, [syn_idx]*len(valid_delayed_times),
                                            marker='|', color='grey', s=100, label='Delayed Spike' if syn_idx == 0 else "")
            axs["Input Raster"].set_title("Synaptic Input Raster Plot")
            axs["Input Raster"].set_xlabel("Time (ms)")
            axs["Input Raster"].set_ylabel("Synapse")
            axs["Input Raster"].set_ylim(-1, num_synapses)
            axs["Input Raster"].set_xlim(0, simtime)  # Fix x-axis from 0 ms to 100 ms
            axs["Input Raster"].grid(True)
            axs["Input Raster"].legend(loc='upper right')

            plt.show()

    # Create sliders for each parameter
    style = {'description_width': 'initial'}
    syn_lengths_mean_slider = FloatSlider(min=0.1, max=5, step=0.1, value=1, description='Distance: Synapse-Soma (mm) Mean', style=style, layout=Layout(width='500px'))
    syn_lengths_std_slider = FloatSlider(min=0.01, max=2, step=0.01, value=0.5, description='Distance: Synapse-Soma (mm) Std:', style=style, layout=Layout(width='500px'))
    sync_mean_slider = FloatSlider(min=0, max=1, step=0.05, value=0.5, description='Synchronization Mean:', style=style, layout=Layout(width='500px'))
    sync_std_slider = FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='Synchronization Std:', style=style, layout=Layout(width='500px'))
    rate_mean_slider = FloatSlider(min=1, max=100, step=1, value=10, description='Input Rate Mean:', style=style, layout=Layout(width='500px'))
    rate_std_slider = FloatSlider(min=0.01, max=20, step=0.01, value=2, description='Input Rate Std:', style=style, layout=Layout(width='500px'))
    weights_mean_slider = FloatSlider(min=0, max=20, step=0.5, value=5, description='Synaptic Weight Mean:', style=style, layout=Layout(width='500px'))
    weights_std_slider = FloatSlider(min=0.01, max=5, step=0.01, value=1, description='Synaptic Weight Std:', style=style, layout=Layout(width='500px'))
    tau_slider = FloatSlider(min=1, max=100, step=1, value=10, description='Leak Time Constant (ms):', style=style, layout=Layout(width='500px'))
    threshold_slider = FloatSlider(min=-60, max=100, step=1, value=40, description='Threshold (mV):', style=style, layout=Layout(width='500px'))

    # Combine sliders into a vertical layout
    slider_box = VBox([syn_lengths_mean_slider, syn_lengths_std_slider,
                       sync_mean_slider, sync_std_slider, rate_mean_slider, rate_std_slider,
                       weights_mean_slider, weights_std_slider, tau_slider, threshold_slider])

    # Create the "Update Plot" button
    plot_button = Button(description="Update Plot", button_style='success')

    # Callback function to update plot when the button is clicked
    def on_button_click(b):
        update_plot(
            syn_lengths_mean=syn_lengths_mean_slider.value,
            syn_lengths_std=syn_lengths_std_slider.value,
            sync_mean=sync_mean_slider.value,
            sync_std=sync_std_slider.value,
            rate_mean=rate_mean_slider.value,
            rate_std=rate_std_slider.value,
            weights_mean=weights_mean_slider.value,
            weights_std=weights_std_slider.value,
            tau=tau_slider.value,
            threshold=threshold_slider.value
        )

    # Link button to callback function
    plot_button.on_click(on_button_click)

    # Display the sliders and button together with the plot output
    ui = VBox([slider_box, plot_button, output_plot])
    display(ui)

    update_plot(
        syn_lengths_mean=syn_lengths_mean_slider.value,
        syn_lengths_std=syn_lengths_std_slider.value,
        sync_mean=sync_mean_slider.value,
        sync_std=sync_std_slider.value,
        rate_mean=rate_mean_slider.value,
        rate_std=rate_std_slider.value,
        weights_mean=weights_mean_slider.value,
        weights_std=weights_std_slider.value,
        tau=tau_slider.value,
        threshold=threshold_slider.value
    )