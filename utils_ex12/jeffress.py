from IPython.display import display
from PIL import Image
from brian2 import *
import matplotlib.pyplot as plt
import matplotlib
from ipywidgets import Output, VBox, Button, IntSlider, widgets
import matplotlib.gridspec as gridspec
import urllib


def jeffress(max_angle=180):
    def visualize_plot(degree, num_neurons: int):
        def plot_owl(ax, degree):
            url = "https://raw.githubusercontent.com/ManteLab/Iton_notebooks_public/refs/heads/main/utils_ex12/Figures/owl_head.png"

            with urllib.request.urlopen(url) as url_response:
                img = Image.open(url_response)
                img = np.array(img)

            # img = plt.imread("https://raw.githubusercontent.com/ManteLab/Iton_notebooks_public/refs/heads/main/utils_ex12/Figures/owl_head.png")
            plt.imshow(img, extent=[-0.5, 0.5, -0.5, 0.5])

            # Add neurons in a circle
            for i in range(num_neurons):
                angle = np.pi * i / (num_neurons-1)
                x = 0.4 * np.cos(angle)
                y = 0.4 * np.sin(angle)
                x = -x
                neuron = Circle((x, y), 0.02, color='blue')  # Adjust circle size as needed
                ax.add_patch(neuron)
                ax.text(x, y, str(i), ha='center', va='center', fontsize=6, color='white')

                # Add an arrow to visualize the ipd_deg in the second subplot
            ax.set_aspect('equal')  # Ensure the arrow plot is a circle
            ax.set_axis_off()
            arrow_length = 0.4
            sx, sy = -arrow_length * np.cos(np.radians(degree - 90)), -arrow_length * np.sin(np.radians(degree - 90))
            sx, sy = sx * 2, sy * 2
            arrow = Arrow(-sx, sy, sx / 2, -sy / 2, width=0.1, color='g')
            ax.add_patch(arrow)
            ax.set_xlim([-0.7, 0.7])  # Set x-axis limits for arrow plot
            ax.set_ylim([-0.7, 0.7])  # Set y-axis limits for arrow plot

        def plot_neurons(ax0, ax1, t, i, traces):
            # Get a colormap
            #  cmap = matplotlib.cm.get_cmap('viridis')  # Or any other colormap you prefer
            cmap = matplotlib.colormaps['viridis']

            # Normalize neuron indices to [0, 1] for color mapping
            colors = cmap(i / max(i))

            # Create the scatter plot
            ax0.scatter(t / 1e-3, i, c=colors, marker='.')  # Use 'c' for color

            ax0.set_xlabel('Time (ms)')
            ax0.set_ylabel('Neuron index')
            ax0.set_ylim(-1, num_neurons)
            ax0.set_xlim(0, 1000)

            ax1.plot(traces.t / 1e-3, traces.delay.T / 1e-3, '-')
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Input delay (ms)')
            ax1.set_ylim(-1, 1)
            ax1.set_xlim(0, 1000)

        defaultclock.dt = .02 * ms

        # Sound
        sound = TimedArray(10 * randn(50000), dt=defaultclock.dt)  # white noise

        # Ears and sound motion around the head (constant angular speed)
        sound_speed = 300 * metre / second
        interaural_distance = 20 * cm  # big head!
        max_delay = interaural_distance / sound_speed
        angular_speed = 0 / second  # 1 turn/second
        tau_ear = 1 * ms
        sigma_ear = .1
        ipd_deg = degree
        theta = ipd_deg * pi / 180
        eqs_ears = '''
        dx/dt = (sound(t-delay)-x)/tau_ear+sigma_ear*(2./tau_ear)**.5*xi : 1 (unless refractory)
        delay = distance*sin(theta) : second
        distance : second # distance to the centre of the head in time units
        '''
        # dtheta/dt = angular_speed : radian
        ears = NeuronGroup(2, eqs_ears, threshold='x > 1', reset='x = 0', refractory=2.5 * ms, name='ears',
                           method='euler')
        # ears = NeuronGroup(2, eqs_ears, threshold='rand()<rate*dt', dt=1*ms)
        ears.distance = [-.5 * max_delay, .5 * max_delay]
        traces = StateMonitor(ears, 'delay', record=True)
        M_x = SpikeMonitor(ears)
        # Coincidence detectors
        tau = 1 * ms
        sigma = .1
        eqs_neurons = '''
        dv/dt = -v / tau + sigma * (2 / tau)**.5 * xi : 1
        '''
        neurons = NeuronGroup(num_neurons, eqs_neurons, threshold='v > 1', reset='v = 0', name='neurons',
                              method='euler')

        synapses = Synapses(ears, neurons, on_pre='v += .5')
        synapses.connect()

        synapses.delay['i==0'] = '(1.0*j)/(num_neurons-1)*1.1*max_delay'
        synapses.delay['i==1'] = '(1.0*(num_neurons-j-1))/(num_neurons-1)*1.1*max_delay'

        #  synapses.delay['i==0'] = 'min(j, num_neurons-j-1) * 1.1 * max_delay / (num_neurons - 1)'
        #  synapses.delay['i==1'] = 'min(j, num_neurons-j-1) * 1.1 * max_delay / (num_neurons - 1)'  # Same for i==1

        #  synapses.delay['i==0'] = '(int(j < (num_neurons-j-1)) * j + int(j >= (num_neurons-j-1)) * (num_neurons-j-1))/(num_neurons/2-1)*1.1*max_delay'
        #  synapses.delay['i==1'] = '(1 - (int(j < (num_neurons-j- 1)) * j + int(j >= (num_neurons-j-1)) * (num_neurons-j-1))/(num_neurons/2-1))*1.1*max_delay'

        # synapses.delay['i==0'] = '(int(j < (num_neurons - j)) * j + int(j >= (num_neurons - j)) * (num_neurons - j)) / (num_neurons/2) * 1.1 * max_delay'
        # synapses.delay['i==1'] = '(1 - (int(j < (num_neurons - j)) * j + int(j >= (num_neurons - j)) * (num_neurons - j)) / (num_neurons/2)) * 1.1 * max_delay'

        M_n = SpikeMonitor(neurons)

        run(1 * second)

        trains = M_x.spike_trains()

        # Plot the results
        i, t = M_n.it

        fig = figure(figsize=(12, 8), dpi=200)
        # 2 => 0.3
        gs = gridspec.GridSpec(3, 2, hspace=0.5, height_ratios=[1, .2, 1], width_ratios=[1, 1])  # 2x2 GridSpec

        ax0 = subplot(gs[0, 0])
        ax1 = subplot(gs[1, 0])
        plot_neurons(ax0, ax1, t, i, traces)

        ax_owl = plt.subplot(gs[:, 1])  # Span both rows in the first column
        plot_owl(ax_owl, ipd_deg)

        ax = subplot(gs[2, 0])
        plot(trains[0] / ms, [0] * len(trains[0]), '|')
        plot(trains[1] / ms, [1] * len(trains[1]), '|')
        ylim(-1, 2)
        gca().set_frame_on(False)
        xlabel('Time')
        ylabel('Spikes')
        yticks([])
        xticks([])

        plt.show()

    output = Output()
    angle = IntSlider(value=0, min=0, max=max_angle, step=15, description='Angle (deg)')
    num_neurons = IntSlider(value=16, min=2, max=30, step=1, description='# neurons')
    update_button = Button(description="Update", button_style='success')

    def update_plot():
        degree = (angle.value - 90 + 360) % 360
        with output:
            output.clear_output(wait=True)
            visualize_plot(degree=degree, num_neurons=num_neurons.value)

    update_button.on_click(lambda _: update_plot())
    update_button.click()

    return VBox([angle, num_neurons, update_button, output])


def jeffress_app():
    # Setup the tabs
    tab = widgets.Tab()
    tab.children = [
        jeffress(max_angle=180),
        jeffress(max_angle=360)
    ]
    tab.titles = ('Jeffress Model', 'Investigate limits')

    display(tab)
