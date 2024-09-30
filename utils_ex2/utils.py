import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider

px = 1 / plt.rcParams['figure.dpi']

def plot_cable_v(y_axis_lim=80, plot_lambda=False):
    i_e_default = 1e-6
    a_default = 2e-3
    r_m_default = 1e6
    r_L_default = 1e3
    lambda_elc_default = np.sqrt((a_default*r_m_default) / (2*r_L_default))
    R_l_default = r_L_default * lambda_elc_default / (np.pi * a_default ** 2)
    x = np.linspace(-10, 10, 1000)
    v_default = (i_e_default * R_l_default / 2) * np.exp(-np.abs(x) / lambda_elc_default)
    fig, ax = plt.subplots(1, 1, figsize=(800 * px, 400 * px))
    v_line = ax.plot(x, v_default, label='Membrane potential')[0]
    thr_line = ax.axhline(y=20, color='r', linestyle='--', label='Threshold 20 mV')

    if plot_lambda:
        # lambda_elc_line = ax.axvline(x=lambda_elc_default, color='r', linestyle='--', label=r'$\lambda$')
        # two_lambda_elc_line = ax.axvline(x=2*lambda_elc_default, color='g', linestyle='--', label=r'$2\lambda$')
        # lambda_elc_hori_default = (i_e_default * R_l_default / 2) * np.exp(-np.abs(lambda_elc_default) / lambda_elc_default)
        # two_lambda_elc_line_hori_default = (i_e_default * R_l_default / 2) * np.exp(-np.abs(2*lambda_elc_default) / lambda_elc_default)
        # lambda_elc_hori_line = ax.axhline(y=lambda_elc_hori_default, color='r', linestyle='--')
        # two_lambda_elc_hori_line = ax.axhline(y=two_lambda_elc_line_hori_default, color='g', linestyle='--')
        four_mm_line = ax.axvline(x=4, color='b', linestyle='--', label='V(x=4 mm)')
        four_mm_hori = (i_e_default * R_l_default / 2) * np.exp(-4 / lambda_elc_default)
        four_mm_hori_line = ax.axhline(y=four_mm_hori, color='b', linestyle='--')



    ax.set_xlabel('Position (mm)')
    ax.set_ylabel(r'$v$ (mV)')
    ax.set_ylim(0, y_axis_lim)
    ax.set_title('Membrane potential along the cable')

    fig.subplots_adjust(left=0.15, bottom=0.45)
    ax_a =  plt.axes([0.15, 0.1, 0.65, 0.03])
    ax_ie = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_rm = plt.axes([0.15, 0.2, 0.65, 0.03])
    ax_rl = plt.axes([0.15, 0.25, 0.65, 0.03])

    a_slider = Slider(ax=ax_a, label='$a$ [$mm$]', valmin=0.5, valmax=5, valinit=a_default * 1000, valstep=0.1)
    ie_slider = Slider(ax=ax_ie, label=r'$i_e$ [$\mu A$]', valmin=0.01, valmax=10, valinit=i_e_default * 1e6, valstep=0.01)
    rm_slider = Slider(ax=ax_rm, label=r'$r_m$ [$M\Omega \cdot mm^2$]', valmin=0.1, valmax=10, valinit=r_m_default / 1e6, valstep=0.1)
    rl_slider = Slider(ax=ax_rl, label=r'$r_L$ [$k\Omega \cdot mm^2$]', valmin=0.1, valmax=5, valinit=r_L_default / 1e3, valstep=0.1)
    text_display = ax.text(0.2, 0.65, r'$\lambda$ ' + f' = {lambda_elc_default:.2f} \n' + r'$V_m(0)$' + f' = {i_e_default * R_l_default / 2:.2f}',
                           transform=ax.transAxes, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))

    def update(val):
        a = a_slider.val / 1000
        i_e = ie_slider.val / 1e6
        r_m = rm_slider.val * 1e6
        r_L = rl_slider.val * 1e3
        lambda_elc = np.sqrt((a * r_m) / (2 * r_L))
        R_l = r_L * lambda_elc / (np.pi * a ** 2)
        if plot_lambda:
            # lambda_elc_line.set_xdata([lambda_elc])
            # two_lambda_elc_line.set_xdata([2*lambda_elc])
            # lambda_elc_hori = (i_e * R_l_default / 2) * np.exp(-np.abs(lambda_elc) / lambda_elc)
            # lambda_elc_hori_line.set_ydata([lambda_elc_hori])
            # two_lambda_elc_hori = (i_e * R_l / 2) * np.exp(-np.abs(2*lambda_elc) / lambda_elc)
            # two_lambda_elc_hori_line.set_ydata([two_lambda_elc_hori])
            four_mm_hori = (i_e * R_l / 2) * np.exp(-4 / lambda_elc)
            four_mm_hori_line.set_ydata([four_mm_hori])

        v = (i_e * R_l / 2) * np.exp(-np.abs(x) / lambda_elc)
        v_line.set_ydata(v)
        vmem_0 = i_e * R_l / 2
        text_display.set_text(r'$\lambda$ ' + f' = {lambda_elc:.2f} \n' + r'$V_m(0)$' + f' = {vmem_0:.2f}')
        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    ie_slider.on_changed(update)
    rm_slider.on_changed(update)
    rl_slider.on_changed(update)

    resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    resetax._button = button

    two_lambda_ax = plt.axes([0.85, 0.075, 0.1, 0.04])
    two_lambda_button = Button(two_lambda_ax, '2$\lambda$', hovercolor='0.975')
    two_lambda_ax._button = two_lambda_button


    def reset(event):
        a_slider.reset()
        ie_slider.reset()
        rm_slider.reset()
        rl_slider.reset()
        plt.draw()

    def set_two_lambda(event):
        a_slider.set_val(1)
        ie_slider.set_val(0.39)
        rm_slider.set_val(6.4)
        rl_slider.set_val(0.8)
        plt.draw()

    # set x-axis ticks from -10 to 10 with step 1
    ax.set_xticks(np.arange(-10, 11, 1))

    button.on_clicked(reset)
    two_lambda_button.on_clicked(set_two_lambda)
    ax.grid()
    ax.legend()
    plt.show()



def plot_multi_injection(y_axis_lim=80):
    x_default = [-1, 0, 4.5]
    i_default = [1e-6 for _ in range(3)]
    a_default = 2e-3
    r_m_default = 1e6
    r_L_default = 1e3
    lambda_elc_default = np.sqrt((a_default * r_m_default) / (2 * r_L_default))
    R_l_default = r_L_default * lambda_elc_default / (np.pi * a_default ** 2)
    x = np.linspace(-10, 10, 1000)
    v_default = []
    for i in range(3):
        v_default.append((i_default[i] * R_l_default / 2) * np.exp(-np.abs(x - x_default[i]) / lambda_elc_default))

    vsum_default = v_default[0] + v_default[1] + v_default[2]
    fig, ax = plt.subplots(1, 1, figsize=(800 * px, 600 * px))
    v0_line = ax.plot(x, v_default[0], label=r'$v_0$')[0]
    v1_line = ax.plot(x, v_default[1], label=r'$v_1$')[0]
    v2_line = ax.plot(x, v_default[2], label=r'$v_2$')[0]
    vsum_line = ax.plot(x, vsum_default, label=r'$\sum v_k$')[0]

    ax.set_xlabel('Position (mm)')
    ax.set_ylabel(r'$v$ (mV)')
    ax.set_ylim(0, y_axis_lim)
    ax.set_title('Membrane potential along the cable')

    fig.subplots_adjust(left=0.15, bottom=0.60)
    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03])
    ax_i2 = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_i1 = plt.axes([0.15, 0.2, 0.65, 0.03])
    ax_i0 = plt.axes([0.15, 0.25, 0.65, 0.03])
    ax_x2 = plt.axes([0.15, 0.3, 0.65, 0.03])
    # ax_x1 = plt.axes([0.15, 0.35, 0.65, 0.03])
    ax_x0 = plt.axes([0.15, 0.35, 0.65, 0.03])
    ax_rm = plt.axes([0.15, 0.4, 0.65, 0.03])
    ax_rl = plt.axes([0.15, 0.45, 0.65, 0.03])

    a_slider = Slider(ax=ax_a, label=r'$a$ [$mm$]', valmin=0.5, valmax=5, valinit=a_default * 1000, valstep=0.1,
                      facecolor='black')
    i0_slider = Slider(ax=ax_i0, label=r'$i_0$ [$\mu A$]', valmin=0.01, valmax=2, valinit=i_default[0] * 1e6,
                       valstep=0.01, facecolor='tab:blue')
    i1_slider = Slider(ax=ax_i1, label=r'$i_1$ [$\mu A$]', valmin=0.01, valmax=2, valinit=i_default[0] * 1e6,
                       valstep=0.01, facecolor='tab:orange')
    i2_slider = Slider(ax=ax_i2, label=r'$i_2$ [$\mu A$]', valmin=0.01, valmax=2, valinit=i_default[0] * 1e6,
                       valstep=0.01, facecolor='tab:green')
    x0_slider = Slider(ax=ax_x0, label=r'$x_0$ [$mm$]', valmin=-5, valmax=-0.5, valinit=x_default[0], valstep=0.1,
                       facecolor='tab:blue')
    # x1_slider = Slider(ax=ax_x1, label=r'x1 [$mm$]',  valmin=0.5,  valmax=5,    valinit=x_default[1],       valstep=0.1,  facecolor='tab:orange')
    x2_slider = Slider(ax=ax_x2, label=r'$x_2$ [$mm$]', valmin=0.5, valmax=5, valinit=x_default[2], valstep=0.1,
                       facecolor='tab:green')
    rm_slider = Slider(ax=ax_rm, label=r'$r_m$ [$M\Omega \cdot mm^2$]', valmin=0.1, valmax=5, valinit=r_m_default / 1e6,
                       valstep=0.1, facecolor='black')
    rl_slider = Slider(ax=ax_rl, label=r'$r_L$ [$\Omega \cdot mm$]', valmin=10, valmax=5000, valinit=r_L_default,
                       valstep=0.1, facecolor='black')

    text_display = ax.text(0.3, 0.8, r'$\lambda$ ' + f' = {lambda_elc_default:.2f}', transform=ax.transAxes,
                           fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))

    def update(val):
        a = a_slider.val / 1000
        i0 = i0_slider.val / 1e6
        i1 = i1_slider.val / 1e6
        i2 = i2_slider.val / 1e6
        _i = [i0, i1, i2]
        r_m = rm_slider.val * 1e6
        r_L = rl_slider.val
        lambda_elc = np.sqrt((a * r_m) / (2 * r_L))
        x0 = x0_slider.val
        # x1 = x1_slider.val
        x2 = x2_slider.val
        _x = [x0, 0, x2]
        R_l = r_L * lambda_elc / (np.pi * a ** 2)
        v = []
        for i in range(3):
            v.append((_i[i] * R_l / 2) * np.exp(-np.abs(x - _x[i]) / lambda_elc))
        vsum = v[0] + v[1] + v[2]
        v0_line.set_ydata(v[0])
        v1_line.set_ydata(v[1])
        v2_line.set_ydata(v[2])
        vsum_line.set_ydata(vsum)
        text_display.set_text(r'$\lambda$ ' + f' = {lambda_elc:.2f} mm')
        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    i0_slider.on_changed(update)
    i1_slider.on_changed(update)
    i2_slider.on_changed(update)
    x0_slider.on_changed(update)
    # x1_slider.on_changed(update)
    x2_slider.on_changed(update)
    rm_slider.on_changed(update)
    rl_slider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    resetax._button = button

    def reset(event):
        a_slider.reset()
        i0_slider.reset()
        i1_slider.reset()
        i2_slider.reset()
        x0_slider.reset()
        # x1_slider.reset()
        x2_slider.reset()
        rm_slider.reset()
        rl_slider.reset()
        plt.draw()

    ax.set_xticks(np.arange(-10, 11, 1))

    button.on_clicked(reset)
    ax.grid()
    ax.legend()
    plt.show()

def iplot_InoF_model():
    # Parameters
    simtime = 100  # Total number of timesteps (e.g., 100 ms simulation)
    dt = 1  # Time step in milliseconds

    # Synaptic input times (fixed)
    input_times = [
        [20, 50],  # Synapse 1 fires at t=20ms and t=50ms
        [30, 60],  # Synapse 2 fires at t=30ms and t=60ms
        [40, 80]   # Synapse 3 fires at t=40ms and t=80ms
    ]
    num_synapses = len(input_times)

    # Function to update the plot
    def update_plot(weight1=5, weight2=10, weight3=15, delay1=5, delay2=10, delay3=15):
        # Membrane potential
        V = np.zeros(simtime)

        # Input raster plot data
        raster = np.zeros((num_synapses, simtime))

        # Weights and delays for each synapse
        synaptic_weights = [weight1, weight2, weight3]
        synaptic_delays = [delay1, delay2, delay3]

        # Simulation loop
        for t in range(1, simtime):
            # Carry over membrane potential from previous timestep
            V[t] = V[t-1]

            # Add contribution of each input with its respective weight and delay
            for i in range(num_synapses):
                for spike_time in input_times[i]:
                    original_time = spike_time
                    input_time = spike_time + synaptic_delays[i]
                    if t == input_time:  # Register the input at the specific time (delayed)
                        V[t] += synaptic_weights[i]
                        raster[i, t] = 1  # Mark delayed spike in raster plot

        # Plot the results
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Plot membrane potential
        axs[0].plot(np.arange(simtime) * dt, V, color='b')
        axs[0].set_title("Integrate-and-No-Fire Neuron")
        axs[0].set_ylabel("Membrane Potential (mV)")
        axs[0].grid(False)

        # Plot input spikes as raster
        for i in range(num_synapses):
            for spike_time in input_times[i]:
                # Plot original input time in black
                axs[1].scatter(spike_time, i, marker='|', color='black', s=100, label='Original input time' if i == 0 and spike_time == input_times[i][0] else "")
                # Plot delayed input at soma in grey
                delayed_time = spike_time + synaptic_delays[i]
                axs[1].scatter(delayed_time, i, marker='|', color='silver', s=100, label='Delayed input at soma' if i == 0 and spike_time == input_times[i][0] else "")

        axs[1].set_title("Synaptic Input Raster Plot (Black: Original, Grey: Delayed)")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_xticks(np.arange(0, simtime + 1, 10))
        axs[1].set_yticks(np.arange(num_synapses))
        axs[1].set_yticklabels([f'Synapse {i+1}' for i in range(num_synapses)])
        axs[1].grid(False)

        # Adding legend
        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    # Create sliders for synaptic weights and delays
    style = {'description_width': 'initial'}
    weight1_slider = FloatSlider(min=0, max=100, step=5, value=5, description='Synaptic  Weight 1:', style=style)
    weight2_slider = FloatSlider(min=0, max=100, step=5, value=10, description='Synaptic  Weight 2:', style=style)
    weight3_slider = FloatSlider(min=0, max=100, step=5, value=15, description='Synaptic  Weight 3:', style=style)

    delay1_slider = IntSlider(min=0, max=40, step=1, value=5, description='Synaptic Delay 1 (ms)', style=style)
    delay2_slider = IntSlider(min=0, max=40, step=1, value=10, description='Synaptic Delay 2 (ms)', style=style)
    delay3_slider = IntSlider(min=0, max=40, step=1, value=15, description='Synaptic Delay 3 (ms)', style=style)

    # Create the interactive plot with sliders
    interact(update_plot,
            weight1=weight1_slider,
            weight2=weight2_slider,
            weight3=weight3_slider,
            delay1=delay1_slider,
            delay2=delay2_slider,
            delay3=delay3_slider);