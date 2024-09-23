import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from IPython.display import display, clear_output
import ipywidgets as widgets
from scipy.constants import R, physical_constants
px = 1/plt.rcParams['figure.dpi']

from matplotlib.patches import Arrow
from IPython.display import display, clear_output, HTML

def rc_euler(Vs, V0, R, C, dt, t):
    V_t = np.zeros(t.shape)
    V_t[0] = V0
    for i in range(1, len(t)):
        V_t[i] = (1 - np.exp(-dt / (R*C)))*Vs + np.exp(-dt / (R * C)) * V_t[i - 1]  # (1 - np.exp(-dt / (R * C))) * V_t[i - 1] + (1 - np.exp(-dt / (R*C))) * Vs
    return V_t

def rc_charging_exact(Vs, R, C, t):
    return Vs * (1 - np.exp(-(t/1000) / (R * C)))

def rc_discharging_exact(V0, R, C, t):
    return V0 * np.exp(-(t/1000) / (R * C))

def plot_vc_slider(Vs_default=5, R_default=10000, C_default=0.000005, V0=0):
    px = 1/plt.rcParams['figure.dpi']
    t = np.arange(0, 100, 1) # Time in ms
    fig, ax = plt.subplots(1, 1, figsize=(800 * px, 600 * px))

    Vc_euler_data_default = rc_euler(Vs_default, V0, R_default, C_default, 0.001, t)  # Use a time step of 1 ms
    Vc_euler_line = ax.plot(t, Vc_euler_data_default, label=r'$V_C$', linestyle='solid', linewidth=400 * px)[0]

    vs_data_default = Vs_default * np.ones(t.shape)
    vs_line = ax.plot(t, vs_data_default, label=r'$V_S$', linestyle='--', linewidth=400 * px)[0]

    ax_vs = fig.add_axes([0.15, 0.1, 0.65, 0.03])
    ax_r = fig.add_axes([0.15, 0.15, 0.65, 0.03])
    ax_c = fig.add_axes([0.15, 0.2, 0.65, 0.03])

    vs_slider = Slider(ax_vs, 'Vs', 0, 10, valinit=Vs_default)
    r_slider = Slider(ax_r, 'R', 100, 100000, valinit=R_default, valstep=100)
    c_slider = Slider(ax_c, 'C', 0.000001, 0.00001, valinit=C_default, valstep=0.000001)

    ax.set_ylim(0, 10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'$V_C$ (V)')
    fig.subplots_adjust(left=0.15, bottom=0.35)

    def update(val):
        Vs = vs_slider.val
        R = r_slider.val
        C = c_slider.val

        Vc_euler_data = rc_euler(Vs, V0, R, C, 0.001, t)  # Use a time step of 1 ms
        Vc_euler_line.set_ydata(Vc_euler_data)

        vs_data = Vs * np.ones(t.shape)
        vs_line.set_ydata(vs_data)

        fig.canvas.draw_idle()
        # ax.relim()
        # ax.autoscale_view()

    vs_slider.on_changed(update)
    r_slider.on_changed(update)
    c_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    resetax._button = button

    def reset(event):
        vs_slider.reset()
        r_slider.reset()
        c_slider.reset()
        # ax.relim()
        # ax.autoscale_view()

    button.on_clicked(reset)
    # plot legend inside the plot
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_title('Dependence of the RC circuit dynamics on its parameters')
    plt.show()


def plot_vc_tc_slider(plot_vertical_tc=False):
    fig, ax = plt.subplots(1, 1, figsize=(800 * px, 600 * px))
    Vs_default = 5
    R_default = 10000
    C_default = 0.000001
    V0_default = 0
    t = np.arange(0, 100, 1)  # Time in ms
    Vc_euler_data_default = rc_euler(Vs_default, V0_default, R_default, C_default, 0.001, t)  # Use a time step of 1 ms
    Vc_euler_line = ax.plot(t, Vc_euler_data_default, label=r'$V_C$', linestyle='solid', linewidth=400 * px)[0]

    vs_data_default = Vs_default * np.ones(t.shape)
    vs_line = ax.plot(t, vs_data_default, label=r'$V_S$', linestyle='--', linewidth=400 * px)[0]

    tangent_data_default = (Vs_default / (R_default * C_default)) * (t / 1000)
    tangent_line = ax.plot(t, tangent_data_default, label='tangent', linestyle='dashed', linewidth=400 * px)[0]

    exp_minus_1_data_default = Vs_default * (1 - np.exp(-1)) * np.ones(t.shape)
    exp_minus_1_line = \
    ax.plot(t, exp_minus_1_data_default, label=r'$Vs*(1 - e^{-1})=0.63*Vs$', linestyle='dashed', linewidth=400 * px)[0]

    if plot_vertical_tc:
        vertical_tc_data_default = R_default * C_default * 1000
        vertical_tc_line = ax.axvline(x=vertical_tc_data_default, color='r', linestyle='--', label=r'$\tau=RC$')

    ax_vs = fig.add_axes([0.15, 0.1, 0.65, 0.03])
    ax_r = fig.add_axes([0.15, 0.15, 0.65, 0.03])
    ax_c = fig.add_axes([0.15, 0.2, 0.65, 0.03])
    vs_slider = Slider(ax_vs, 'Vs', 0, 10, valinit=Vs_default)
    r_slider = Slider(ax_r, 'R', 100, 100000, valinit=R_default, valstep=100)
    c_slider = Slider(ax_c, 'C', 0.000001, 0.00001, valinit=C_default, valstep=0.000001)

    ax.set_ylim(0, 10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'$V_C$ (V)')
    fig.subplots_adjust(left=0.15, bottom=0.35)

    def update(val):
        Vs = vs_slider.val
        R = r_slider.val
        C = c_slider.val

        Vc_euler_data = rc_euler(Vs, V0_default, R, C, 0.001, t)  # Use a time step of 1 ms
        Vc_euler_line.set_ydata(Vc_euler_data)
        # line_tc_data = ax.plot(t, R*C*t, label='RC', linestyle='dashed', linewidth=400*px)[0] --> gives the multiples lines

        vs_data = Vs * np.ones(t.shape)
        vs_line.set_ydata(vs_data)

        tangent_data = (Vs / (R * C)) * (t / 1000)
        tangent_line.set_ydata(tangent_data)

        exp_minus_1_data = Vs * (1 - np.exp(-1)) * np.ones(t.shape)
        exp_minus_1_line.set_ydata(exp_minus_1_data)

        if plot_vertical_tc:
            vertical_tc_data = R * C * 1000
            vertical_tc_line.set_xdata([vertical_tc_data])

        fig.canvas.draw_idle()
        # ax.relim()
        # ax.autoscale_view()

    vs_slider.on_changed(update)
    r_slider.on_changed(update)
    c_slider.on_changed(update)
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    resetax._button = button

    def reset(event):
        vs_slider.reset()
        r_slider.reset()
        c_slider.reset()
        # ax.relim()
        # ax.autoscale_view()

    button.on_clicked(reset)
    # plot legend inside the plot
    ax.legend(loc='upper right')
    ax.grid()
    plt.show()


# Nernst equation function
def nernst_potential(z, Cin, Cout, T):
    return (R_constant * T / (z * F_constant)) * np.log(Cout / Cin) * 1000  # in mV

# Nernst-Planck equation function
def nernst_planck_flux(D_i, C_i, z_i, dVdx, dCdx, T):
        J_i = -D_i * (dCdx + (z_i * F_constant / (R_constant * T)) * C_i * dVdx)
        return J_i

# GHK equation function
def ghk_potential(PK, PNa, PCl, K_in, K_out, Na_in, Na_out, Cl_in, Cl_out, T):
    num = PK * K_out + PNa * Na_out + PCl * Cl_in
    denom = PK * K_in + PNa * Na_in + PCl * Cl_out
    return (R_constant * T / F_constant) * np.log(num / denom) * 1000  # in mV

# Faraday constant (F) from scipy's physical_constants
F_constant = physical_constants['Faraday constant'][0]

# Define constants
T_default = 310  # Temperature in Kelvin (37°C, human body temperature)
R_constant = R  # Universal gas constant in J/(mol·K)


def plot_cable_v():
    i_e_default = 0.05e-9
    a_default = 2e-3
    r_m_default = 1e6
    r_L_default = 1e3
    lambda_elc_default = np.sqrt(r_m_default / r_L_default)
    R_l_default = r_L_default * lambda_elc_default / (np.pi * a_default ** 2)
    x = np.linspace(-100, 100, 1000)
    v_default = (i_e_default * R_l_default / 2) * np.exp(-np.abs(x) / lambda_elc_default)
    fig, ax = plt.subplots(1, 1, figsize=(800 * px, 400 * px))
    v_line = ax.plot(x, v_default * 1e3)[0]
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel(r'$v$ (mV)')
    ax.set_ylim(0, 120)
    ax.set_title('Membrane potential along the cable')

    fig.subplots_adjust(left=0.15, bottom=0.35)
    ax_a = plt.axes([0.1, 0.1, 0.65, 0.03])
    ax_ie = plt.axes([0.1, 0.15, 0.65, 0.03])

    a_slider = Slider(ax=ax_a, label='a', valmin=1, valmax=3, valinit=a_default * 1000, valstep=0.1)
    ie_slider = Slider(ax=ax_ie, label='i_e', valmin=0.01, valmax=0.1, valinit=i_e_default * 1e9, valstep=0.01)

    def update(val):
        a = a_slider.val / 1000
        i_e = ie_slider.val / 1e9
        R_l = r_L_default * lambda_elc_default / (np.pi * a ** 2)
        v = (i_e * R_l / 2) * np.exp(-np.abs(x) / lambda_elc_default)
        v_line.set_ydata(v * 1e3)
        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    ie_slider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    resetax._button = button

    def reset(event):
        a_slider.reset()
        ie_slider.reset()
        plt.draw()

    button.on_clicked(reset)
    # ax.grid()
    plt.show()

def nernst_interactive_plot():
    # Ion properties
    ions = ['K⁺', 'Na⁺', 'Cl⁻']
    z_i = {'K⁺': 1, 'Na⁺': 1, 'Cl⁻': -1}

    # Colors for plotting
    colors = {
        'K⁺': '#1f77b4',   # blue
        'Na⁺': '#ff7f0e',  # orange
        'Cl⁻': '#2ca02c',  # green
        'membrane': '#9467bd'  # purple
    }

    # Define widget sliders with custom style
    slider_style = {'description_width': '150px'}
    slider_layout = widgets.Layout(width='400px')

    # Ion selection dropdown
    ion_dropdown = widgets.Dropdown(
        options=ions,
        value='K⁺',
        description='Select Ion:',
        style=slider_style,
        layout=slider_layout
    )

    # Sliders for ion concentrations
    Cin_slider = widgets.FloatSlider(
        value=140, min=1, max=400, step=1,
        description='[Ion]_in (mM)', continuous_update=False,
        style=slider_style, layout=slider_layout
    )
    Cout_slider = widgets.FloatSlider(
        value=5, min=1, max=400, step=1,
        description='[Ion]_out (mM)', continuous_update=False,
        style=slider_style, layout=slider_layout
    )

    # Temperature slider
    T_slider = widgets.FloatSlider(
        value=T_default, min=273, max=373, step=1,
        description='Temperature (K)', continuous_update=False,
        style=slider_style, layout=slider_layout
    )

    # Output area for the plot
    output_area = widgets.Output()

    def update_plot(*args):
        with output_area:
            clear_output(wait=True)

            # Retrieve slider values
            ion = ion_dropdown.value
            Cin = Cin_slider.value
            Cout = Cout_slider.value
            T = T_slider.value
            z = z_i[ion]

            # Calculate Nernst potential
            E_ion = nernst_potential(z, Cin, Cout, T)

            # Visualization
            fig, ax = plt.subplots(figsize=(8, 8))

            # Draw cell membrane
            theta = np.linspace(0, 2 * np.pi, 100)
            x_outer = np.cos(theta)
            y_outer = np.sin(theta)
            ax.plot(x_outer, y_outer, color=colors['membrane'])  # Circle outline

            # Scatter intracellular ions
            color = colors[ion]
            np.random.seed(42)  # For reproducibility
            num_in = int(Cin)
            num_in = min(num_in, 100)  # Limit for visualization
            x_in = np.random.uniform(-0.7, 0.7, num_in)
            y_in = np.random.uniform(-0.7, 0.7, num_in)
            ax.scatter(x_in, y_in, color=color, s=10, alpha=0.5)

            # Scatter extracellular ions
            num_out = int(Cout)
            num_out = min(num_out, 100)  # Limit for visualization
            theta_out = np.random.uniform(0, 2 * np.pi, num_out)
            r_out = np.random.uniform(1.1, 1.3, num_out)
            x_out = r_out * np.cos(theta_out)
            y_out = r_out * np.sin(theta_out)
            ax.scatter(x_out, y_out, color=color, s=10, alpha=0.5)

            # Adjust plot
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            # Create a custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=ion, markerfacecolor=color, markersize=10),
                Line2D([0], [0], color=colors['membrane'], lw=2, label='Cell Membrane'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

            ax.set_title(f"Ion Distribution for {ion}\n")
            ax.text(0, -1.4, f"Nernst Potential for {ion}: {E_ion:.2f} mV", fontsize=12, ha='center')

            plt.tight_layout()
            plt.show()

    # Link the widgets to the plot
    ion_dropdown.observe(update_plot, 'value')
    Cin_slider.observe(update_plot, 'value')
    Cout_slider.observe(update_plot, 'value')
    T_slider.observe(update_plot, 'value')

    # Function to update sliders based on ion selection
    def update_sliders(*args):
        ion = ion_dropdown.value
        if ion == 'K⁺':
            Cin_slider.value = 140
            Cin_slider.max = 400
            Cout_slider.value = 5
            Cout_slider.max = 400
        elif ion == 'Na⁺':
            Cin_slider.value = 12
            Cin_slider.max = 150
            Cout_slider.value = 145
            Cout_slider.max = 150
        elif ion == 'Cl⁻':
            Cin_slider.value = 4
            Cin_slider.max = 150
            Cout_slider.value = 110
            Cout_slider.max = 150
        update_plot()

    ion_dropdown.observe(update_sliders, 'value')

    # Organize widgets
    sliders = widgets.VBox([ion_dropdown, Cin_slider, Cout_slider, T_slider])
    layout = widgets.HBox([sliders, output_area])

    display(layout)

    # Initial plot
    update_sliders()

def nernst_planck_interactive_plot():
    # Ion properties
    ions = ['K⁺', 'Na⁺', 'Cl⁻']
    z_i = {'K⁺': 1, 'Na⁺': 1, 'Cl⁻': -1}
    D_i = {'K⁺': 1.96e-9, 'Na⁺': 1.33e-9, 'Cl⁻': 2.03e-9}  # Diffusion coefficients in m²/s

    # Colors for plotting
    colors = {
        'K⁺': '#1f77b4',   # blue
        'Na⁺': '#ff7f0e',  # orange
        'Cl⁻': '#2ca02c',  # green
        'membrane': '#9467bd'  # purple
    }

    # Define widget sliders with custom style
    slider_style = {'description_width': '150px'}
    slider_layout = widgets.Layout(width='400px')

    # Ion selection dropdown
    ion_dropdown = widgets.Dropdown(
        options=ions,
        value='K⁺',
        description='Select Ion:',
        style=slider_style,
        layout=slider_layout
    )

    # Sliders for ion concentrations
    Cin_slider = widgets.FloatSlider(value=140, min=1, max=400, step=1, description='[Ion]_in (mM)', continuous_update=False, style=slider_style, layout=slider_layout)
    Cout_slider = widgets.FloatSlider(value=5, min=1, max=400, step=1, description='[Ion]_out (mM)', continuous_update=False, style=slider_style, layout=slider_layout)

    # Membrane potential slider
    V_mem_slider = widgets.FloatSlider(value=-70e-3, min=-0.1, max=0.1, step=1e-3, description='V_mem (V)', continuous_update=False, style=slider_style, layout=slider_layout)

    # Temperature slider
    T_slider = widgets.FloatSlider(value=T_default, min=273, max=373, step=1, description='Temperature (K)', continuous_update=False, style=slider_style, layout=slider_layout)

    # Output area for the plot
    output_area = widgets.Output()

    def update_plot(*args):
        with output_area:
            clear_output(wait=True)

            # Retrieve slider values
            ion = ion_dropdown.value
            Cin = Cin_slider.value
            Cout = Cout_slider.value
            V_mem = V_mem_slider.value
            T = T_slider.value
            z = z_i[ion]
            D = D_i[ion]

            # Membrane thickness and position array
            x = np.linspace(0, 1e-8, 100)  # Membrane thickness of 10 nm

            # Potential across the membrane
            V_out = 0  # Extracellular potential
            V_in = V_mem  # Intracellular potential
            V = np.linspace(V_out, V_in, len(x))
            dVdx = np.gradient(V, x)

            # Linearly varying concentration across the membrane
            C_i = np.linspace(Cout, Cin, len(x))  # From outside to inside
            dCdx = np.gradient(C_i, x)

            # Calculate flux
            J = nernst_planck_flux(D, C_i, z, dVdx, dCdx, T)
            avg_J = np.mean(J)

            # Visualization
            fig, ax = plt.subplots(figsize=(8, 8))

            # Draw cell membrane
            theta = np.linspace(0, 2 * np.pi, 100)
            x_outer = np.cos(theta)
            y_outer = np.sin(theta)
            ax.plot(x_outer, y_outer, color=colors['membrane'], label='Cell Membrane')  # Circle outline

            # Scatter intracellular ions
            np.random.seed(42)  # For reproducibility
            color = colors[ion]
            num_in = int(Cin)
            num_in = min(num_in, 100)  # Limit for visualization
            x_in = np.random.uniform(-0.7, 0.7, num_in)
            y_in = np.random.uniform(-0.7, 0.7, num_in)
            ax.scatter(x_in, y_in, color=color, s=10, alpha=0.5)

            # Scatter extracellular ions
            num_out = int(Cout)
            num_out = min(num_out, 100)  # Limit for visualization
            theta_out = np.random.uniform(0, 2 * np.pi, num_out)
            r_out = np.random.uniform(1.1, 1.3, num_out)
            x_out = r_out * np.cos(theta_out)
            y_out = r_out * np.sin(theta_out)
            ax.scatter(x_out, y_out, color=color, s=10, alpha=0.5)

            # Show flux arrow 
            arrow_length = 0.2

            if avg_J < 0:
                # Flux from inside to outside
                start_radius = 0.9
                end_radius = start_radius + arrow_length
            else:
                # Flux from outside to inside
                start_radius = 1.1
                end_radius = start_radius - arrow_length

            angle = 0  # Arbitrary angle
            start_x = start_radius * np.cos(angle)
            start_y = start_radius * np.sin(angle)
            end_x = end_radius * np.cos(angle)
            end_y = end_radius * np.sin(angle)
            dx = end_x - start_x
            dy = end_y - start_y

            # Adjust plot
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"Ion Flux Across Neuron Membrane for {ion}")
            ax.text(0, -1.4, f"Average Flux: {avg_J:.2e} mol/(m²·s)", fontsize=12, ha='center')
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            # Draw the arrow
            ax.arrow(start_x, start_y, dx, dy, head_width=0.05, head_length=0.05, fc=color, ec=color, linewidth=2)

            # Create a custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=ion, markerfacecolor=color, markersize=10),
                Line2D([0], [0], color=colors['membrane'], lw=2, label='Cell Membrane'),
                Line2D([0], [0], marker=(3,0,0), color=color, label=f'{ion} Flux', markersize=15),
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

            plt.tight_layout()
            plt.show()

    def update_sliders(*args):
        ion = ion_dropdown.value
        if ion == 'K⁺':
            Cin_slider.value = 140
            Cin_slider.min = 1
            Cin_slider.max = 400
            Cin_slider.step = 1
            Cout_slider.value = 5
            Cout_slider.min = 1
            Cout_slider.max = 400
            Cout_slider.step = 1
        elif ion == 'Na⁺':
            Cin_slider.value = 12
            Cin_slider.min = 1
            Cin_slider.max = 150
            Cin_slider.step = 1
            Cout_slider.value = 145
            Cout_slider.min = 1
            Cout_slider.max = 150
            Cout_slider.step = 1
        elif ion == 'Cl⁻':
            Cin_slider.value = 4
            Cin_slider.min = 1
            Cin_slider.max = 150
            Cin_slider.step = 1
            Cout_slider.value = 110
            Cout_slider.min = 1
            Cout_slider.max = 150
            Cout_slider.step = 1
        update_plot()

    # Link the widgets to the plot and slider update functions
    ion_dropdown.observe(update_sliders, 'value')
    Cin_slider.observe(update_plot, 'value')
    Cout_slider.observe(update_plot, 'value')
    V_mem_slider.observe(update_plot, 'value')
    T_slider.observe(update_plot, 'value')

    # Organize widgets
    sliders = widgets.VBox([
        ion_dropdown,
        Cin_slider, Cout_slider,
        V_mem_slider,
        T_slider
    ])

    layout = widgets.HBox([sliders, output_area])

    display(layout)

    # Initial plot
    update_sliders()

def ghk_interactive_plot():
    # Colors for plotting
    colors = {
        'K+': '#1f77b4',   # blue
        'Na+': '#ff7f0e',  # orange
        'Cl-': '#2ca02c',  # green
        'membrane': '#9467bd'  # purple
    }

    # Define widget sliders with custom style
    slider_style = {'description_width': '150px'}
    slider_layout = widgets.Layout(width='400px')

    K_in_slider = widgets.FloatSlider(value=140, min=1, max=400, step=1, description='[K⁺]_in (mM)', continuous_update=False, style=slider_style, layout=slider_layout)
    K_out_slider = widgets.FloatSlider(value=5, min=1, max=400, step=1, description='[K⁺]_out (mM)', continuous_update=False, style=slider_style, layout=slider_layout)
    Na_in_slider = widgets.FloatSlider(value=12, min=1, max=150, step=1, description='[Na⁺]_in (mM)', continuous_update=False, style=slider_style, layout=slider_layout)
    Na_out_slider = widgets.FloatSlider(value=145, min=1, max=150, step=1, description='[Na⁺]_out (mM)', continuous_update=False, style=slider_style, layout=slider_layout)
    Cl_in_slider = widgets.FloatSlider(value=4, min=1, max=150, step=1, description='[Cl⁻]_in (mM)', continuous_update=False, style=slider_style, layout=slider_layout)
    Cl_out_slider = widgets.FloatSlider(value=110, min=1, max=150, step=1, description='[Cl⁻]_out (mM)', continuous_update=False, style=slider_style, layout=slider_layout)

    P_K_slider = widgets.FloatSlider(value=1, min=0, max=1, step=0.01, description='P_K', continuous_update=False, style=slider_style, layout=slider_layout)
    P_Na_slider = widgets.FloatSlider(value=0.03, min=0, max=1, step=0.01, description='P_Na', continuous_update=False, style=slider_style, layout=slider_layout)
    P_Cl_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='P_Cl', continuous_update=False, style=slider_style, layout=slider_layout)

    T_slider = widgets.FloatSlider(value=T_default, min=273, max=373, step=1, description='Temperature (K)', continuous_update=False, style=slider_style, layout=slider_layout)

    # Output area for the plot
    output_area = widgets.Output()

    def update_plot(*args):
        with output_area:
            clear_output(wait=True)

            # Retrieve slider values
            K_in = K_in_slider.value
            K_out = K_out_slider.value
            Na_in = Na_in_slider.value
            Na_out = Na_out_slider.value
            Cl_in = Cl_in_slider.value
            Cl_out = Cl_out_slider.value
            P_K = P_K_slider.value
            P_Na = P_Na_slider.value
            P_Cl = P_Cl_slider.value
            T = T_slider.value

            # Calculate Nernst potentials
            E_K = nernst_potential(1, K_in, K_out, T)
            E_Na = nernst_potential(1, Na_in, Na_out, T)
            E_Cl = nernst_potential(-1, Cl_in, Cl_out, T)

            # Calculate GHK potential
            V_m = ghk_potential(P_K, P_Na, P_Cl, K_in, K_out, Na_in, Na_out, Cl_in, Cl_out, T)

            # Visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # Draw cell membrane
            theta = np.linspace(0, 2 * np.pi, 100)
            x_outer = np.cos(theta)
            y_outer = np.sin(theta)
            ax.plot(x_outer, y_outer, color=colors['membrane'], label='Cell Membrane')  # Circle outline

            # Scatter intracellular ions
            ions = ['K+', 'Na+', 'Cl-']
            z_i = {'K+': 1, 'Na+': 1, 'Cl-': -1}
            concentrations_in = {'K+': K_in, 'Na+': Na_in, 'Cl-': Cl_in}
            concentrations_out = {'K+': K_out, 'Na+': Na_out, 'Cl-': Cl_out}

            np.random.seed(42)  # For reproducibility
            for ion in ions:
                num_in = int(concentrations_in[ion])
                color = colors[ion]
                # Limit the number of ions for visualization purposes
                num_in = min(num_in, 100)
                x_in = np.random.uniform(-0.7, 0.7, num_in)
                y_in = np.random.uniform(-0.7, 0.7, num_in)
                ax.scatter(x_in, y_in, color=color, s=10, alpha=0.5)

            # Scatter extracellular ions
            for ion in ions:
                num_out = int(concentrations_out[ion])
                color = colors[ion]
                # Limit the number of ions for visualization purposes
                num_out = min(num_out, 100)
                theta_out = np.random.uniform(0, 2 * np.pi, num_out)
                r_out = np.random.uniform(1.1, 1.3, num_out)
                x_out = r_out * np.cos(theta_out)
                y_out = r_out * np.sin(theta_out)
                ax.scatter(x_out, y_out, color=color, s=10, alpha=0.5)

            # Adjust plot limits to accommodate text
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.8, 1.5)  # Extended lower limit to make room for text
            ax.set_title("Ion Distribution Across Neuron Membrane")
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            # Create a custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='K⁺', markerfacecolor=colors['K+'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Na⁺', markerfacecolor=colors['Na+'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Cl⁻', markerfacecolor=colors['Cl-'], markersize=10),
                Line2D([0], [0], color=colors['membrane'], lw=2, label='Cell Membrane'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

            # Add membrane potentials inside the plot
            text_str = (f"Membrane Potential (GHK): {V_m:.2f} mV\n"
                        f"Nernst Potentials:\n"
                        f"Eₖ⁺: {E_K:.2f} mV\n"
                        f"Eₙₐ⁺: {E_Na:.2f} mV\n"
                        f"E_Cl⁻: {E_Cl:.2f} mV")
            # Place the text at a suitable location, e.g., at the bottom center
            ax.text(0, -1.6, text_str, fontsize=12, ha='center', va='top')

            plt.tight_layout()
            plt.show()

    # Link the widgets to the plot
    K_in_slider.observe(update_plot, 'value')
    K_out_slider.observe(update_plot, 'value')
    Na_in_slider.observe(update_plot, 'value')
    Na_out_slider.observe(update_plot, 'value')
    Cl_in_slider.observe(update_plot, 'value')
    Cl_out_slider.observe(update_plot, 'value')
    P_K_slider.observe(update_plot, 'value')
    P_Na_slider.observe(update_plot, 'value')
    P_Cl_slider.observe(update_plot, 'value')
    T_slider.observe(update_plot, 'value')

    # Organize sliders
    sliders = widgets.VBox(
        [K_in_slider, K_out_slider, Na_in_slider, Na_out_slider, Cl_in_slider, Cl_out_slider,
        P_K_slider, P_Na_slider, P_Cl_slider, T_slider])
    layout = widgets.HBox([sliders, output_area])

    display(layout)

    # Initial plot
    update_plot()