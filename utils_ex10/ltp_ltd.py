from typing import List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Button, FloatSlider, Layout, Output, VBox


spike_pattern = []

def init_spike_pattern():
    global spike_pattern
    spike_pattern = [0.02, 0.095, 0.15, 0.2, 0.225]
init_spike_pattern()


def iplot_InoF_model():
    output_plot = Output()

    # Update the plot (triggered by the button)
    def update_plot(x0: float,
                    u0: float,
                    ts: float,
                    tf: float,
                    td: float,
                    U: float,
                    A: float,
                    spike_pattern: List[int],
                    res: int = 1000):
        spike_pattern = list(map(lambda x: int(x * res), spike_pattern))

        time = np.arange(0, res + 1, 1)
        dt = 1 / res

        def du_dt__decay(u: float, tf: float) -> float:
            return -u / tf

        def du_dt__spike(u_minus: float, U: float, spike_occured: bool) -> float:
            return U * (1 - u_minus) * int(spike_occured)

        def dx_dt__decay(x: float, td: float) -> float:
            return (1 - x) / td

        def dx_dt__spike(x_minus: float, u_plus: float, spike_occured: bool) -> float:
            return u_plus * x_minus * int(spike_occured)

        def calc_x_minus(x: float, td: float) -> float:
            return x + dt * dx_dt__decay(x, td)

        def calc_x_plus(x_minus: float, u_plus: float, spike_occured: bool) -> float:
            return x_minus - dx_dt__spike(x_minus, u_plus, spike_occured)

        def calc_u_minus(u: float, tf: float) -> float:
            return u + dt * du_dt__decay(u, tf)

        def calc_u_plus(u_minus: float, U: float, spike_occured: bool) -> float:
            return u_minus + du_dt__spike(u_minus, U, spike_occured)

        def dy_dt(y: float, ts: float, u_plus: float, x_minus: float, spike_occured: bool) -> float:
            return - y / ts + A * u_plus * x_minus * res * int(spike_occured)

        def calc_y(y: float, ts: float, u_plus: float, x_minus: float, spike_occured: bool) -> float:
            return y + dt * dy_dt(y, ts, u_plus, x_minus, spike_occured)

        # ys = neuron current
        xs, us, ys = [x0], [u0], [0]
        spike_value = [0]
        for t_idx, t in enumerate(time[1:]):
            spiked = t in spike_pattern
            u_minus = calc_u_minus(us[-1], tf)
            u_plus = calc_u_plus(u_minus, U, spiked)
            x_minus = calc_x_minus(xs[-1], td)
            x_plus = calc_x_plus(x_minus, u_plus, spiked)
            y = calc_y(ys[-1], ts, u_plus, x_minus, spiked)
            us.append(u_plus)
            ys.append(y)
            xs.append(x_plus)
            spike_value.append(int(spiked))

        with output_plot:
            time_norm = time / res
            output_plot.clear_output(wait=True)

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 6), sharex=True)

            ax1.plot(time_norm, us, label='u')
            ax1.plot(time_norm, xs, label='x')
            ax1.set_title("Neuron internal state ($u$, $x$)")
            ax1.set_ylabel("value")
            ax2.set_ylim(-.1, 1.1)
            ax1.grid(True)
            ax1.legend()

            ax2.plot(time_norm, ys, color='black', label='current')
            ax2.set_title("Neuron current output ($I$)")
            ax2.set_ylabel("output current")
            ax2.set_ylim(-.1, 1.1)
            ax2.grid(True)
            ax2.legend()

            ax3.plot(time_norm, spike_value, color='black', label='spikes')
            ax3.set_title("Neuron spikes (input)")
            ax3.set_ylabel("input current")
            ax3.set_ylim(-.1, 1.1)
            ax2.set_xlabel("time ($t$)")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            plt.show()

    # Create sliders for synaptic weight, frequency, and length
    style = {'description_width': 'initial'}

    x0_slider = FloatSlider(min=0, max=1, step=0.01, value=0.8, description='x0:', style=style,
                            layout=Layout(width='500px'))
    u0_slider = FloatSlider(min=0, max=1, step=0.01, value=0.0, description='u0:', style=style,
                            layout=Layout(width='500px'))

    ts_slider = FloatSlider(min=0.01, max=3, step=0.01, value=1.0, description='ts:', style=style,
                            layout=Layout(width='500px'))
    tf_slider = FloatSlider(min=0.01, max=3, step=0.01, value=1.0, description='tf:', style=style,
                            layout=Layout(width='500px'))
    td_slider = FloatSlider(min=0.01, max=3, step=0.01, value=1.0, description='td:', style=style,
                            layout=Layout(width='500px'))

    U_slider = FloatSlider(min=0, max=2, step=0.01, value=0.8, description='U:', style=style,
                           layout=Layout(width='500px'))
    A_slider = FloatSlider(min=0, max=2, step=1, value=1.0, description='A:', style=style, layout=Layout(width='500px'))

    plot_button = Button(description="Update Plot", button_style='success')

    spd_dominated_button = Button(description="SPD-dominated", button_style='success')
    spf_dominated_button = Button(description="SPF-dominated", button_style='success')

    reset_spike_pattern_button = Button(description="Reset spike pattern", button_style='success')
    delete_spike_pattern_button = Button(description="Delete spike pattern", button_style='success')
    spike_slider = FloatSlider(min=0.01, max=0.99, step=0.01, value=0.5, description='Add spike at:', style=style, layout=Layout(width='300px'))
    spike_submit_button = Button(description="Add spike", button_style='success')

    def update_plot_by_values(_=None):
        global spike_pattern
        update_plot(
            x0=x0_slider.value,
            u0=u0_slider.value,
            ts=ts_slider.value,
            tf=tf_slider.value,
            td=td_slider.value,
            U=U_slider.value,
            A=A_slider.value,
            spike_pattern=spike_pattern
        )

    def set_sdp_dominated_parameters(_=None):
        x0_slider.value = 1.0
        u0_slider.value = 0.0
        ts_slider.value = 20 / 1000
        tf_slider.value = 50 / 1000
        td_slider.value = 750 / 1000
        U_slider.value = 0.45
        A_slider.value = 1.0
        update_plot_by_values()

    def set_sdf_dominated_parameters(_=None):
        x0_slider.value = 1.0
        u0_slider.value = 0.0
        ts_slider.value = 20 / 1000
        tf_slider.value = 750 / 1000
        td_slider.value = 50 / 1000
        U_slider.value = 0.15
        A_slider.value = 1.0
        update_plot_by_values()

    def reset_spike_pattern(_=None):
        global spike_pattern
        init_spike_pattern()
        update_plot_by_values()

    def delete_spike_pattern(_=None):
        global spike_pattern
        spike_pattern = []
        update_plot_by_values()

    def add_spike_to_pattern(_=None):
        global spike_pattern
        spike_pattern.append(spike_slider.value)
        spike_pattern = sorted(spike_pattern)
        update_plot_by_values()

    spd_dominated_button.on_click(set_sdp_dominated_parameters)
    spf_dominated_button.on_click(set_sdf_dominated_parameters)
    reset_spike_pattern_button.on_click(reset_spike_pattern)
    delete_spike_pattern_button.on_click(delete_spike_pattern)
    spike_submit_button.on_click(add_spike_to_pattern)

    plot_button.on_click(update_plot_by_values)

    ui = VBox(
        [x0_slider, u0_slider, ts_slider, tf_slider, td_slider, U_slider, A_slider, plot_button, spd_dominated_button,
         spf_dominated_button, reset_spike_pattern_button, delete_spike_pattern_button, spike_slider, spike_submit_button, output_plot])

    display(ui)

    update_plot_by_values()
