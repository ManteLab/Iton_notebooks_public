"""Helper utilities for exercise session 3: passive membrane properties, part 2.

The notebook that uses this module (`3_passive_mem_properties_part_2.ipynb`)
covers the passive cable, and this file holds its three interactive figures:

1. **`plot_cable_v`** — steady-state voltage along a cable driven by a single
   current injection at the origin, with the length constant lambda and the
   voltage at the injection site reported as you move the sliders.
2. **`plot_multi_injection`** — the same cable driven at three sites at once,
   showing that the individual voltage profiles simply add up.
3. **`iplot_InoF_model`** — an integrate-and-*no*-fire soma summing three
   synaptic inputs, each with its own weight and conduction delay.

The maths is deliberately kept in small pure functions (`cable_length_constant`,
`cable_input_resistance`, `cable_voltage`), so the formulas the assignments ask
about can be read, and tested, without going through a figure.

Interactive figures all share one structure, so once you have read one you have
read them all:

    controls  ->  a column of ipywidgets sliders on the left
    canvas    ->  an ipywidgets `Output` area on the right holding the figure
    redraw()  ->  clears the canvas and draws the figure from the current
                  control values; re-run automatically whenever a control moves

`_build_interactive` below wires those three pieces together, so the individual
plotting functions only have to say *what* to draw, never *how* to hook up the
widgets.

A note on widget toolkits: everything here uses **ipywidgets**, never
`matplotlib.widgets`. Matplotlib's own `Slider` needs an interactive drawing
backend, which Google Colab does not provide under its default inline backend --
the sliders appear but do nothing, and installing `ipympl` to supply one only
works after a session restart, because Colab imports matplotlib before the
install. ipywidgets works in Colab as it is, so it is the only toolkit used
here. This mirrors `utils_ex2/utils.py`, which was ported for the same reason.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import ipywidgets as widgets

# --------------------------------------------------------------------------
# Figure and widget styling
# --------------------------------------------------------------------------

# One CSS pixel expressed in inches, so figure sizes below can be written in
# pixels (matplotlib sizes figures in inches).
PX = 1 / plt.rcParams["figure.dpi"]

# Shared slider geometry, so every control column lines up the same way.
SLIDER_STYLE = {"description_width": "150px"}

# Positions along the cable at which every figure is evaluated, in mm.
X_MM = np.linspace(-10, 10, 1000)

# One colour per injection site, so a trace and the sliders that drive it are
# recognisably the same thing. These are matplotlib's first four default cycle
# colours written as hex, so the traces look exactly as they did before the
# ipywidgets port -- and hex is required, because a slider's handle_color trait
# rejects matplotlib's "tab:blue" spelling.
SITE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green.
SUM_COLOR = "#d62728"  # Red.


def slider_layout() -> widgets.Layout:
    """A fresh slider Layout, built by the call that displays it.

    A Layout is itself a widget model. Building one at import time puts it in
    the notebook's imports cell, and Colab does not reliably resolve a model
    from an earlier cell: the slider silently renders as nothing. Verified on
    Colab 2026-09-16 against notebook 1's copy of this same bug.

    Returns:
        widgets.Layout: The standard slider layout for this module.
    """
    return widgets.Layout(width="400px")


def _slider(
    value: float,
    min: float,
    max: float,
    step: float,
    description: str,
    color: str | None = None,
) -> widgets.FloatSlider:
    """Build a float slider with this module's standard styling.

    Args:
        value: Initial value.
        min: Lower bound.
        max: Upper bound.
        step: Slider increment.
        description: Label shown to the left of the slider. **Plain text only.**
            A widget description is rendered as HTML, not typeset, so `$a$`
            shows the dollar signs literally -- unlike a matplotlib slider
            label, which did render mathtext. Write units with Unicode (µ, Ω,
            ²) and subscripts as `r_m`.
        color: Optional handle colour, used to tie a slider to the trace it
            drives. Must be an HTML colour: the `handle_color` trait rejects
            matplotlib's "tab:blue" spelling.

    Returns:
        widgets.FloatSlider: The configured slider.
    """
    style = dict(SLIDER_STYLE)
    if color is not None:
        style["handle_color"] = color
    return widgets.FloatSlider(
        value=value,
        min=min,
        max=max,
        step=step,
        description=description,
        continuous_update=False,  # Redraw on release, not during the drag.
        style=style,
        layout=slider_layout(),
    )


def _build_interactive(controls, draw, buttons=None):
    """Wire a set of widgets to a drawing function and display them.

    This is the single place where the controls-canvas-redraw pattern described
    in the module docstring is implemented.

    Args:
        controls: Widgets to show, top to bottom, in the left-hand column.
        draw: Zero-argument callable that draws one figure using the current
            widget values. It should create its figure and leave it open;
            showing and clearing are handled here.
        buttons: Optional list of (label, values) pairs. Each becomes a button
            below the controls that sets those widget values and redraws once,
            which is how both the "Reset" and the preset buttons are made.

    Returns:
        The redraw callable, in case a caller needs to trigger a redraw itself.
    """
    canvas = widgets.Output()

    def redraw(*_):
        with canvas:
            clear_output(wait=True)  # Replace the old figure, do not stack.
            draw()
            plt.show()
            # Every redraw builds a fresh figure; without this the old ones
            # stay open and accumulate for as long as the notebook runs.
            plt.close()

    for control in controls:
        control.observe(redraw, "value")

    column = list(controls)
    for label, values in buttons or []:
        button = widgets.Button(description=label, button_style="info")

        def on_click(_, values=values):
            # Setting the values retriggers each observer, so silence them
            # while they are applied and redraw exactly once at the end.
            for widget in values:
                widget.unobserve(redraw, "value")
            for widget, value in values.items():
                widget.value = value
            for widget in values:
                widget.observe(redraw, "value")
            redraw()

        button.on_click(on_click)
        column.append(button)

    display(widgets.HBox([widgets.VBox(column), canvas]))
    redraw()  # Draw once immediately, so the figure is never blank on arrival.
    return redraw


# --------------------------------------------------------------------------
# The passive cable: maths
# --------------------------------------------------------------------------

# Defaults shared by both cable figures, in SI units except where noted.
I_E_DEFAULT = 1e-6  # Injected current, in A.
A_DEFAULT = 2e-3  # Cable radius.
R_M_DEFAULT = 1e6  # Specific membrane resistance.
R_L_DEFAULT = 1e3  # Specific longitudinal resistance.


def cable_length_constant(a: float, r_m: float, r_L: float) -> float:
    """Length constant of a passive cable, in mm.

    The distance over which a steady voltage decays by a factor of e. A wider
    cable, or a leakier one, spreads voltage further.

    Args:
        a: Cable radius.
        r_m: Specific membrane resistance.
        r_L: Specific longitudinal resistance.

    Returns:
        float: The length constant lambda, in mm.
    """
    return np.sqrt((a * r_m) / (2 * r_L))


def cable_input_resistance(a: float, r_m: float, r_L: float) -> float:
    """Longitudinal resistance seen from the injection site, in Ohm.

    Args:
        a: Cable radius.
        r_m: Specific membrane resistance.
        r_L: Specific longitudinal resistance.

    Returns:
        float: The input resistance R_L.
    """
    return r_L * cable_length_constant(a, r_m, r_L) / (np.pi * a**2)


def cable_voltage(
    x: np.ndarray,
    i_e: float,
    a: float,
    r_m: float,
    r_L: float,
    x_0: float = 0.0,
) -> np.ndarray:
    """Steady-state membrane potential along a cable, in mV.

    Half the injected current flows each way from the injection site, and the
    voltage decays exponentially with distance from it.

    Args:
        x: Positions along the cable, in mm.
        i_e: Injected current, in A.
        a: Cable radius.
        r_m: Specific membrane resistance.
        r_L: Specific longitudinal resistance.
        x_0: Position of the injection site, in mm.

    Returns:
        np.ndarray: Membrane potential at each position in `x`, in mV.
    """
    lambda_elc = cable_length_constant(a, r_m, r_L)
    peak = i_e * cable_input_resistance(a, r_m, r_L) / 2
    return peak * np.exp(-np.abs(x - x_0) / lambda_elc)


# --------------------------------------------------------------------------
# The passive cable: interactive figures
# --------------------------------------------------------------------------


def plot_cable_v(y_axis_lim: float = 80, plot_lambda: bool = False) -> None:
    """Interactive steady-state voltage along a cable injected at x = 0.

    The text box reports the length constant and the voltage at the injection
    site, which are the two numbers the assignments ask you to reason about.
    The "2 lambda" button jumps to a parameter set whose length constant puts
    x = 4 mm exactly two length constants from the injection site.

    Args:
        y_axis_lim: Upper limit of the voltage axis, in mV.
        plot_lambda: When True, mark x = 4 mm and the voltage reached there.
    """
    a_slider = _slider(A_DEFAULT * 1000, 0.5, 5, 0.1, "a (mm)")
    ie_slider = _slider(I_E_DEFAULT * 1e6, 0.01, 10, 0.01, "i_e (µA)")
    rm_slider = _slider(R_M_DEFAULT / 1e6, 0.1, 10, 0.1, "r_m (MΩ mm²)")
    rl_slider = _slider(R_L_DEFAULT / 1e3, 0.1, 5, 0.1, "r_L (kΩ mm²)")

    def draw():
        # Back to the units the formulas are written in.
        a = a_slider.value / 1000
        i_e = ie_slider.value / 1e6
        r_m = rm_slider.value * 1e6
        r_L = rl_slider.value * 1e3

        lambda_elc = cable_length_constant(a, r_m, r_L)
        v = cable_voltage(X_MM, i_e, a, r_m, r_L)
        v_0 = i_e * cable_input_resistance(a, r_m, r_L) / 2

        _, ax = plt.subplots(1, 1, figsize=(800 * PX, 400 * PX))
        ax.plot(X_MM, v, label="Membrane potential")
        ax.axhline(y=20, color="r", linestyle="--", label="Threshold 20 mV")
        if plot_lambda:
            ax.axvline(x=4, color="b", linestyle="--", label="V(x=4 mm)")
            ax.axhline(
                y=cable_voltage(np.array([4.0]), i_e, a, r_m, r_L)[0],
                color="b",
                linestyle="--",
            )
        ax.text(
            0.2,
            0.65,
            r"$\lambda$" + f" = {lambda_elc:.2f} mm\n" + r"$V_m(0)$" + f" = {v_0:.2f} mV",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel(r"$v$ (mV)")
        ax.set_ylim(0, y_axis_lim)
        ax.set_xticks(np.arange(-10, 11, 1))
        ax.set_title("Membrane potential along the cable")
        ax.grid()
        ax.legend()

    _build_interactive(
        [a_slider, ie_slider, rm_slider, rl_slider],
        draw,
        buttons=[
            (
                "Reset",
                {
                    a_slider: A_DEFAULT * 1000,
                    ie_slider: I_E_DEFAULT * 1e6,
                    rm_slider: R_M_DEFAULT / 1e6,
                    rl_slider: R_L_DEFAULT / 1e3,
                },
            ),
            # The parameter set that puts x = 4 mm at two length constants.
            ("2$\\lambda$", {a_slider: 1, ie_slider: 0.39, rm_slider: 6.4, rl_slider: 0.8}),
        ],
    )


def plot_multi_injection(y_axis_lim: float = 80) -> None:
    """Interactive voltage along a cable injected at three sites at once.

    The three individual profiles and their sum are drawn together: because the
    cable is passive, the sum is exactly the arithmetic sum of the parts. The
    middle site stays at x = 0; the outer two can be moved.

    Args:
        y_axis_lim: Upper limit of the voltage axis, in mV.
    """
    x_defaults = [-1, 0, 4.5]
    i_defaults = [1e-6, 1e-6, 1e-6]

    a_slider = _slider(A_DEFAULT * 1000, 0.5, 5, 0.1, "a (mm)")
    i_sliders = [
        _slider(i_defaults[k] * 1e6, 0.01, 2, 0.01, f"i_{k} (µA)", SITE_COLORS[k])
        for k in range(3)
    ]
    x0_slider = _slider(x_defaults[0], -5, -0.5, 0.1, "x_0 (mm)", SITE_COLORS[0])
    x2_slider = _slider(x_defaults[2], 0.5, 5, 0.1, "x_2 (mm)", SITE_COLORS[2])
    rm_slider = _slider(R_M_DEFAULT / 1e6, 0.1, 5, 0.1, "r_m (MΩ mm²)")
    # Note the unit: this slider is in Ohm mm, not kOhm mm^2 as in plot_cable_v.
    rl_slider = _slider(R_L_DEFAULT, 10, 5000, 0.1, "r_L (Ω mm)")

    def draw():
        a = a_slider.value / 1000
        currents = [s.value / 1e6 for s in i_sliders]
        r_m = rm_slider.value * 1e6
        r_L = rl_slider.value
        sites = [x0_slider.value, 0, x2_slider.value]

        lambda_elc = cable_length_constant(a, r_m, r_L)
        v = [
            cable_voltage(X_MM, currents[k], a, r_m, r_L, x_0=sites[k])
            for k in range(3)
        ]

        _, ax = plt.subplots(1, 1, figsize=(800 * PX, 600 * PX))
        for k in range(3):
            ax.plot(X_MM, v[k], color=SITE_COLORS[k], label=rf"$v_{k}$")
        ax.plot(X_MM, sum(v), color=SUM_COLOR, label=r"$\sum v_k$")
        ax.text(
            0.3,
            0.8,
            r"$\lambda$" + f" = {lambda_elc:.2f} mm",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel(r"$v$ (mV)")
        ax.set_ylim(0, y_axis_lim)
        ax.set_xticks(np.arange(-10, 11, 1))
        ax.set_title("Membrane potential along the cable")
        ax.grid()
        ax.legend()

    controls = [a_slider, *i_sliders, x0_slider, x2_slider, rm_slider, rl_slider]
    _build_interactive(
        controls,
        draw,
        buttons=[
            (
                "Reset",
                {
                    a_slider: A_DEFAULT * 1000,
                    i_sliders[0]: i_defaults[0] * 1e6,
                    i_sliders[1]: i_defaults[1] * 1e6,
                    i_sliders[2]: i_defaults[2] * 1e6,
                    x0_slider: x_defaults[0],
                    x2_slider: x_defaults[2],
                    rm_slider: R_M_DEFAULT / 1e6,
                    rl_slider: R_L_DEFAULT,
                },
            )
        ],
    )


# --------------------------------------------------------------------------
# Integrate-and-no-fire soma
# --------------------------------------------------------------------------

# Simulated duration, in ms, at a 1 ms timestep.
SIMTIME_MS = 100

# When each synapse fires, in ms. Fixed: the assignment varies the weights and
# the conduction delays, not the input times.
INPUT_TIMES_MS = [
    [20, 50],  # Synapse 1.
    [30, 60],  # Synapse 2.
    [40, 80],  # Synapse 3.
]


def iplot_InoF_model() -> None:
    """Interactive integrate-and-no-fire soma summing three synaptic inputs.

    Each synaptic input arrives at the soma after its own conduction delay and
    adds its own weight to the membrane potential. Nothing resets the
    potential, hence "no fire": the trace is a running sum of everything that
    has arrived so far.
    """
    num_synapses = len(INPUT_TIMES_MS)

    def update_plot(weight1=5, weight2=10, weight3=15, delay1=5, delay2=10, delay3=15):
        weights = [weight1, weight2, weight3]
        delays = [delay1, delay2, delay3]

        potential = np.zeros(SIMTIME_MS)
        for t in range(1, SIMTIME_MS):
            # Nothing leaks away, so carry the previous value forward.
            potential[t] = potential[t - 1]
            for i in range(num_synapses):
                for spike_time in INPUT_TIMES_MS[i]:
                    if t == spike_time + delays[i]:
                        potential[t] += weights[i]

        _, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        axs[0].plot(np.arange(SIMTIME_MS), potential, color="b")
        axs[0].set_title("Integrate-and-No-Fire Neuron")
        axs[0].set_ylabel("Membrane Potential (mV)")
        axs[0].grid(False)

        for i in range(num_synapses):
            for spike_time in INPUT_TIMES_MS[i]:
                first = i == 0 and spike_time == INPUT_TIMES_MS[i][0]
                axs[1].scatter(
                    spike_time,
                    i,
                    marker="|",
                    color="black",
                    s=100,
                    label="Original input time" if first else "",
                )
                axs[1].scatter(
                    spike_time + delays[i],
                    i,
                    marker="|",
                    color="silver",
                    s=100,
                    label="Delayed input at soma" if first else "",
                )

        axs[1].set_title("Synaptic Input Raster Plot (Black: Original, Grey: Delayed)")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_xticks(np.arange(0, SIMTIME_MS + 1, 10))
        axs[1].set_yticks(np.arange(num_synapses))
        axs[1].set_yticklabels([f"Synapse {i + 1}" for i in range(num_synapses)])
        axs[1].grid(False)

        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    style = {"description_width": "initial"}
    widgets.interact(
        update_plot,
        weight1=widgets.FloatSlider(
            min=0, max=100, step=5, value=5, description="Synaptic  Weight 1:", style=style
        ),
        weight2=widgets.FloatSlider(
            min=0, max=100, step=5, value=10, description="Synaptic  Weight 2:", style=style
        ),
        weight3=widgets.FloatSlider(
            min=0, max=100, step=5, value=15, description="Synaptic  Weight 3:", style=style
        ),
        delay1=widgets.IntSlider(
            min=0, max=40, step=1, value=5, description="Synaptic Delay 1 (ms)", style=style
        ),
        delay2=widgets.IntSlider(
            min=0, max=40, step=1, value=10, description="Synaptic Delay 2 (ms)", style=style
        ),
        delay3=widgets.IntSlider(
            min=0, max=40, step=1, value=15, description="Synaptic Delay 3 (ms)", style=style
        ),
    )
