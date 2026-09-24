"""Helper utilities for exercise session 2: passive membrane properties, part 1.

The notebook that uses this module (`2_passive_mem_properties_part_1.ipynb`)
covers two topics, and this file is organised to match them:

1. **Ionic equilibria** — the Nernst equation for a single ion species, the
   Nernst-Planck equation for the flux driven by concentration and voltage
   gradients, and the Goldman-Hodgkin-Katz (GHK) equation for the resting
   potential set by several ion species at once.
2. **The series RC circuit** — the simplest electrical model of a patch of
   passive membrane, solved both numerically (forward Euler) and analytically.

Every function here is either a small piece of *maths* you could reproduce with
pen and paper, or an *interactive figure*. The maths functions are the ones you
are asked about in the assignments; the plotting functions exist so you can
build intuition by moving a slider and watching what happens.

Interactive figures all share one structure, so once you have read one you have
read them all:

    controls  ->  a column of ipywidgets sliders / dropdowns on the left
    canvas    ->  an ipywidgets `Output` area on the right holding the figure
    redraw()  ->  clears the canvas and draws the figure from the current
                  control values; re-run automatically whenever a control moves

`_build_interactive` below wires those three pieces together, so the individual
plotting functions only have to say *what* to draw, never *how* to hook up the
widgets.

A note on widget toolkits: everything here uses **ipywidgets**, never
`matplotlib.widgets`. Matplotlib's own `Slider` needs an interactive drawing
backend, which Google Colab does not provide under its default inline backend --
the sliders appear but do nothing. ipywidgets works in Colab, so it is the only
toolkit used in this module.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.lines import Line2D
from scipy.constants import R, physical_constants

import ipywidgets as widgets

# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------

# Universal gas constant, in J/(mol K).
R_constant = R

# Faraday constant, in C/mol: the charge carried by one mole of elementary
# charges. Taken from scipy so the value matches CODATA rather than a rounded
# textbook figure.
F_constant = physical_constants["Faraday constant"][0]

# Body temperature in Kelvin (37 degrees C). Used as the default everywhere a
# temperature is needed, so numbers in the notebook are physiological.
T_default = 310

# --------------------------------------------------------------------------
# Ion properties
# --------------------------------------------------------------------------

# The three ion species this session deals with, in the order they appear in
# the dropdowns.
IONS = ["K⁺", "Na⁺", "Cl⁻"]

# Valence (signed charge number) per species. Chloride is the only anion here,
# and its negative valence is what flips the sign of its Nernst potential.
ION_VALENCE = {"K⁺": 1, "Na⁺": 1, "Cl⁻": -1}

# Diffusion coefficients in water, in m^2/s. Only the Nernst-Planck figure
# needs these.
ION_DIFFUSION = {"K⁺": 1.96e-9, "Na⁺": 1.33e-9, "Cl⁻": 2.03e-9}

# Typical mammalian concentrations, in mM, as (inside, outside) pairs. These
# are the values the sliders snap back to when you pick a different ion.
ION_CONCENTRATIONS = {
    "K⁺": (140.0, 5.0),
    "Na⁺": (12.0, 145.0),
    "Cl⁻": (4.0, 110.0),
}

# Upper slider bound per species, in mM. Potassium is given more headroom
# because its intracellular concentration is the largest of the three.
ION_CONCENTRATION_MAX = {"K⁺": 400.0, "Na⁺": 150.0, "Cl⁻": 150.0}

# One colour per species, reused by every figure so that a given ion always
# looks the same. "membrane" is the colour of the cell outline.
COLORS = {
    "K⁺": "#1f77b4",  # Blue.
    "Na⁺": "#ff7f0e",  # Orange.
    "Cl⁻": "#2ca02c",  # Green.
    "membrane": "#9467bd",  # Purple.
}

# --------------------------------------------------------------------------
# Figure and widget styling
# --------------------------------------------------------------------------

# One CSS pixel expressed in inches, so figure sizes below can be written in
# pixels (matplotlib sizes figures in inches).
PX = 1 / plt.rcParams["figure.dpi"]

# Standard sizes: wide-and-short for time courses, square for the schematic
# "cell with ions floating around it" figures.
FIGSIZE_TIMESERIES = (800 * PX, 600 * PX)
FIGSIZE_SCHEMATIC = (8, 8)

# Line width used by the time-course plots, in points.
LINEWIDTH = 400 * PX

# Shared slider geometry, so every control column lines up the same way.
SLIDER_STYLE = {"description_width": "150px"}


def slider_layout():
    """A fresh slider Layout, built by the call that displays it.

    A Layout is itself a widget model. Building one at import time puts it in
    the notebook's imports cell, and Colab does not reliably resolve a model
    from an earlier cell: the slider silently renders as nothing. Verified on
    Colab 2026-09-16 against notebook 1's copy of this same bug.
    """
    return widgets.Layout(width="400px")

# At most this many ions are drawn per species in the schematic figures. Real
# concentrations are in the hundreds of mM; drawing one dot per mM would be
# unreadable, so the count is capped purely for legibility.
MAX_IONS_DRAWN = 100

# Fixed seed for the ion scatter positions. Without it the ions would jump to
# new random positions on every redraw, which makes it impossible to see what
# actually changed when you move a slider.
SCATTER_SEED = 42


def _slider(value, min, max, step, description):
    """Build a float slider with this module's standard styling.

    Args:
        value: Initial value.
        min: Lower bound.
        max: Upper bound.
        step: Slider increment.
        description: Label shown to the left of the slider.

    Returns:
        widgets.FloatSlider: The configured slider.
    """
    return widgets.FloatSlider(
        value=value,
        min=min,
        max=max,
        step=step,
        description=description,
        continuous_update=False,  # Redraw on release, not during the drag.
        style=SLIDER_STYLE,
        layout=slider_layout(),
    )


def _build_interactive(controls, draw, reset_values=None):
    """Wire a set of widgets to a drawing function and display them.

    This is the single place where the controls-canvas-redraw pattern described
    in the module docstring is implemented.

    Args:
        controls: Widgets to show, top to bottom, in the left-hand column.
        draw: Zero-argument callable that draws one figure using the current
            widget values. It should create its figure and leave it open;
            showing and clearing are handled here.
        reset_values: Optional mapping of widget to the value a "Reset" button
            should restore. When given, a Reset button is appended below the
            controls.

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
    if reset_values:
        reset_button = widgets.Button(description="Reset", button_style="info")

        def on_reset(_):
            # Restoring the defaults retriggers each observer, so silence them
            # while the values are set and redraw exactly once at the end.
            for widget in reset_values:
                widget.unobserve(redraw, "value")
            for widget, value in reset_values.items():
                widget.value = value
            for widget in reset_values:
                widget.observe(redraw, "value")
            redraw()

        reset_button.on_click(on_reset)
        column.append(reset_button)

    display(widgets.HBox([widgets.VBox(column), canvas]))
    redraw()  # Draw once immediately, so the figure is never blank on arrival.
    return redraw


def _draw_membrane(ax):
    """Draw the circular cell outline shared by the schematic figures.

    Args:
        ax: Axes to draw on.
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color=COLORS["membrane"])


def _scatter_ions(ax, concentration_in, concentration_out, color):
    """Scatter dots inside and outside the cell to suggest concentrations.

    The dot counts are proportional to concentration but capped at
    `MAX_IONS_DRAWN`; this is a cartoon, not a quantitative rendering.

    Args:
        ax: Axes to draw on.
        concentration_in: Intracellular concentration, in mM.
        concentration_out: Extracellular concentration, in mM.
        color: Colour to draw this species in.
    """
    # Inside: uniformly scattered over a square well within the cell outline.
    num_in = min(int(concentration_in), MAX_IONS_DRAWN)
    ax.scatter(
        np.random.uniform(-0.7, 0.7, num_in),
        np.random.uniform(-0.7, 0.7, num_in),
        color=color,
        s=10,
        alpha=0.5,
    )

    # Outside: scattered in a ring just beyond the membrane.
    num_out = min(int(concentration_out), MAX_IONS_DRAWN)
    angle = np.random.uniform(0, 2 * np.pi, num_out)
    radius = np.random.uniform(1.1, 1.3, num_out)
    ax.scatter(
        radius * np.cos(angle), radius * np.sin(angle), color=color, s=10, alpha=0.5
    )


def _style_schematic(ax, title, ylim=(-1.5, 1.5)):
    """Apply the shared framing of the schematic figures.

    Args:
        ax: Axes to style.
        title: Figure title.
        ylim: Vertical limits, widened when a caption sits below the cell.
    """
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", "box")
    ax.axis("off")  # These are cartoons; axes ticks would be meaningless.
    ax.set_title(title)


# --------------------------------------------------------------------------
# 1. Ionic equilibria: the maths
# --------------------------------------------------------------------------


def nernst_potential(z, Cin, Cout, T):
    """Equilibrium potential of a single ion species (Nernst equation).

    This is the membrane potential at which the electrical force on the ion
    exactly cancels the diffusive force from its concentration gradient, so
    there is no net flux.

        E = (R T) / (z F) * ln(Cout / Cin)

    Args:
        z: Valence of the ion (+1 for K⁺ and Na⁺, -1 for Cl⁻).
        Cin: Intracellular concentration. Any unit, as long as it matches Cout.
        Cout: Extracellular concentration, in the same unit as Cin.
        T: Absolute temperature, in K.

    Returns:
        float: Equilibrium potential in mV. Note the trailing factor of 1000,
        which converts the volts produced by the formula into millivolts.
    """
    return (R_constant * T / (z * F_constant)) * np.log(Cout / Cin) * 1000


def nernst_planck_flux(D_i, C_i, z_i, dVdx, dCdx, T):
    """Ion flux driven by both a concentration and a voltage gradient.

    The Nernst-Planck equation splits the flux into a diffusive term (down the
    concentration gradient) and a drift term (along the electric field):

        J = -D * (dC/dx + (z F) / (R T) * C * dV/dx)

    Setting J to zero and solving recovers the Nernst equation above, which is
    why the two are introduced together.

    Args:
        D_i: Diffusion coefficient of the species, in m^2/s.
        C_i: Concentration at each position, as an array or scalar.
        z_i: Valence of the ion.
        dVdx: Voltage gradient at each position, in V/m.
        dCdx: Concentration gradient at each position.
        T: Absolute temperature, in K.

    Returns:
        The flux at each position, positive meaning "in the +x direction".
    """
    return -D_i * (dCdx + (z_i * F_constant / (R_constant * T)) * C_i * dVdx)


def ghk_potential(PK, PNa, PCl, K_in, K_out, Na_in, Na_out, Cl_in, Cl_out, T):
    """Resting potential set by K⁺, Na⁺ and Cl⁻ together (GHK equation).

    Where the Nernst equation describes one species in isolation, the
    Goldman-Hodgkin-Katz equation weights each species by how permeable the
    membrane is to it. The species with the largest permeability dominates,
    which is why the resting potential of a real neuron sits close to E_K.

    Note the asymmetry in the chloride terms: because Cl⁻ is negatively
    charged, its intracellular concentration appears in the numerator and its
    extracellular concentration in the denominator -- the opposite way round
    from the two cations.

    Args:
        PK: Relative permeability to K⁺.
        PNa: Relative permeability to Na⁺.
        PCl: Relative permeability to Cl⁻.
        K_in: Intracellular K⁺ concentration, in mM.
        K_out: Extracellular K⁺ concentration, in mM.
        Na_in: Intracellular Na⁺ concentration, in mM.
        Na_out: Extracellular Na⁺ concentration, in mM.
        Cl_in: Intracellular Cl⁻ concentration, in mM.
        Cl_out: Extracellular Cl⁻ concentration, in mM.
        T: Absolute temperature, in K.

    Returns:
        float: Resting membrane potential in mV.
    """
    numerator = PK * K_out + PNa * Na_out + PCl * Cl_in
    denominator = PK * K_in + PNa * Na_in + PCl * Cl_out
    return (R_constant * T / F_constant) * np.log(numerator / denominator) * 1000


# --------------------------------------------------------------------------
# 2. The series RC circuit: the maths
# --------------------------------------------------------------------------


def rc_euler(Vs, V0, R, C, dt, t):
    """Simulate the capacitor voltage of a series RC circuit, step by step.

    Each step uses the exact solution over one interval `dt` rather than a
    naive first-order update, which is why the result matches
    `rc_charging_exact` closely even at coarse time steps:

        V[i] = (1 - exp(-dt / RC)) * Vs + exp(-dt / RC) * V[i - 1]

    Args:
        Vs: Source voltage the capacitor charges towards, in V.
        V0: Initial capacitor voltage, in V.
        R: Resistance, in Ohm.
        C: Capacitance, in F.
        dt: Time step, in s.
        t: Time points. Only its length is used; the values set the length of
            the returned trace.

    Returns:
        numpy.ndarray: Capacitor voltage at each time point, in V.
    """
    V_t = np.zeros(t.shape)
    V_t[0] = V0
    decay = np.exp(-dt / (R * C))
    for i in range(1, len(t)):
        V_t[i] = (1 - decay) * Vs + decay * V_t[i - 1]
    return V_t


def rc_charging_exact(Vs, R, C, t):
    """Analytical capacitor voltage while charging from 0 V towards Vs.

    Args:
        Vs: Source voltage, in V.
        R: Resistance, in Ohm.
        C: Capacitance, in F.
        t: Time points, in ms. Converted to seconds internally.

    Returns:
        numpy.ndarray: Capacitor voltage at each time point, in V.
    """
    return Vs * (1 - np.exp(-(t / 1000) / (R * C)))


def rc_discharging_exact(V0, R, C, t):
    """Analytical capacitor voltage while discharging from V0 towards 0 V.

    Args:
        V0: Initial capacitor voltage, in V.
        R: Resistance, in Ohm.
        C: Capacitance, in F.
        t: Time points, in ms. Converted to seconds internally.

    Returns:
        numpy.ndarray: Capacitor voltage at each time point, in V.
    """
    return V0 * np.exp(-(t / 1000) / (R * C))


# --------------------------------------------------------------------------
# 3. The series RC circuit: interactive figures
# --------------------------------------------------------------------------

# Slider bounds shared by both RC figures, so the two behave identically.
_VS_RANGE = (0.0, 10.0, 0.1)  # Source voltage, in V.
_R_RANGE = (100.0, 100000.0, 100.0)  # Resistance, in Ohm.
_C_RANGE = (0.000001, 0.00001, 0.000001)  # Capacitance, in F.

# Simulation grid: 100 ms sampled every 1 ms.
_T_MS = np.arange(0, 100, 1)
_DT_S = 0.001


def _rc_controls(Vs_default, R_default, C_default):
    """Build the Vs / R / C slider trio used by both RC figures.

    Args:
        Vs_default: Initial source voltage, in V.
        R_default: Initial resistance, in Ohm.
        C_default: Initial capacitance, in F.

    Returns:
        tuple: The three sliders, in the order Vs, R, C.
    """
    vs_slider = _slider(Vs_default, *_VS_RANGE, "Vs (V)")
    r_slider = _slider(R_default, *_R_RANGE, "R (Ohm)")
    c_slider = _slider(C_default, *_C_RANGE, "C (F)")
    # Capacitances are microfarads, so the default "0.00" display would show
    # every value as zero.
    c_slider.readout_format = ".6f"
    return vs_slider, r_slider, c_slider


def plot_vc_slider(Vs_default=5, R_default=10000, C_default=0.000005, V0=0):
    """Interactive charging curve of a series RC circuit.

    Shows the capacitor voltage V_C rising towards the source voltage V_S, and
    lets you vary all three circuit parameters. The point of the figure is that
    R and C only ever appear as their product: doubling R and halving C leaves
    the curve unchanged.

    Args:
        Vs_default: Initial source voltage, in V.
        R_default: Initial resistance, in Ohm.
        C_default: Initial capacitance, in F.
        V0: Initial capacitor voltage, in V. Held fixed; it is not a slider.
    """
    vs_slider, r_slider, c_slider = _rc_controls(Vs_default, R_default, C_default)

    def draw():
        Vs, R, C = vs_slider.value, r_slider.value, c_slider.value

        _, ax = plt.subplots(1, 1, figsize=FIGSIZE_TIMESERIES)
        ax.plot(
            _T_MS,
            rc_euler(Vs, V0, R, C, _DT_S, _T_MS),
            label=r"$V_C$",
            linestyle="solid",
            linewidth=LINEWIDTH,
        )
        ax.plot(
            _T_MS,
            Vs * np.ones(_T_MS.shape),
            label=r"$V_S$",
            linestyle="dashed",
            linewidth=LINEWIDTH,
        )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"$V_C$ (V)")
        ax.set_ylim(0, 10)
        ax.set_title("Dependence of the RC circuit dynamics on its parameters")
        ax.legend(loc="upper right")
        ax.grid()

    _build_interactive(
        [vs_slider, r_slider, c_slider],
        draw,
        reset_values={
            vs_slider: Vs_default,
            r_slider: R_default,
            c_slider: C_default,
        },
    )


def plot_vc_tc_slider(plot_vertical_tc=False):
    """Interactive charging curve annotated with the time constant tau = RC.

    Same circuit as `plot_vc_slider`, with two constructions added that show
    where tau comes from:

    * the tangent to V_C at t = 0, which reaches V_S exactly at t = tau, and
    * the level 0.63 * V_S, which V_C crosses at t = tau.

    Args:
        plot_vertical_tc: If True, also draw a vertical line at t = tau, so the
            two constructions above can be read off against it directly.
    """
    Vs_default, R_default, C_default, V0_default = 5, 10000, 0.000001, 0
    vs_slider, r_slider, c_slider = _rc_controls(Vs_default, R_default, C_default)

    def draw():
        Vs, R, C = vs_slider.value, r_slider.value, c_slider.value

        _, ax = plt.subplots(1, 1, figsize=FIGSIZE_TIMESERIES)
        ax.plot(
            _T_MS,
            rc_euler(Vs, V0_default, R, C, _DT_S, _T_MS),
            label=r"$V_C$",
            linestyle="solid",
            linewidth=LINEWIDTH,
        )
        ax.plot(
            _T_MS,
            Vs * np.ones(_T_MS.shape),
            label=r"$V_S$",
            linestyle="dashed",
            linewidth=LINEWIDTH,
        )

        # Initial slope dV/dt = Vs / RC, drawn as a straight line from the
        # origin. Time is in ms here and seconds in the formula, hence /1000.
        ax.plot(
            _T_MS,
            (Vs / (R * C)) * (_T_MS / 1000),
            label="tangent",
            linestyle="dashed",
            linewidth=LINEWIDTH,
        )

        # The 1 - 1/e level that defines the time constant.
        ax.plot(
            _T_MS,
            Vs * (1 - np.exp(-1)) * np.ones(_T_MS.shape),
            label=r"$Vs*(1 - e^{-1})=0.63*Vs$",
            linestyle="dashed",
            linewidth=LINEWIDTH,
        )

        if plot_vertical_tc:
            # tau = RC is in seconds; the time axis is in ms.
            ax.axvline(x=R * C * 1000, color="r", linestyle="--", label=r"$\tau=RC$")

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"$V_C$ (V)")
        ax.set_ylim(0, 10)
        ax.set_title("The time constant of the RC circuit")
        ax.legend(loc="upper right")
        ax.grid()

    _build_interactive(
        [vs_slider, r_slider, c_slider],
        draw,
        reset_values={
            vs_slider: Vs_default,
            r_slider: R_default,
            c_slider: C_default,
        },
    )


def plot_cable_v():
    """Interactive steady-state voltage along a passive cable.

    Injecting a steady current i_e at x = 0 produces a voltage that decays
    exponentially in both directions with length constant lambda = sqrt(r_m /
    r_L). Widening the cable (larger radius a) lowers its longitudinal
    resistance and so lowers the peak voltage.

    Note:
        This belongs to the *next* session's material (the cable equation) and
        is not called by notebook 2. It is kept here so that this module stays
        a superset of the upstream file it mirrors.
    """
    i_e_default = 0.05e-9  # Injected current, in A.
    a_default = 2e-3  # Cable radius, in m.
    r_m_default = 1e6  # Specific membrane resistance.
    r_L_default = 1e3  # Specific longitudinal resistance.

    # Length constant, in mm: how far the voltage spreads before decaying by
    # a factor of e.
    lambda_elc = np.sqrt(r_m_default / r_L_default)
    x = np.linspace(-100, 100, 1000)

    a_slider = _slider(a_default * 1000, 1.0, 3.0, 0.1, "a (mm)")
    ie_slider = _slider(i_e_default * 1e9, 0.01, 0.1, 0.01, "i_e (nA)")

    def draw():
        a = a_slider.value / 1000  # Back to m.
        i_e = ie_slider.value / 1e9  # Back to A.

        # Input resistance of the cable seen from the injection site.
        R_l = r_L_default * lambda_elc / (np.pi * a**2)
        v = (i_e * R_l / 2) * np.exp(-np.abs(x) / lambda_elc)

        _, ax = plt.subplots(1, 1, figsize=(800 * PX, 400 * PX))
        ax.plot(x, v * 1e3, linewidth=LINEWIDTH)  # Volts to millivolts.
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel(r"$v$ (mV)")
        ax.set_ylim(0, 120)
        ax.set_title("Membrane potential along the cable")
        ax.grid()

    _build_interactive(
        [a_slider, ie_slider],
        draw,
        reset_values={a_slider: a_default * 1000, ie_slider: i_e_default * 1e9},
    )


# --------------------------------------------------------------------------
# 4. Ionic equilibria: interactive figures
# --------------------------------------------------------------------------


def _ion_legend(ion, extra=None):
    """Build the legend entries shared by the schematic figures.

    Args:
        ion: Name of the species being drawn, or None for the multi-ion figure.
        extra: Optional list of additional `Line2D` handles to append.

    Returns:
        list: Legend handles, ready to pass to `ax.legend`.
    """
    if ion is None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=name,
                markerfacecolor=COLORS[name],
                markersize=10,
            )
            for name in IONS
        ]
    else:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=ion,
                markerfacecolor=COLORS[ion],
                markersize=10,
            )
        ]
    handles.append(
        Line2D([0], [0], color=COLORS["membrane"], lw=2, label="Cell Membrane")
    )
    return handles + list(extra or [])


def _concentration_sliders():
    """Build the intracellular / extracellular concentration slider pair.

    Returns:
        tuple: The two sliders, inside first.
    """
    inside_default, outside_default = ION_CONCENTRATIONS["K⁺"]
    return (
        _slider(inside_default, 1, ION_CONCENTRATION_MAX["K⁺"], 1, "[Ion]_in (mM)"),
        _slider(outside_default, 1, ION_CONCENTRATION_MAX["K⁺"], 1, "[Ion]_out (mM)"),
    )


def _sync_to_ion(ion, inside_slider, outside_slider, redraw):
    """Snap the concentration sliders to physiological values for one species.

    Bounds are widened before the values are written and narrowed afterwards,
    so a new value is never clipped by a bound left over from the previous ion.

    Args:
        ion: Species that was just selected.
        inside_slider: Intracellular concentration slider.
        outside_slider: Extracellular concentration slider.
        redraw: The redraw callback, temporarily detached so that setting two
            values does not queue two extra redraws.
    """
    inside_default, outside_default = ION_CONCENTRATIONS[ion]
    upper = ION_CONCENTRATION_MAX[ion]

    sliders = (inside_slider, outside_slider)
    if redraw is not None:
        for slider in sliders:
            slider.unobserve(redraw, "value")

    for slider, value in zip(sliders, (inside_default, outside_default)):
        slider.max = max(slider.max, upper, value)
        slider.value = value
        slider.max = upper

    if redraw is not None:
        for slider in sliders:
            slider.observe(redraw, "value")


def nernst_interactive_plot():
    """Interactive Nernst potential for one ion species at a time.

    Pick a species, then move its intracellular and extracellular
    concentrations apart and watch the equilibrium potential printed under the
    cell. Selecting a different species snaps the concentrations back to
    physiological values for that ion.
    """
    ion_dropdown = widgets.Dropdown(
        options=IONS,
        value="K⁺",
        description="Select Ion:",
        style=SLIDER_STYLE,
        layout=slider_layout(),
    )
    Cin_slider, Cout_slider = _concentration_sliders()
    T_slider = _slider(T_default, 273, 373, 1, "Temperature (K)")

    # Filled in once `_build_interactive` has created the redraw callback; the
    # dropdown handler below needs it to suppress duplicate redraws.
    state = {}

    def on_ion_change(*_):
        _sync_to_ion(ion_dropdown.value, Cin_slider, Cout_slider, state.get("redraw"))

    # Registered before `_build_interactive` attaches its own handler, so the
    # sliders are already correct by the time the figure is redrawn.
    ion_dropdown.observe(on_ion_change, "value")

    def draw():
        ion = ion_dropdown.value
        E_ion = nernst_potential(
            ION_VALENCE[ion], Cin_slider.value, Cout_slider.value, T_slider.value
        )

        _, ax = plt.subplots(figsize=FIGSIZE_SCHEMATIC)
        _draw_membrane(ax)
        np.random.seed(SCATTER_SEED)
        _scatter_ions(ax, Cin_slider.value, Cout_slider.value, COLORS[ion])

        _style_schematic(ax, f"Ion Distribution for {ion}\n")
        ax.legend(handles=_ion_legend(ion), loc="upper right", bbox_to_anchor=(1.3, 1))
        ax.text(
            0,
            -1.4,
            f"Nernst Potential for {ion}: {E_ion:.2f} mV",
            fontsize=12,
            ha="center",
        )
        plt.tight_layout()

    state["redraw"] = _build_interactive(
        [ion_dropdown, Cin_slider, Cout_slider, T_slider], draw
    )


def nernst_planck_interactive_plot():
    """Interactive Nernst-Planck flux across the membrane.

    Adds a membrane potential slider to the previous figure. The arrow shows
    the direction of the net flux: at the Nernst potential for the selected
    species the two driving forces cancel and the flux passes through zero, so
    the arrow flips as you sweep V_mem through that value.
    """
    ion_dropdown = widgets.Dropdown(
        options=IONS,
        value="K⁺",
        description="Select Ion:",
        style=SLIDER_STYLE,
        layout=slider_layout(),
    )
    Cin_slider, Cout_slider = _concentration_sliders()
    V_mem_slider = _slider(-70e-3, -0.1, 0.1, 1e-3, "V_mem (V)")
    V_mem_slider.readout_format = ".3f"
    T_slider = _slider(T_default, 273, 373, 1, "Temperature (K)")

    state = {}

    def on_ion_change(*_):
        _sync_to_ion(ion_dropdown.value, Cin_slider, Cout_slider, state.get("redraw"))

    ion_dropdown.observe(on_ion_change, "value")

    def draw():
        ion = ion_dropdown.value
        color = COLORS[ion]

        # Model the membrane as a 10 nm slab with the potential and the
        # concentration both varying linearly across it.
        x = np.linspace(0, 1e-8, 100)
        V = np.linspace(0, V_mem_slider.value, len(x))  # Outside is 0 V.
        C = np.linspace(Cout_slider.value, Cin_slider.value, len(x))

        J = nernst_planck_flux(
            ION_DIFFUSION[ion],
            C,
            ION_VALENCE[ion],
            np.gradient(V, x),
            np.gradient(C, x),
            T_slider.value,
        )
        avg_J = np.mean(J)

        _, ax = plt.subplots(figsize=FIGSIZE_SCHEMATIC)
        _draw_membrane(ax)
        np.random.seed(SCATTER_SEED)
        _scatter_ions(ax, Cin_slider.value, Cout_slider.value, color)

        # Draw the flux arrow crossing the membrane, pointing outwards for a
        # negative average flux and inwards otherwise.
        arrow_length = 0.2
        start_radius = 0.9 if avg_J < 0 else 1.1
        end_radius = (
            start_radius + arrow_length if avg_J < 0 else start_radius - arrow_length
        )
        ax.arrow(
            start_radius,
            0,
            end_radius - start_radius,
            0,
            head_width=0.05,
            head_length=0.05,
            fc=color,
            ec=color,
            linewidth=2,
        )

        _style_schematic(ax, f"Ion Flux Across Neuron Membrane for {ion}")
        flux_handle = Line2D(
            [0], [0], marker=(3, 0, 0), color=color, label=f"{ion} Flux", markersize=15
        )
        ax.legend(
            handles=_ion_legend(ion, [flux_handle]),
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
        )
        ax.text(
            0, -1.4, f"Average Flux: {avg_J:.2e} mol/(m²·s)", fontsize=12, ha="center"
        )
        plt.tight_layout()

    state["redraw"] = _build_interactive(
        [ion_dropdown, Cin_slider, Cout_slider, V_mem_slider, T_slider], draw
    )


def ghk_interactive_plot():
    """Interactive resting potential set by K⁺, Na⁺ and Cl⁻ together.

    All three species are drawn at once, and each has its own concentration
    pair and permeability. Starting from the defaults, raising P_Na pulls the
    membrane potential away from E_K and towards E_Na -- the same movement that
    underlies the rising phase of an action potential.
    """
    concentration_sliders = {}
    for ion in IONS:
        inside_default, outside_default = ION_CONCENTRATIONS[ion]
        upper = ION_CONCENTRATION_MAX[ion]
        concentration_sliders[ion] = (
            _slider(inside_default, 1, upper, 1, f"[{ion}]_in (mM)"),
            _slider(outside_default, 1, upper, 1, f"[{ion}]_out (mM)"),
        )

    # Relative permeabilities. The defaults are the classic squid-axon ratios,
    # in which the membrane is far more permeable to K⁺ than to anything else.
    permeability_sliders = {
        "K⁺": _slider(1, 0, 1, 0.01, "P_K"),
        "Na⁺": _slider(0.03, 0, 1, 0.01, "P_Na"),
        "Cl⁻": _slider(0.1, 0, 1, 0.01, "P_Cl"),
    }
    T_slider = _slider(T_default, 273, 373, 1, "Temperature (K)")

    def draw():
        values = {
            ion: (sliders[0].value, sliders[1].value)
            for ion, sliders in concentration_sliders.items()
        }
        T = T_slider.value

        E = {
            ion: nernst_potential(ION_VALENCE[ion], values[ion][0], values[ion][1], T)
            for ion in IONS
        }
        V_m = ghk_potential(
            permeability_sliders["K⁺"].value,
            permeability_sliders["Na⁺"].value,
            permeability_sliders["Cl⁻"].value,
            values["K⁺"][0],
            values["K⁺"][1],
            values["Na⁺"][0],
            values["Na⁺"][1],
            values["Cl⁻"][0],
            values["Cl⁻"][1],
            T,
        )

        _, ax = plt.subplots(1, 1, figsize=FIGSIZE_SCHEMATIC)
        _draw_membrane(ax)
        np.random.seed(SCATTER_SEED)
        for ion in IONS:
            _scatter_ions(ax, values[ion][0], values[ion][1], COLORS[ion])

        # The lower limit is extended to leave room for the caption below.
        _style_schematic(ax, "Ion Distribution Across Neuron Membrane", ylim=(-1.8, 1.5))
        ax.legend(handles=_ion_legend(None), loc="upper right", bbox_to_anchor=(1.3, 1))
        ax.text(
            0,
            -1.6,
            (
                f"Membrane Potential (GHK): {V_m:.2f} mV\n"
                f"Nernst Potentials:\n"
                f"Eₖ⁺: {E['K⁺']:.2f} mV\n"
                f"Eₙₐ⁺: {E['Na⁺']:.2f} mV\n"
                f"E_Cl⁻: {E['Cl⁻']:.2f} mV"
            ),
            fontsize=12,
            ha="center",
            va="top",
        )
        plt.tight_layout()

    controls = []
    for ion in IONS:
        controls.extend(concentration_sliders[ion])
    controls.extend(permeability_sliders[ion] for ion in IONS)
    controls.append(T_slider)

    _build_interactive(controls, draw)
