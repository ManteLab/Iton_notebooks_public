"""Helpers for notebook 1: a Hodgkin-Huxley neuron, its F-I curve, and spike sparsity.

The model is the classic squid giant axon of Hodgkin & Huxley (1952), written in the
modern convention where depolarisation is positive. Everything is in the units the
original paper used: mV, ms, uA/cm^2, uF/cm^2 and mS/cm^2.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Button, FloatSlider, IntSlider, Layout, Output, VBox

# Membrane and channel parameters (Hodgkin & Huxley 1952, squid giant axon).
C_M = 1.0  # Membrane capacitance, uF/cm^2.
G_NA = 120.0  # Maximal sodium conductance, mS/cm^2.
G_K = 36.0  # Maximal potassium conductance, mS/cm^2.
G_L = 0.3  # Leak conductance, mS/cm^2.
E_NA = 50.0  # Sodium reversal potential, mV.
E_K = -77.0  # Potassium reversal potential, mV.
E_L = -54.4  # Leak reversal potential, mV.
V_REST = -65.0  # Resting membrane potential, mV.

_SLIDER_STYLE = {"description_width": "initial"}


def _slider_layout():
    """A fresh slider Layout, built by the call that displays it.

    A Layout is itself a widget model. Building one at import time puts it in
    the notebook's imports cell, and Colab does not reliably resolve a model
    from an earlier cell: the slider silently renders as nothing. Verified on
    Colab 2026-09-16. utils_ex13's working sliders build theirs inline too.
    """
    return Layout(width="500px")


def _limit_ratio(numerator, denominator, limit: float):
    """Divide, substituting the analytic limit wherever the denominator vanishes."""
    denominator = np.asarray(denominator, dtype=float)
    singular = np.abs(denominator) < 1e-9
    # Divide by 1.0 at the singular points so numpy never evaluates 0/0.
    safe = np.where(singular, 1.0, denominator)
    return np.where(singular, limit, np.asarray(numerator, dtype=float) / safe)


def alpha_m(V):
    """Sodium activation opening rate in 1/ms; the 0/0 point at V = -40 mV is the limit 1.0."""
    u = np.asarray(V, dtype=float) + 40.0
    # -expm1(-x) is 1 - exp(-x), but stays accurate when x is tiny.
    return _limit_ratio(0.1 * u, -np.expm1(-u / 10.0), 1.0)


def beta_m(V):
    """Sodium activation closing rate in 1/ms."""
    return 4.0 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 18.0)


def alpha_h(V):
    """Sodium inactivation opening rate in 1/ms."""
    return 0.07 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 20.0)


def beta_h(V):
    """Sodium inactivation closing rate in 1/ms."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(V, dtype=float) + 35.0) / 10.0))


def alpha_n(V):
    """Potassium activation opening rate in 1/ms; the 0/0 point at V = -55 mV is the limit 0.1."""
    u = np.asarray(V, dtype=float) + 55.0
    return _limit_ratio(0.01 * u, -np.expm1(-u / 10.0), 0.1)


def beta_n(V):
    """Potassium activation closing rate in 1/ms."""
    return 0.125 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 80.0)


def steady_state(V):
    """Steady-state values (m, h, n) of the three gating variables at a held voltage."""
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    return m, h, n


def simulate_hh(I_ext, t_max: float = 50.0, dt: float = 0.01):
    """Integrate the Hodgkin-Huxley equations with forward Euler and return (t, V, m, h, n).

    `I_ext` may be a scalar or an array of injected currents in uA/cm^2. An array is
    broadcast over a trailing "current" axis, so a whole F-I sweep runs in one pass:
    the returned traces then have shape (n_steps,) + np.shape(I_ext).
    """
    scalar_input = np.ndim(I_ext) == 0
    current = np.atleast_1d(np.asarray(I_ext, dtype=float))
    n_steps = int(round(t_max / dt)) + 1
    t = np.arange(n_steps) * dt

    V = np.empty((n_steps, *current.shape))
    m = np.empty_like(V)
    h = np.empty_like(V)
    n = np.empty_like(V)

    # Start every neuron at rest, with its gates already at their steady-state values.
    V[0] = V_REST
    m[0], h[0], n[0] = steady_state(V_REST)

    for k in range(n_steps - 1):
        v, m_k, h_k, n_k = V[k], m[k], h[k], n[k]

        i_na = G_NA * m_k**3 * h_k * (v - E_NA)
        i_k = G_K * n_k**4 * (v - E_K)
        i_l = G_L * (v - E_L)

        V[k + 1] = v + dt * (current - i_na - i_k - i_l) / C_M
        m[k + 1] = m_k + dt * (alpha_m(v) * (1.0 - m_k) - beta_m(v) * m_k)
        h[k + 1] = h_k + dt * (alpha_h(v) * (1.0 - h_k) - beta_h(v) * h_k)
        n[k + 1] = n_k + dt * (alpha_n(v) * (1.0 - n_k) - beta_n(v) * n_k)

    if scalar_input:
        V, m, h, n = V[:, 0], m[:, 0], h[:, 0], n[:, 0]
    return t, V, m, h, n


def count_spikes(V, threshold: float = 0.0):
    """Count spikes as upward crossings of `threshold` along the time axis of a voltage trace."""
    above = np.asarray(V) > threshold
    return np.sum(above[1:] & ~above[:-1], axis=0)


def spike_times(t, V, threshold: float = 0.0):
    """Times in ms of the upward `threshold` crossings of a single voltage trace."""
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    above = V > threshold
    crossings = np.flatnonzero(above[1:] & ~above[:-1])
    if crossings.size == 0:
        return np.empty(0)
    # Interpolate linearly between the two samples that straddle the threshold, so a
    # spike time is not rounded to the simulation grid.
    fraction = (threshold - V[crossings]) / (V[crossings + 1] - V[crossings])
    return t[crossings] + fraction * (t[crossings + 1] - t[crossings])


def _isi_rate(times, t_end: float):
    """Firing rate in Hz from the mean interspike interval; 0 unless the neuron fires steadily."""
    if np.size(times) < 2:
        return 0.0
    isi = float(np.mean(np.diff(times)))
    # Just below rheobase the neuron fires a damped burst that dies out part way through
    # the run. A steadily firing neuron is always within one interval of its next spike,
    # so a final gap much larger than that means the firing stopped: report it as silent.
    if t_end - times[-1] > 1.5 * isi:
        return 0.0
    return 1000.0 / isi


def fi_curve(currents, t_max: float = 150.0, dt: float = 0.01, transient: float = 50.0):
    """Firing rate in Hz for each injected current, measured after discarding a transient.

    The rate is the reciprocal of the mean interspike interval, not a spike count over a
    window, so the curve is smooth rather than quantised in whole spikes. A current is
    reported as silent unless it makes the neuron fire steadily to the end of the run,
    which keeps the sub-rheobase arm at exactly 0 and stops the onset burst just below
    rheobase from registering as a rate.
    """
    scalar_input = np.ndim(currents) == 0
    currents = np.atleast_1d(np.asarray(currents, dtype=float))
    t, V, _, _, _ = simulate_hh(currents, t_max=t_max, dt=dt)
    steady = t >= transient
    t_steady = t[steady]
    V_steady = V[steady]
    rates = np.array([_isi_rate(spike_times(t_steady, V_steady[:, k]), t_steady[-1])
                      for k in range(currents.size)])
    return float(rates[0]) if scalar_input else rates


def iplot_action_potential(t_max: float = 50.0, dt: float = 0.01):
    """Interactive action potential: a slider over the injected current, with the gates underneath."""
    current_slider = FloatSlider(
        min=0.0, max=25.0, step=0.5, value=10.0,
        description="Injected current (uA/cm^2):",
        style=_SLIDER_STYLE, layout=_slider_layout(),
    )
    output = Output()

    def refresh_output():
        with output:
            output.clear_output(wait=True)
            t, V, m, h, n = simulate_hh(current_slider.value, t_max=t_max, dt=dt)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t, V, color="black", label="V")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Membrane potential (mV)")
            ax.set_ylim(-90, 60)
            ax.set_title(
                f"Hodgkin-Huxley neuron, I = {current_slider.value:.1f} uA/cm^2 "
                f"({count_spikes(V)} spikes)"
            )
            ax.grid(True, alpha=0.3)

            gates = ax.twinx()
            gates.plot(t, m, color="tab:red", alpha=0.7, label="m (Na activation)")
            gates.plot(t, h, color="tab:blue", alpha=0.7, label="h (Na inactivation)")
            gates.plot(t, n, color="tab:green", alpha=0.7, label="n (K activation)")
            gates.set_ylabel("Gating variable")
            gates.set_ylim(0, 1)

            handles, labels = ax.get_legend_handles_labels()
            g_handles, g_labels = gates.get_legend_handles_labels()
            ax.legend(handles + g_handles, labels + g_labels, loc="upper right")

            plt.tight_layout()
            plt.show()

    current_slider.observe(lambda _: refresh_output(), names="value")
    refresh_output()

    display(VBox([current_slider, output]))


def iplot_fi_curve(t_max: float = 150.0, dt: float = 0.01):
    """Interactive F-I curve: sweep the injected current, count spikes, plot firing rate."""
    max_current_slider = FloatSlider(
        min=5.0, max=50.0, step=1.0, value=20.0,
        description="Largest current in the sweep (uA/cm^2):",
        style=_SLIDER_STYLE, layout=_slider_layout(),
    )
    n_points_slider = IntSlider(
        min=5, max=40, step=1, value=30,
        description="Number of currents:",
        style=_SLIDER_STYLE, layout=_slider_layout(),
    )
    sweep_button = Button(description="Run sweep", button_style="success")
    output = Output()

    def refresh_output():
        with output:
            output.clear_output(wait=True)
            currents = np.linspace(0.0, max_current_slider.value, n_points_slider.value)
            rates = fi_curve(currents, t_max=t_max, dt=dt)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(currents, rates, marker="o", color="black")
            ax.set_xlabel("Injected current (uA/cm^2)")
            ax.set_ylabel("Firing rate (Hz)")
            ax.set_title("F-I curve of a Hodgkin-Huxley neuron")
            ax.grid(True, alpha=0.3)

            spiking = np.flatnonzero(rates > 0)
            if spiking.size:
                rheobase = currents[spiking[0]]
                ax.axvline(rheobase, color="tab:red", linestyle="--",
                           label=f"rheobase near {rheobase:.1f} uA/cm^2")
                ax.legend(loc="lower right")

            plt.tight_layout()
            plt.show()

    sweep_button.on_click(lambda _: refresh_output())
    refresh_output()

    display(VBox([max_current_slider, n_points_slider, sweep_button, output]))


def spike_sparsity(dataset, n_samples: int = 200):
    """Per-timestep fraction of active units in a spiking dataset, returned as (T,) and (n, T) arrays."""
    n_samples = min(n_samples, len(dataset))
    spikes = []
    for index in range(n_samples):
        item = dataset[index]
        sample = item[0] if isinstance(item, (tuple, list)) else item
        spikes.append(np.asarray(sample, dtype=float))
    spikes = np.stack(spikes)  # Shape (n_samples, T, units).

    per_sample = spikes.mean(axis=-1)  # Fraction of units active, per sample and timestep.
    return per_sample.mean(axis=0), per_sample


def plot_spike_sparsity(dataset, n_samples: int = 200, sample_index: int = 0):
    """Show how few units of a spiking dataset are active at any one timestep."""
    mean_active, per_sample = spike_sparsity(dataset, n_samples=n_samples)

    item = dataset[sample_index]
    sample = np.asarray(item[0] if isinstance(item, (tuple, list)) else item, dtype=float)
    n_timesteps, n_units = sample.shape
    timesteps = np.arange(n_timesteps)

    fig, (ax_raster, ax_rate) = plt.subplots(1, 2, figsize=(14, 5))

    unit, step = np.nonzero(sample.T)
    ax_raster.scatter(step, unit, marker="|", s=60, color="black")
    ax_raster.set_xlabel("Timestep")
    ax_raster.set_ylabel("Input unit")
    ax_raster.set_yticks(np.arange(n_units))
    ax_raster.set_xlim(0, n_timesteps)
    ax_raster.set_title(f"One sample: {int(sample.sum())} spikes "
                        f"in {n_timesteps * n_units} unit-timesteps")

    ax_rate.plot(timesteps, mean_active, color="black")
    overall = float(per_sample.mean())
    ax_rate.axhline(overall, color="tab:red", linestyle="--",
                    label=f"mean activity {overall:.1%}")
    ax_rate.set_xlabel("Timestep")
    ax_rate.set_ylabel("Fraction of units active")
    ax_rate.set_ylim(0, 1)
    ax_rate.set_xlim(0, n_timesteps)
    ax_rate.set_title(f"Averaged over {per_sample.shape[0]} samples")
    ax_rate.legend(loc="upper right")
    ax_rate.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
