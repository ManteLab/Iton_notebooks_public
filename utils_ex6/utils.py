from itertools import product
import numpy as np
from scipy.integrate import trapezoid
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
from ipywidgets import interact, interact_manual, IntSlider, FloatSlider, IntRangeSlider, ToggleButton, ToggleButtons, Layout


def in_colab():
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


is_colab = in_colab()
continuous_update = not is_colab
if is_colab:
    from google.colab import output
    output.enable_custom_widget_manager()


def draw_figure(fig):
    if not is_colab:
        fig.canvas.draw_idle()
    else:
        plt.show()


def generate_sims(C, k, alpha, sigma_a, sigma_s, lambda_, n_sim=100, tau=100, dt_total=11 / 85):
    dt = dt_total / tau

    # discretize C
    if isinstance(k, np.ndarray):
        C_scaled = np.repeat(C * k[:, np.newaxis], tau, axis=1)
        n_sim = len(k)
    else:
        C_scaled = np.repeat(C * k, tau)[np.newaxis, :]
    
    T = C_scaled.shape[-1]

    # noise terms
    xiR = np.random.randn(n_sim) * alpha / k
    xiL = np.random.randn(n_sim) * alpha / k
    directional_noise = (
        xiR[:, np.newaxis] * (C_scaled > 0) +
        xiL[:, np.newaxis] * (C_scaled < 0)
    )
    dW = np.sqrt(dt) * np.random.randn(n_sim, T)
    eta = 1 + np.random.randn(n_sim, T) * (sigma_s * np.sqrt(tau))

    # accumulated evidence
    a = np.zeros((n_sim, T + 1))
    mE = np.zeros((n_sim, T + 1))
    for t in range(T):
        a[:, t + 1] = a[:, t] + (
            directional_noise[:, t] * C_scaled[:, t] * (dt_total / tau) +
            lambda_ * a[:, t] * (dt_total / tau) +
            sigma_a * dW[:, t] +
            eta[:, t] * C_scaled[:, t] * (dt_total / tau)
        )

        # momentary evidence
        mE[:, t+1] = eta[:, t] * C_scaled[:, t] * (dt_total / tau) + lambda_ * a[:, t] * (dt_total / tau)

    return a[:, 1:], mE, tau, dt


def generate_sims_conditions(ks, directions, sim_parameters, num_sims_per_condition):
    simulation_combinations = list(product(ks, directions))

    a_all = []
    mE_all = []
    k_idx_all = []
    direction_all = []
    for idx, (k, direction) in enumerate(simulation_combinations):
        C = sim_parameters['C'] * direction
        dir_label = 1 if direction == 1 else 0

        a_temp, mE_temp, tau, dt = generate_sims(**{
            **sim_parameters,
            'C': C,
            'k': k,
            'n_sim': num_sims_per_condition
        })

        # subsample at every tau steps
        a_sampled = a_temp[:, tau-1::tau]
        mE_sampled = mE_temp[:, tau-1::tau] / dt

        a_all.append(a_sampled)
        mE_all.append(mE_sampled)
        k_idx_all.extend([k] * num_sims_per_condition)
        direction_all.extend([dir_label] * num_sims_per_condition)

    a_all = np.vstack(a_all)
    mE_all = np.vstack(mE_all)
    k_idx_all = np.array(k_idx_all)
    direction_all = np.array(direction_all)

    choices = (a_all > 0).astype(int)  # 1 is right, 0 is left
    is_correct = (choices == direction_all[:, np.newaxis]).astype(int)

    time = np.arange(len(C))

    return time, a_all, mE_all, k_idx_all, choices, is_correct


def plot_sims(C_size=11, num_sims=30 if not is_colab else 5):
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(figsize=(6.5, 5))
    
    evidence_line = axes.plot([], [], color='C2', alpha=1)[0]
    sim_lines = []
    for i in range(num_sims):
        sim_line = axes.plot([], [], color='C0', alpha=0.3)[0]
        sim_lines += [sim_line]

    axes.set(
        title=f"{num_sims} Simulations",
        ylabel="value",
        xlabel="time $t$",
        xlim=(0, 11),
        ylim=(-1.5, 1.5)
    )

    plt.axhline(0., color='black', alpha=0.3)
    plt.tight_layout()
    
    legend_elements = [
        Line2D([], [], color='C2', label='evidence pulse'),
        Line2D([], [], color='C0', label='accumulator $a$ (decision: right)'),
        Line2D([], [], color='C1', label='accumulator $a$ (decision: left)')
    ]
    axes.legend(handles=legend_elements, loc='upper right')

    random_seed = 42

    def update_plot(C_dir, C, k, alpha, sigma_a, sigma_s, lambda_, fixed_noise):
        if fixed_noise == 'redraw noise':
            nonlocal random_seed
            random_seed = np.random.randint(0, 2**32)
        np.random.seed(random_seed)
        
        C = np.concatenate([np.zeros(C[0]), np.ones(C[1] - C[0]), np.zeros(C_size - C[1])])
        C *= 1 if C_dir == 'pulse right' else -1

        sims, *_ = generate_sims(C, k, alpha, sigma_a, sigma_s, lambda_, n_sim=num_sims)

        for sim, sim_line in zip(sims, sim_lines):
            sim_line.set_data(np.linspace(0, len(C), len(sim)), sim)
            sim_line.set_color('C0' if sim[-1] > 0 else 'C1')
        
        evidence_line.set_data(np.linspace(0., len(C), len(C) * 1_000), np.repeat(C, 1_000) * k)

        draw_figure(fig)
    
    style = {'description_width': '150px'}
    layout = Layout(width='600px')
    sliders = {
        'C_dir': ToggleButtons(options=['pulse left', 'pulse right'], value='pulse right', description=' '),
        'C': IntRangeSlider(min=0, max=C_size, value=[3, 7], description='evidence pulse timing', style=style, layout=layout, continuous_update=continuous_update),
        'k': FloatSlider(min=1e-6, max=1., step=0.01, value=0.5, description='coherence', style=style, layout=layout, continuous_update=continuous_update),
        'sigma_s': FloatSlider(min=0, max=3, step=0.01, value=0., description='fast noise (input)', style=style, layout=layout, continuous_update=continuous_update),
        'alpha': FloatSlider(min=0, max=1, step=0.01, value=0., description='slow noise (brain)', style=style, layout=layout, continuous_update=continuous_update),
        'sigma_a': FloatSlider(min=0, max=1, step=0.01, value=0., description='fast inner noise (brain)', style=style, layout=layout, continuous_update=continuous_update),
        'lambda_': FloatSlider(min=-5, max=5, step=0.01, value=0., description='leakiness', style=style, layout=layout, continuous_update=continuous_update),
        'fixed_noise': ToggleButtons(options=['fix noise', 'redraw noise'], value='fix noise', description=' '),
    }

    interact(update_plot, **sliders)


def update_errorbar(err_container, x, y, yerr):
    err_container.lines[0].set_data(x, y)
    linecol = err_container.lines[2][0]

    segments = []
    for xi, yi, yerri in zip(x, y, yerr):
        segments.append([[xi, yi - yerri], [xi, yi + yerri]])

    linecol.set_segments(segments)


def plot_model_free_analysis_conditions(C, ks, num_sims_per_condition=2_000):
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    accuracy_lines = [axes[0].errorbar([], [], yerr=[], label=f'$k = {k}$') for k in ks]
    kernel_lines = [axes[1].plot([], [], label=f'$k = {k}$')[0] for k in ks]

    axes[0].set(
        title="accuracy",
        xlabel="$t$",
        xlim=(0, len(C) - 1),
        ylim=(0, 1)
    )
    axes[0].legend(loc='lower right', fontsize='small')

    axes[1].set(
        title="psychophysical kernel",
        xlabel="$t$",
        ylim=(-3, 3)
    )
    axes[1].legend(loc='lower left', fontsize='small')
    
    fig.tight_layout()

    def update_plot(sigma_s, alpha, sigma_a, lambda_):

        sim_parameters = {
            'C': C,
            'sigma_s': sigma_s,
            'alpha': alpha,
            'sigma_a': sigma_a,
            'lambda_': lambda_
        }

        directions = [1, -1]
        time, a_all, mE_all, k_idx_all, choices, is_correct = generate_sims_conditions(
            ks, directions, sim_parameters, num_sims_per_condition
        )

        for i, k in enumerate(ks):
            mask = (k_idx_all == k)
            is_corr_k = is_correct[mask, :]
            perf = is_corr_k.mean(axis=0)
            ci95 = 1.96 * is_corr_k.std(axis=0) / np.sqrt(mask.sum())

            update_errorbar(accuracy_lines[i], time, perf, yerr=ci95)

            psy_kernel = (
                mE_all[ (choices[:, -1] == 1) & mask ].mean(axis=0) -
                mE_all[ (choices[:, -1] != 1) & mask ].mean(axis=0)
            )

            kernel_lines[i].set_data(time, psy_kernel)

    style = {'description_width': '150px'}
    layout = Layout(width='600px')
    sliders = {
        'sigma_s': FloatSlider(min=0, max=5, step=0.01, value=0., description='fast noise (input)', style=style, layout=layout),
        'alpha': FloatSlider(min=0, max=1, step=0.01, value=0., description='slow noise (brain)', style=style, layout=layout),
        'sigma_a': FloatSlider(min=0, max=2, step=0.01, value=0., description='fast inner noise (brain)', style=style, layout=layout),
        'lambda_': FloatSlider(min=-5, max=5, step=0.01, value=0., description='leakiness', style=style, layout=layout)
    }
    
    interact_manual.options(manual_name='run simulations')(
        update_plot,
        **sliders
    )


def model_free_analysis(dataset):
    is_correct = dataset['choices'] == dataset['direction'].flatten()
    time = np.arange(dataset['a'].shape[1])

    perfs = []
    ci95s = []
    psy_kernels = []
    for k_idx in [1, 2, 3]:
        mask = (dataset['kIdx'].flatten() == k_idx)
        is_corr_k = is_correct[:, mask]
        perf = is_corr_k.mean(axis=1)
        ci95 = 1.96 * is_corr_k.std(axis=1) / np.sqrt(mask.sum())
        
        psy_kernel = (
            dataset['mE'][ (dataset['choices'][-1, :] == 1) & mask ].mean(axis=0) -
            dataset['mE'][ (dataset['choices'][-1, :] != 1) & mask ].mean(axis=0)
        )

        perfs += [perf]
        ci95s += [ci95]
        psy_kernels += [psy_kernel]

    return time, perfs, ci95s, psy_kernels


def plot_model_free_analysis_conditions_vs_baseline(baseline_data, num_sims_per_condition=2_000):
    C = np.concatenate(([0], np.ones(10)))
    ks = [0.2, 0.4, 0.8]
    
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    accuracy_lines = [axes[0].errorbar([], [], yerr=[], label=f'$k = {k}$') for k in ks]
    kernel_lines = [axes[1].plot([], [], label=f'$k = {k}$')[0] for k in ks]

    axes[0].set(
        title="accuracy",
        xlabel="$t$",
        xlim=(0, len(C) - 1),
        ylim=(0, 1)
    )

    axes[1].set(
        title="psychophysical kernel",
        xlabel="$t$",
        ylim=(-3, 3)
    )

    time, perfs, ci95s, psy_kernels = model_free_analysis(baseline_data)
    for i, (perf, ci95, psy_kernel) in enumerate(zip(perfs, ci95s, psy_kernels, strict=True)):
        axes[0].errorbar(time, perf, yerr=ci95, color=f'C{i}', label=f'$k = {ks[i]}$ (baseline)', linestyle='--', alpha=0.3)
        axes[1].plot(time, psy_kernel, color=f'C{i}', label=f'$k = {ks[i]}$ (baseline)', linestyle='--', alpha=0.3)
    
    axes[0].legend(loc='lower right', fontsize='small')
    axes[1].legend(loc='lower left', fontsize='small')
    fig.tight_layout()

    def update_plot(sigma_s, alpha, sigma_a, lambda_):

        sim_parameters = {
            'C': C,
            'sigma_s': sigma_s,
            'alpha': alpha,
            'sigma_a': sigma_a,
            'lambda_': lambda_
        }

        directions = [1, -1]
        time, a_all, mE_all, k_idx_all, choices, is_correct = generate_sims_conditions(
            ks, directions, sim_parameters, num_sims_per_condition
        )

        for i, k in enumerate(ks):
            mask = (k_idx_all == k)
            is_corr_k = is_correct[mask, :]
            perf = is_corr_k.mean(axis=0)
            ci95 = 1.96 * is_corr_k.std(axis=0) / np.sqrt(mask.sum())

            update_errorbar(accuracy_lines[i], time, perf, yerr=ci95)

            psy_kernel = (
                mE_all[ (choices[:, -1] == 1) & mask ].mean(axis=0) -
                mE_all[ (choices[:, -1] != 1) & mask ].mean(axis=0)
            )

            kernel_lines[i].set_data(time, psy_kernel)

    style = {'description_width': '150px'}
    layout = Layout(width='600px')
    sliders = {
        'sigma_s': FloatSlider(min=0, max=5, step=0.01, value=0., description='fast noise (input)', style=style, layout=layout),
        'alpha': FloatSlider(min=0, max=1, step=0.01, value=0., description='slow noise (brain)', style=style, layout=layout),
        'sigma_a': FloatSlider(min=0, max=2, step=0.01, value=0., description='fast inner noise (brain)', style=style, layout=layout),
        'lambda_': FloatSlider(min=-5, max=5, step=0.01, value=0., description='leakiness', style=style, layout=layout)
    }
    
    interact_manual.options(manual_name='run simulations')(
        update_plot,
        **sliders
    )


def bin_spikes(raw_spike_matrix, bin_size=50):
    num_bins = raw_spike_matrix.shape[1] // bin_size

    truncated_raw_spike_matrix = raw_spike_matrix[:, :num_bins * bin_size, :]
    binned_spike_matrix = truncated_raw_spike_matrix.reshape([
        truncated_raw_spike_matrix.shape[0],
        num_bins,
        -1,
        truncated_raw_spike_matrix.shape[2]
    ]).sum(axis=2)

    return binned_spike_matrix


def get_binned_spike_matrix(mat_data):
    raw_spike_matrix = mat_data['RawSpikeMatrix1'][:, 149:1000, :]
    binned_spike_matrix = bin_spikes(raw_spike_matrix)
    binned_spike_matrix = np.sqrt(binned_spike_matrix)
    time = np.arange(binned_spike_matrix.shape[1]) * 50
    return time, binned_spike_matrix


def plot_single_neuron(mat_data):
    time, binned_spike_matrix = get_binned_spike_matrix(mat_data)
    
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(figsize=(6.5, 4.5))

    neuron_line = axes.plot([], [])[0]

    axes.set(
        ylabel=r'$\sqrt{N_\mathrm{spikes}}$',
        xlabel='time [ms]',
        xlim=(0, 800)
    )

    def update_plot(neuron_idx):
        neuron_line.set_data(time, binned_spike_matrix.mean(axis=0)[:, neuron_idx])

        axes.relim()
        axes.autoscale(axis='y')
        axes.set_title(f'Neuron #{neuron_idx}', fontsize='small')
        draw_figure(fig)

    sliders = {
        'neuron_idx': IntSlider(min=0, max=binned_spike_matrix.shape[2] - 1, description='neuron #', layout=Layout(width='800px'), continuous_update=continuous_update)
    }

    interact(update_plot, **sliders)


def plot_neuron_by_choice(mat_data):
    time, binned_spike_matrix = get_binned_spike_matrix(mat_data)

    correct_trials_mask = (mat_data['targ_cho'].flatten() == mat_data['targ_cor'].flatten())
    right_choice = (mat_data['targ_cho'].flatten() == 1)
    
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    choices = ['right choice', 'left choice']
    correct_lines = []
    for choice in choices:
        correct_line = axes[0].plot([], [], label=choice)[0]
        correct_lines += [correct_line]

    incorrect_lines = []
    for choice in choices:
        incorrect_line = axes[1].plot([], [], label=choice)[0]
        incorrect_lines += [incorrect_line]

    axes[0].set(
        title='correct trials',
        ylabel=r'$\sqrt{N_\mathrm{spikes}}$',
        xlabel='time [ms]',
        xlim=(0, 800)
    )
    axes[1].set(
        title='incorrect trials',
        xlabel='time [ms]'
    )
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')

    def update_plot(neuron_idx):
        correct_lines[0].set_data(time, binned_spike_matrix[correct_trials_mask & right_choice].mean(axis=0)[:, neuron_idx])
        correct_lines[1].set_data(time, binned_spike_matrix[correct_trials_mask & ~right_choice].mean(axis=0)[:, neuron_idx])
        incorrect_lines[0].set_data(time, binned_spike_matrix[~correct_trials_mask & right_choice].mean(axis=0)[:, neuron_idx])
        incorrect_lines[1].set_data(time, binned_spike_matrix[~correct_trials_mask & ~right_choice].mean(axis=0)[:, neuron_idx])

        axes[0].relim()
        axes[1].relim()
        axes[0].autoscale(axis='y')
        axes[1].autoscale(axis='y')
        fig.suptitle(f'Neuron #{neuron_idx}', fontsize='small')
        draw_figure(fig)

    sliders = {
        'neuron_idx': IntSlider(min=0, max=binned_spike_matrix.shape[2] - 1, description='neuron #', layout=Layout(width='800px'), continuous_update=continuous_update)
    }

    interact(update_plot, **sliders)


def plot_neuron_by_coherence(mat_data):
    time, binned_spike_matrix = get_binned_spike_matrix(mat_data)

    correct_trials_mask = (mat_data['targ_cho'].flatten() == mat_data['targ_cor'].flatten())
    coherences = np.sort(
        np.unique(mat_data['dot_coh'])
    )
    coherences = coherences[[0, 3, 5]]
    
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    choices = ['right choice', 'left choice']
    correct_lines = []
    for coherence in coherences:
        correct_line = axes[0].plot([], [], label=f'{coherence = :.1%}')[0]
        correct_lines += [correct_line]

    incorrect_lines = []
    for coherence in coherences:
        incorrect_line = axes[1].plot([], [], label=f'{coherence = :.1%}')[0]
        incorrect_lines += [incorrect_line]

    axes[0].set(
        title='correct trials',
        ylabel=r'$\sqrt{N_\mathrm{spikes}}$',
        xlabel='time [ms]',
        xlim=(0, 800)
    )
    axes[1].set(
        title='incorrect trials',
        xlabel='time [ms]'
    )
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')

    def update_plot(neuron_idx):
        for i, coherence in enumerate(coherences):
            coherence_mask = (mat_data['dot_coh'].flatten() == coherence)
            correct_lines[i].set_data(time, binned_spike_matrix[correct_trials_mask & coherence_mask].mean(axis=0)[:, neuron_idx])
            incorrect_lines[i].set_data(time, binned_spike_matrix[~correct_trials_mask & coherence_mask].mean(axis=0)[:, neuron_idx])

        axes[0].relim()
        axes[1].relim()
        axes[0].autoscale(axis='y')
        axes[1].autoscale(axis='y')
        fig.suptitle(f'Neuron #{neuron_idx}', fontsize='small')
        draw_figure(fig)

    sliders = {
        'neuron_idx': IntSlider(min=0, max=binned_spike_matrix.shape[2] - 1, description='neuron #', layout=Layout(width='800px'), continuous_update=continuous_update)
    }

    interact(update_plot, **sliders)


def calculate_deltas(mat_data):
    time, binned_spike_matrix = get_binned_spike_matrix(mat_data)
    right_choice = (mat_data['targ_cho'].flatten() == 1)
    
    mean_spikes_right = binned_spike_matrix[right_choice].mean(axis=0)
    mean_spikes_left = binned_spike_matrix[~right_choice].mean(axis=0)
    
    deltas = (
        trapezoid(mean_spikes_right, axis=0) -
        trapezoid(mean_spikes_left, axis=0)
    )

    return deltas


def plot_deltas(deltas):
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    
    axes[0].hist(deltas, bins=16, range=(-4, 4))
    axes[1].hist(np.abs(deltas), bins=15, range=(0, 4.2))
    
    axes[0].set(
        ylabel='counts',
        xlabel=r'$\Delta$'
    )
    axes[1].set(
        xlabel=r'|$\Delta$|'
    )
    plt.tight_layout()


def plot_aggregated_neurons(mat_data):
    time, binned_spike_matrix = get_binned_spike_matrix(mat_data)
    right_choice = (mat_data['targ_cho'].flatten() == 1)
    mean_spikes_right = binned_spike_matrix[right_choice].mean(axis=0)
    mean_spikes_left = binned_spike_matrix[~right_choice].mean(axis=0)

    deltas = calculate_deltas(mat_data)
    
    if is_colab:
        plt.close()
    fig, axes = plt.subplots()

    lines = [
        axes.plot([], [], label='right choice')[0],
        axes.plot([], [], label='left choice')[0]
    ]

    axes.set(
        ylabel=r'$\sqrt{N_\mathrm{spikes}}$',
        xlabel='time [ms]',
        xlim=(0, 800)
    )
    axes.legend(loc='upper right')

    def update_plot(delta_threshold):
        lines[0].set_data(time, (mean_spikes_right * np.sign(deltas))[:, np.abs(deltas) > delta_threshold].mean(axis=1))
        lines[1].set_data(time, (mean_spikes_left * np.sign(deltas))[:, np.abs(deltas) > delta_threshold].mean(axis=1))

        axes.relim()
        axes.autoscale(axis='y')
        axes.set(
            title=f'|Δ| > {delta_threshold:.2f}'
        )
        draw_figure(fig)

    sliders = {
        'delta_threshold': FloatSlider(min=0, max=np.abs(deltas).max() - 1e-3, description='threshold |Δ|', layout=Layout(width='800px'), continuous_update=continuous_update)
    }

    interact(update_plot, **sliders)


def simulate_conditions(mat_data, alpha, sigma_a, sigma_s, lambda_):
    dot_coh = mat_data['dot_coh'].flatten()
    dot_dir = mat_data['dot_dir'].flatten()
    targ_cor = mat_data['targ_cor'].flatten()
    
    C = np.array([0] + [1]*16)
    
    dot_coh[dot_coh == 0] = 1e-12
    k = np.unique(dot_coh)
    
    # map directions: 0 -> 1 (right), 180 -> -1 (left)
    d = np.copy(dot_dir)
    d[dot_dir == 0] = 1
    d[dot_dir == 180] = -1

    a, _, tau, dt = generate_sims(np.outer(d, C), dot_coh, alpha, sigma_a, sigma_s, lambda_)
    a = a[:, tau-1::tau]

    # determine choices and correctness
    cho = (a[:, -1] > 0).astype(int)
    cho[cho == 0] = 2  # 2 is left, 1 is right
    isCorr = cho == targ_cor
    
    # separate correct and incorrect trials
    a_Cor = a[isCorr, :]
    d_Cor = d[isCorr]
    cho_Cor = cho[isCorr]
    coh_Cor = dot_coh[isCorr]
    
    a_Inc = a[~isCorr, :]
    d_Inc = d[~isCorr]
    cho_Inc = cho[~isCorr]
    coh_Inc = dot_coh[~isCorr]
    
    # plot average accumulation for correct trials by direction
    unq_dir = np.unique(d)

    means_a = []
    for dir_ in unq_dir:
        mean_a = np.mean(a_Cor[d_Cor == dir_, :], axis=0)
        means_a += [mean_a]

    return means_a


def plot_sims_conditions(mat_data):
    if is_colab:
        plt.close()
    fig, axes = plt.subplots(figsize=(6.5, 5))
    
    evidence_line = axes.plot([], [], color='C2', alpha=1)[0]
    sim_lines = []
    for choice in ['right choice', 'left choice']:
        sim_line = axes.plot([], [], label=choice)[0]
        sim_lines += [sim_line]

    axes.set(
        ylabel="mean $a$",
        xlabel="time $t$",
        xlim=(0, 800),
        ylim=(-0.5, .5)
    )

    plt.tight_layout()
    
    axes.legend(loc='upper right')

    random_seed = 42

    def update_plot(alpha, sigma_a, sigma_s, lambda_, fixed_noise):
        if fixed_noise == 'redraw noise':
            nonlocal random_seed
            random_seed = np.random.randint(0, 2**32)
        np.random.seed(random_seed)
        
        means_a = simulate_conditions(mat_data, alpha, sigma_a, sigma_s, lambda_)

        for mean_a, line in zip(means_a[::-1], sim_lines, strict=True):
            line.set_data(np.arange(len(mean_a)) * 50, mean_a)
        
        axes.relim()
        axes.autoscale(axis='y')
        draw_figure(fig)
    
    style = {'description_width': '150px'}
    layout = Layout(width='600px')
    sliders = {
        'sigma_s': FloatSlider(min=0, max=3, step=0.01, value=0., description='fast noise (input)', style=style, layout=layout, continuous_update=continuous_update),
        'alpha': FloatSlider(min=0, max=1, step=0.01, value=0., description='slow noise (brain)', style=style, layout=layout, continuous_update=continuous_update),
        'sigma_a': FloatSlider(min=0, max=1, step=0.01, value=0., description='fast inner noise (brain)', style=style, layout=layout, continuous_update=continuous_update),
        'lambda_': FloatSlider(min=-5, max=5, step=0.01, value=0., description='leakiness', style=style, layout=layout, continuous_update=continuous_update),
        'fixed_noise': ToggleButtons(options=['fix noise', 'redraw noise'], value='fix noise', description=' '),
    }

    interact(update_plot, **sliders)
