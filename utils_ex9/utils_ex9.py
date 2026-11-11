import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

def h_input(theta0, c, epsilon, N=50):
    theta_i = neuron_preference(N)
    if theta0 is None:
        h_ext = np.zeros_like(theta_i)
    else:
        h_ext = c * ((1 - epsilon) + epsilon * np.cos(2 * (theta_i - theta0)))
    return h_ext

def g(h):
    T = 0
    beta = 0.1
    # Create a tensor of the same shape as h to store the activation values
    g_activ = np.zeros_like(h)

    # Compute the conditions
    condition1 = (h <= T)
    condition2 = (h > T) & (h <= T + 1 / beta)

    # Apply conditions element-wise
    g_activ[condition1] = 0
    g_activ[condition2] = beta * (h[condition2] - T)
    g_activ[~condition1 & ~condition2] = 1

    return g_activ


def model_neuron(h, m, tau=5e-3, dt=1e-3):
    m = m + (-dt/tau) * m + dt/tau * g(h)  # Euler method step
    return m


def neuron_preference(N):
    theta_i = np.linspace(-np.pi/2, np.pi/2, N)
    return theta_i

def generate_connection_matrix(N, J0, J2, theta):
    Jij = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Jij[i, j] = -J0 + J2 * np.cos(2 * (theta[i] - theta[j]))
    return Jij

def h_inputConnections(Jij, N, h_ext, m):
    out = []
    for neuron in range(N):
        if len(h_ext.shape) == 1:
            out.append(np.dot(Jij[neuron, :], m) + h_ext[neuron])
    return np.array(out)



def simulate_network(mi, N, time_steps, c, epsilon, theta0, Jij, ct, add_noise=False):

    activity = []

    for t in range(time_steps):

        # Get h_ext and convert to tensor
        h_ext = ct+h_input(theta0, c, epsilon, N)

        if add_noise:
            # Generate noise and add it to h_ext
            eta = np.normal(0, 0.1, (1, N))
            h_ext += eta

        h =  h_inputConnections(Jij, N, h_ext, mi)

        # Compute new mi using tensors
        mi = model_neuron(h, mi)

        activity.append(mi)


    return np.array(activity), h_ext

def get_network(n_hidden):
    # Set up parameters
    n_inputs = 1
    J0 = 86
    J2 = 112
    c = 100
    epsilon = 0.9
    theta0 = 0
    time_steps = 30  # Number of time steps for simulation
    add_noise = False  # Whether to add noise to the simulation
    thetaij = neuron_preference(n_hidden)

    return n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise

def change_input_strength_no_connectivity():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initial activities to 0
    mi = np.zeros(n_hidden)

    # Define the update function
    def update(c):
        theta0 = 0
        time_steps = 50

        J0 = 0
        J2 = 0

        Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

        # Generate h_ext and activities with the current c value
        activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct=0, add_noise=add_noise)


        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Stem plot for h_ext
        axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
        axes[0].set_title('Contribution due to external input')
        axes[0].set_xlabel('Neurons')
        axes[0].set_ylabel('h_ext Values')
        axes[0].set_ylim(-25, 25)

        # Jij matrix plot
        im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
        axes[1].set_title('Jij Connectivity Matrix')
        axes[1].set_xlabel('Neurons')
        axes[1].set_ylabel('Neurons')
        fig.colorbar(im1, ax=axes[1])

        # Activities matrix plot
        im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[2].set_title('Neuron Activity')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Neurons')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()


    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=0.1, max=20.0, step=2, value=1.0))

def change_input_strength_with_connectivity():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initialize activity to 0
    mi = np.zeros(n_hidden)

    def update(c):

        theta0 = 0
        time_steps = 100

        J0 = 86
        J2 = 112


        Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

        # Generate h_ext and activities with the current c value
        activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 0, add_noise=add_noise)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Stem plot for h_ext
        axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
        axes[0].set_title('Contribution due to external Input')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('h_ext Values')
        axes[0].set_ylim(-25, 25)

        # Jij matrix plot
        im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
        axes[1].set_title('Jij Matrix')
        fig.colorbar(im1, ax=axes[1])

        # Activities matrix plot
        im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[2].set_title('Neuron Activity')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()

    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=0.1, max=20.0, step=2, value=1.0))


def change_input_orientation_with_connectivity():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initialize activity to 0
    mi = np.zeros(n_hidden)
    theta0 = 0
    time_steps = 100

    J0 = 86
    J2 = 112

    c = 10

    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

    # Generate h_ext and activities with the current c value
    activities_pre, h_ext_pre = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 0, add_noise=add_noise)

    mi = activities_pre[-1, :]

    # Define the update function
    def update(c):
        theta0 = 2 * math.pi / 3
        time_steps = 500

        # Generate h_ext and activities with the current c value
        activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 0, add_noise=add_noise)

        activities_all = np.concatenate((activities_pre[:-10,:], activities), axis=0)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Stem plot for h_ext
        axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
        #axes[0].stem(range(len(h_ext)), h_ext_pre+h_ext, linefmt='r-', markerfmt='ro', basefmt='r-')
        axes[0].set_title('Contribution due to external Input')
        axes[0].set_xlabel('Neurons')
        axes[0].set_ylabel('h_ext Values')
        axes[0].set_ylim(-100, 100)

        # Jij matrix plot
        im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
        axes[1].set_title('Jij Connectivity Matrix')
        fig.colorbar(im1, ax=axes[1])

        # Activities matrix plot
        im2 = axes[2].imshow(activities_all.T, cmap='viridis', aspect='auto')
        axes[2].set_title('Neuron Activity')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()



    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=0, max=100.0, step=20, value=0.0))


def change_input_orientation_without_connectivity():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initialize activity to 0
    mi = np.zeros(n_hidden)
    theta0 = 0
    time_steps = 100

    J0 = 86
    J2 = 112

    c = 10

    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

    # Generate h_ext and activities with the current c value
    activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 100, add_noise=add_noise)

    mi = activities[-1, :]

    # Define the update function
    def update(c):
        theta0 = 2 * math.pi / 3
        time_steps = 30

        J0 = 0
        J2 = 0

        Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

        # Generate h_ext and activities with the current c value
        activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 0, add_noise=add_noise)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Stem plot for h_ext
        axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
        axes[0].set_title('Contribution due to external Input')
        axes[0].set_xlabel('Neurons')
        axes[0].set_ylabel('h_ext Values')
        axes[0].set_ylim(-100, 100)

        # Jij matrix plot
        im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
        axes[1].set_title('Jij Matrix')
        fig.colorbar(im1, ax=axes[1])

        # Activities matrix plot
        im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto')
        axes[2].set_title('Neuron Activity')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()


    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=0, max=100.0, step=10, value=0))


def remove_stimulus_with_connectivity():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initialize activity to 0
    mi = np.zeros(n_hidden)
    theta0 = 0
    time_steps = 100

    J0 = 86
    J2 = 112

    c = 10

    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)
    activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 100, add_noise=add_noise)
    mi = activities[-1, :]

    # Define the update function
    def update(c):

        time_steps = 500

        J0 = 86
        J2 = 112

        activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=None, Jij = Jij, ct = 0, add_noise=add_noise)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Stem plot for h_ext
        axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
        axes[0].set_title('Contribution due to external Input')
        axes[0].set_xlabel('Neurons')
        axes[0].set_ylabel('h_ext Values')
        axes[0].set_ylim(-100, 100)

        # Jij matrix plot
        im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
        axes[1].set_title('Jij Matrix')
        fig.colorbar(im1, ax=axes[1])

        # Activities matrix plot
        im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto')
        axes[2].set_title('Neuron Activity')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()

    mi = activities[-1, :]


    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=50, max=100.0, step=5, value=50.0))


def remove_stimulus_without_connectivity():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initialize activity to 0
    mi = np.zeros(n_hidden)
    theta0 = 0
    time_steps = 100

    J0 = 86
    J2 = 112

    c = 10

    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)
    activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 100, add_noise=add_noise)
    mi = activities[-1, :]

    # Define the update function
    def update(c):

        time_steps = 500

        J0 = 0
        J2 = 0
        Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

        activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=None, Jij = Jij, ct = 0, add_noise=add_noise)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Stem plot for h_ext
        axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
        axes[0].set_title('Contribution due to external Input')
        axes[0].set_xlabel('Neurons')
        axes[0].set_ylabel('h_ext Values')
        axes[0].set_ylim(-100, 100)

        # Jij matrix plot
        im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
        axes[1].set_title('Jij Matrix')
        fig.colorbar(im1, ax=axes[1])

        # Activities matrix plot
        im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto')
        axes[2].set_title('Neuron Activity')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()

    mi = activities[-1, :]


    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=0.1, max=100.0, step=10, value=1.0))
