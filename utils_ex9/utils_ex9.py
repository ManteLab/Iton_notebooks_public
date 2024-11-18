import pdb
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt
import math

import ipywidgets as widgets
from ipywidgets import interact

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_network():
    n_hidden, thetaij,J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    # Initial state
    mi = np.zeros(n_hidden)  

    
    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

    activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 0, add_noise=add_noise)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # Create a figure with 1 row and 3 columns

    # Plot h_ext using a stem plot in the first subplot
    axes[0].stem(range(len(h_ext)), h_ext, linefmt='b-', markerfmt='bo', basefmt='r-')
    axes[0].set_title('Contribution due to external input')
    axes[0].set_xlabel('Neurons')
    axes[0].set_ylabel('h_ext values')


    # Plot Jij matrix in the second subplot
    im1 = axes[1].imshow(Jij, cmap='viridis', aspect='auto')
    axes[1].set_title('Jij Connectivity Matrix')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_xlabel('Neurons')
    axes[1].set_ylabel('Neurons')

    # Plot activities matrix in the third subplot
    im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto')
    axes[2].set_title('Neuron activity')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Neurons')
    fig.colorbar(im2, ax=axes[2])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

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
    activities, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta0, Jij = Jij, ct = 100, add_noise=add_noise)

    mi = activities[-1, :]

    # Define the update function
    def update(c):
        theta0 = 2 * math.pi / 3
        time_steps = 500
            
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
        axes[1].set_title('Jij Connectivity Matrix')
        fig.colorbar(im1, ax=axes[1])
        
        # Activities matrix plot
        im2 = axes[2].imshow(activities.T, cmap='viridis', aspect='auto')
        axes[2].set_title('Neuron Activity')
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.show()



    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=50, max=100.0, step=5, value=50.0))


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
        axes[2].set_title('Activities Matrix')
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

    mi = activities[-1, :]


    # Use interact to create a slider for c
    interact(update, c=widgets.FloatSlider(min=0.1, max=100.0, step=10, value=1.0))


def zeros_initializations():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 50)

    theta0 = 0
    mi = np.zeros(n_hidden) 
    theta_i = np.linspace(-np.pi/2, np.pi/2, n_hidden)  # Neuron preferred orientations
    J0 = 8.6
    J1 = 11.2
    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

    c = 1.2
    epsilon = 0.9
    activities_list = []

    for i in range(len(theta_i)):
        activities1, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=i, Jij = Jij, ct = 0, add_noise=add_noise)
        activities_list.append(activities1.reshape(1, activities1.shape[0], activities1.shape[1]))

    activities = np.concatenate(activities_list, axis=0)

    trials, time, units = activities.shape
    print(f'trials = {trials:.0f}, time = {time:.0f}, units = {units:.0f}')

    reshaped_activities = activities.reshape(trials * time, units)  # Shape: (trials * time, units)

    # Perform PCA
    pca = PCA()
    pca.fit(reshaped_activities)

    # Get the explained variance
    explained_variance = pca.explained_variance_ratio_

    # Plot the explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.show()

    # Perform PCA with 3 components for visualization
    pca = PCA(n_components=3)
    pca_activities = pca.fit_transform(reshaped_activities)


    # Reshape the PCA-transformed data back to (trials, time, pcs)
    pca_activities_reshaped = pca_activities.reshape(trials, time, 3)

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / trials) for i in range(trials)]

    # Plot each trial in the 3D PCA space
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    for trial in range(trials):
        # Plot each trial as a line in the 3D space
        ax.plot(
            pca_activities_reshaped[trial, :, 0],
            pca_activities_reshaped[trial, :, 1],
            pca_activities_reshaped[trial, :, 2],
            color=colors[trial],
            label=f'Trial {trial+1}',
            alpha=0.7
        )

    # Set axis labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Trials Projected onto First 3 Principal Components')
    plt.show()


def random_initializations():
    n_hidden, thetaij, J0, J2, c, epsilon, theta0, time_steps, add_noise = get_network(n_hidden = 100)
    theta0 = 0
    mi = np.zeros(n_hidden) 
    theta_i = np.linspace(-np.pi/2, np.pi/2, n_hidden)  # Neuron preferred orientations
    J0 = 8.6
    J1 = 11.2
    Jij = generate_connection_matrix(n_hidden, J0, J2, thetaij)

    c = 1.2
    epsilon = 0.9
    activities_list = []

    for i in range(len(theta_i)):
        activities1, h_ext = simulate_network(mi, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta_i[i], Jij = Jij, ct = 0, add_noise=add_noise)
        activities_list.append(activities1.reshape(1, activities1.shape[0], activities1.shape[1]))

    activities = np.concatenate(activities_list, axis=0)

    trials, time, units = activities.shape
    print(f'trials = {trials:.0f}, time = {time:.0f}, units = {units:.0f}')

    reshaped_activities = activities.reshape(trials * time, units)  # Shape: (trials * time, units)
     # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(reshaped_activities)

    # Perform PCA with 3 components for visualization
    pca_activities = pca.fit_transform(reshaped_activities)
    pcs = pca.components_
    print('SS', )


    # Reshape the PCA-transformed data back to (trials, time, pcs)
    pca_activities_reshaped = pca_activities.reshape(trials, time,2)

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / trials) for i in range(trials)]

    # Set a seed for reproducibility
    np.random.seed(42)

    # Generate 10 random indices for selecting (trial, time) pairs
    random_trials = np.random.randint(0, trials, 10)
    random_times = np.random.randint(0, time, 10)

    # Plot each trial in the 3D PCA space
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    for trial in range(trials):
        if np.isin(trial, random_trials):
            lw = 2
        else:
            lw = 1

        # Plot each trial as a line in the 3D space
        ax.plot(
            pca_activities_reshaped[trial, :, 0],
            pca_activities_reshaped[trial, :, 1],
            color=colors[trial],
            label=f'Trial {trial+1}',
            linewidth = lw,
            alpha=0.7
        )

    ax.scatter( 0,0,color='purple',alpha=0.7)

    # Prepare an empty list to store the perturbed points
    perturbed_list = []
    time_steps = 500

    # Generate the perturbation vector and add it to the selected activities
    for i in range(10):
        trial_idx = random_trials[i]
        time_idx = random_times[i]
        
        # Select a point from the activities matrix
        selected_point = activities[trial_idx, time_idx, :]
        
        # Generate a perturbation vector from the range [-10, 10]
        perturbation = np.random.uniform(-1,1, units)
        
        # Add the perturbation to the selected point
        perturbed_point = selected_point # + perturbation

        activities1, h_ext = simulate_network(perturbed_point, N=n_hidden, time_steps=time_steps, c=c, epsilon=epsilon, theta0=theta_i[trial_idx], Jij = Jij, ct = 0, add_noise=add_noise)
        
        activities1 = np.vstack([selected_point, activities1])
        perturbed_list.append(activities1.reshape(1, activities1.shape[0], activities1.shape[1]))

    time_steps += 1
    activities = np.concatenate(perturbed_list, axis=0)
    reshaped_activities = activities.reshape(10 * time_steps, units)  # Shape: (trials * time, units)

    pca_activities = pca.fit_transform(reshaped_activities)


    pca_activities_reshaped = pca_activities.reshape(10, time_steps, 2)
    print(random_trials)

    for trial in range(10):
        trial_idx = random_trials[trial]
        print(trial, trial_idx, colors[trial_idx])

        random_array = np.random.normal(loc=0, scale=1, size=pca_activities_reshaped[trial, :, 0].shape)
        random_array = random_array / 50

        # Plot each trial as a line in the 3D space
        ax.plot(
            pca_activities_reshaped[trial, :, 0],
            pca_activities_reshaped[trial, :, 1],
            color='black',
            label=f'Trial {trial+1}',
            alpha=0.7
        )

        ax.scatter(
            pca_activities_reshaped[trial, :, 0]+ random_array,
            pca_activities_reshaped[trial, :, 1]+ random_array,
            color=colors[trial_idx],
            alpha=0.7
        )

        ax.scatter(
            pca_activities_reshaped[trial, 0, 0],
            pca_activities_reshaped[trial, 0, 1],
            color='black',
            alpha=0.7
        )

    # Set axis labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Trials Projected onto First 2 Principal Components')
    plt.show()










