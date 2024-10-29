import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, widgets

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, VBox, Output, Layout
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, widgets

def iplot_data_2classes():

    # Output widget for displaying the plot
    output_plot = Output()

    # Update the plot (triggered by the button)
    def update_plot(corr_coeff=0, ortho_order="input / choice", orthogonalize = True):
        # Create a 2D covariance matrix based on the correlation coefficient
        cov_matrix = [[1, corr_coeff], [corr_coeff, 1]]
        np.random.seed(4)
        
        # Generate random data for two classes
        data1 = np.random.multivariate_normal([.5, .5], cov_matrix, size=500)
        data2 = np.random.multivariate_normal([-.5, -.5], cov_matrix, size=500)

        # Set fixed rightmost points for each class
        fixed_point1 = np.array([4, 2])
        fixed_point2 = np.array([-4, -2])

        # Calculate distances from the fixed points
        dist1 = np.linalg.norm(data1 - fixed_point1, axis=1)
        dist2 = np.linalg.norm(data2 - fixed_point2, axis=1)

        # Normalize distances to range [0, 1] for color mapping
        dist1_normalized = (dist1 - dist1.min()) / (dist1.max() - dist1.min())
        dist2_normalized = (dist2 - dist2.min()) / (dist2.max() - dist2.min())

        # Linear regression to normalize beta
        reg = LinearRegression().fit(data1, dist1_normalized)
        betas = reg.coef_
        betas_input_normalized_l2 = betas / np.linalg.norm(betas, ord=2)

        # Create a color gradient based on the normalized distances
        colors1 = cm.Blues(dist1_normalized)
        colors2 = cm.Reds(dist2_normalized)

        # Clear the previous plot output
        with output_plot:
            output_plot.clear_output(wait=True)

            # Create a figure with 4 subplots
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))

            # Plot the data with color gradients in the first subplot
            axs[0].scatter(data1[:, 0], data1[:, 1], alpha=0.5, color=colors1, label='Choice 1')
            axs[0].scatter(data2[:, 0], data2[:, 1], alpha=0.5, color=colors2, label='Choice 2')
            
            # Stack data and labels for further analysis
            data = np.vstack((data1, data2))
            labels = np.hstack((np.ones(data1.shape[0]), np.full(data2.shape[0], 2)))

            # Part 1: Run PCA on the stacked data
            pca = PCA(n_components=2)
            pca.fit(data)
            pcs = pca.components_

            norm2 = np.linalg.norm(pcs[0, :], ord=2)
            pcs1_normalized_l2 = pcs[0, :] / norm2

            # Part 2: Run linear regression on the stacked data
            reg = LinearRegression().fit(data, labels)
            betas = reg.coef_
            betas_choice_normalized_l2 = betas / np.linalg.norm(betas, ord=2)

            # Apply orthogonalization based on the user's selection
            if orthogonalize:
                if ortho_order == "input / choice":
                    projection = np.dot(betas_choice_normalized_l2, betas_input_normalized_l2) / np.dot(betas_input_normalized_l2, betas_input_normalized_l2) * betas_input_normalized_l2
                    v2_orthogonal = betas_choice_normalized_l2 - projection
                    betas_choice_normalized_l2 = v2_orthogonal / np.linalg.norm(v2_orthogonal, ord=2)
                elif ortho_order == "choice / input":
                    projection = np.dot(betas_input_normalized_l2, betas_choice_normalized_l2) / np.dot(betas_choice_normalized_l2, betas_choice_normalized_l2) * betas_choice_normalized_l2
                    v2_orthogonal = betas_input_normalized_l2 - projection
                    betas_input_normalized_l2 = v2_orthogonal / np.linalg.norm(v2_orthogonal, ord=2)

            # Plot the basis vectors in the first subplot
            axs[0].plot([0, betas_choice_normalized_l2[0]], [0, betas_choice_normalized_l2[1]], color='black', linewidth=3, label='TDR-choice')   
            axs[0].plot([0, betas_input_normalized_l2[0]], [0, betas_input_normalized_l2[1]], color='red', linewidth=3, label='TDR-input 2 strength')
            axs[0].plot([0, pcs1_normalized_l2[0]], [0, pcs1_normalized_l2[1]], color='green', linewidth=3, label='PCA')
            
            axs[0].set_title(f'Data with Correlation Coefficient: {corr_coeff:.2f}')
            axs[0].set_xlabel('X1')
            axs[0].set_ylabel('X2')
            axs[0].set_xlim([-5, 5])
            axs[0].set_ylim([-5, 5])
            axs[0].grid(True)
            axs[0].legend()

            # Compute projections of each class onto PCA and TDR axes
            proj_pca1 = np.dot(data1, pcs1_normalized_l2)
            proj_pca2 = np.dot(data2, pcs1_normalized_l2)
            proj_tdr1 = np.dot(data1, betas_choice_normalized_l2)
            proj_tdr2 = np.dot(data2, betas_choice_normalized_l2)

            # Plot histogram of projections along PCA axis for each class in the second subplot
            axs[1].hist(proj_pca1, bins=30, color='blue', alpha=0.5, label='Choice 1')
            axs[1].hist(proj_pca2, bins=30, color='red', alpha=0.5, label='Choice 2')
            axs[1].set_title('Projections onto PCA axis')
            axs[1].set_xlabel('Projection value')
            axs[1].set_ylabel('Counts')
            axs[1].legend()

            # Plot histogram of projections along TDR axis for each class in the third subplot
            axs[2].hist(proj_tdr1, bins=30, color='blue', alpha=0.5, label='Choice 1')
            axs[2].hist(proj_tdr2, bins=30, color='red', alpha=0.5, label='Choice 2')
            axs[2].set_title('Projections onto TDR-choice axis')
            axs[2].set_xlabel('Projection value')
            axs[2].set_ylabel('Counts')
            axs[2].legend()

            proj_input_tdr1 = np.dot(data1, betas_input_normalized_l2)

            # Plot scatter of projections against normalized distance
            axs[3].scatter(dist1_normalized, proj_input_tdr1, color='red', alpha=0.5)
            axs[3].set_title('Projections onto TDR-choice axis')
            axs[3].set_xlabel('Input 2 strength')
            axs[3].set_ylabel('Projection')

            plt.tight_layout()
            plt.show()

    # Create sliders for synaptic weight, frequency, and length
    style = {'description_width': 'initial'}
    corr_coeff_slider = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description='Corr Coeff',continuous_update=False)
    orthogonalize_checkbox = widgets.Checkbox(value=False, description='Orthogonalize TDR')
    ortho_order_dropdown = widgets.Dropdown(options=["choice / input", "input / choice"], value="choice / input", description="Ortho Order TDR")

    # Create the button to trigger the plot update
    plot_button = Button(description="Update Plot", button_style='success')

    # Callback function to update plot when the button is clicked
    def on_button_click(b):
        update_plot(corr_coeff=corr_coeff_slider.value, ortho_order=ortho_order_dropdown.value, orthogonalize=orthogonalize_checkbox.value)

    # Link button to callback function
    plot_button.on_click(on_button_click)

    # Create a vertical box to hold sliders, the button, and the output plot
    ui = VBox([corr_coeff_slider, orthogonalize_checkbox, ortho_order_dropdown, plot_button, output_plot])

    display(ui)

    update_plot(corr_coeff=corr_coeff_slider.value, ortho_order=ortho_order_dropdown.value, orthogonalize=orthogonalize_checkbox.value)





def iplot_data():
    interact(visualize_correlated_data_first, 
         corr_coeff=widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description='Corr Coeff',continuous_update=False))




# Define your function to visualize correlated data with gradient colors based on distance from fixed points
def visualize_correlated_data_first(corr_coeff=0):
    # Create a 2D covariance matrix based on the correlation coefficient
    cov_matrix = [[1, corr_coeff], [corr_coeff, 1]]
    
    # Generate random data for two classes
    data1 = np.random.multivariate_normal([.5, .5], cov_matrix, size=500)
    data2 = np.random.multivariate_normal([-.5, -.5], cov_matrix, size=500)

    # Set fixed rightmost points for each class
    fixed_point1 = np.array([4, 2])
    fixed_point2 = np.array([-4, -2])

    # Calculate distances from the fixed points
    dist1 = np.linalg.norm(data1 - fixed_point1, axis=1)
    dist2 = np.linalg.norm(data2 - fixed_point2, axis=1)

    # Normalize distances to range [0, 1] for color mapping
    dist1_normalized = (dist1 - dist1.min()) / (dist1.max() - dist1.min())
    dist2_normalized = (dist2 - dist2.min()) / (dist2.max() - dist2.min())

    # Linear regression to normalize beta
    reg = LinearRegression().fit(data1, dist1_normalized)
    betas = reg.coef_
    betas_input_normalized_l2 = betas / np.linalg.norm(betas, ord=2)

    # Create a color gradient based on the normalized distances
    colors1 = cm.Blues(dist1_normalized)
    colors2 = cm.Reds(dist2_normalized)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    # Plot the data with color gradients in the first subplot
    axs.scatter(data1[:, 0], data1[:, 1], alpha=0.5, color=colors1, label='Choice 1')
    axs.scatter(data2[:, 0], data2[:, 1], alpha=0.5, color=colors2, label='Choice 2')
    axs.set_xlabel('X1')
    axs.set_ylabel('X2')
    axs.set_xlim([-5, 5])
    axs.set_ylim([-5, 5])
    axs.grid(True)
    axs.legend()

    plt.tight_layout()
    plt.show()



#############


# Define your function to visualize correlated data with gradient colors based on distance from fixed points
def visualize_correlated_data(corr_coeff=0, orthogonalize=False, ortho_order="choice / input"):
    # Create a 2D covariance matrix based on the correlation coefficient
    cov_matrix = [[1, corr_coeff], [corr_coeff, 1]]
    
    # Generate random data for two classes
    data1 = np.random.multivariate_normal([.5, .5], cov_matrix, size=500)
    data2 = np.random.multivariate_normal([-.5, -.5], cov_matrix, size=500)

    # Set fixed rightmost points for each class
    fixed_point1 = np.array([4, 2])
    fixed_point2 = np.array([-4, -2])

    # Calculate distances from the fixed points
    dist1 = np.linalg.norm(data1 - fixed_point1, axis=1)
    dist2 = np.linalg.norm(data2 - fixed_point2, axis=1)

    # Normalize distances to range [0, 1] for color mapping
    dist1_normalized = (dist1 - dist1.min()) / (dist1.max() - dist1.min())
    dist2_normalized = (dist2 - dist2.min()) / (dist2.max() - dist2.min())

    # Linear regression to normalize beta
    reg = LinearRegression().fit(data1, dist1_normalized)
    betas = reg.coef_
    betas_input_normalized_l2 = betas / np.linalg.norm(betas, ord=2)

    # Create a color gradient based on the normalized distances
    colors1 = cm.Blues(dist1_normalized)
    colors2 = cm.Reds(dist2_normalized)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the data with color gradients in the first subplot
    axs[0].scatter(data1[:, 0], data1[:, 1], alpha=0.5, color=colors1, label='Choice 1')
    axs[0].scatter(data2[:, 0], data2[:, 1], alpha=0.5, color=colors2, label='Choice 2')
    
    # Stack data and labels for further analysis
    data = np.vstack((data1, data2))
    labels = np.hstack((np.ones(data1.shape[0]), np.full(data2.shape[0], 2)))

    # Part 1: Run PCA on the stacked data
    pca = PCA(n_components=2)
    pca.fit(data)
    pcs = pca.components_

    norm2 = np.linalg.norm(pcs[0, :], ord=2)
    pcs1_normalized_l2 = pcs[0, :] / norm2

    # Part 2: Run linear regression on the stacked data
    reg = LinearRegression().fit(data, labels)
    betas = reg.coef_
    betas_choice_normalized_l2 = betas / np.linalg.norm(betas, ord=2)

    # Apply orthogonalization based on the user's selection
    if orthogonalize:
        if ortho_order == "input / choice":
            projection = np.dot(betas_choice_normalized_l2, betas_input_normalized_l2) / np.dot(betas_input_normalized_l2, betas_input_normalized_l2) * betas_input_normalized_l2
            v2_orthogonal = betas_choice_normalized_l2 - projection
            betas_choice_normalized_l2 = v2_orthogonal / np.linalg.norm(v2_orthogonal, ord=2)
        elif ortho_order == "choice / input":
            projection = np.dot(betas_input_normalized_l2, betas_choice_normalized_l2) / np.dot(betas_choice_normalized_l2, betas_choice_normalized_l2) * betas_choice_normalized_l2
            v2_orthogonal = betas_input_normalized_l2 - projection
            betas_input_normalized_l2 = v2_orthogonal / np.linalg.norm(v2_orthogonal, ord=2)

    # Plot the basis vectors in the first subplot
    axs[0].plot([0, betas_choice_normalized_l2[0]], [0, betas_choice_normalized_l2[1]], color='black', linewidth=3, label='TDR-choice')   
    axs[0].plot([0, betas_input_normalized_l2[0]], [0, betas_input_normalized_l2[1]], color='red', linewidth=3, label='TDR-input 2 strength')
    axs[0].plot([0, pcs1_normalized_l2[0]], [0, pcs1_normalized_l2[1]], color='green', linewidth=3, label='PCA')
    
    axs[0].set_title(f'Data with Correlation Coefficient: {corr_coeff:.2f}')
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')
    axs[0].set_xlim([-5, 5])
    axs[0].set_ylim([-5, 5])
    axs[0].grid(True)
    axs[0].legend()

    # Compute projections of each class onto PCA and TDR axes
    proj_pca1 = np.dot(data1, pcs1_normalized_l2)
    proj_pca2 = np.dot(data2, pcs1_normalized_l2)
    proj_tdr1 = np.dot(data1, betas_choice_normalized_l2)
    proj_tdr2 = np.dot(data2, betas_choice_normalized_l2)

    # Plot histogram of projections along PCA axis for each class in the second subplot
    axs[1].hist(proj_pca1, bins=30, color='blue', alpha=0.5, label='Choice 1')
    axs[1].hist(proj_pca2, bins=30, color='red', alpha=0.5, label='Choice 2')
    axs[1].set_title('Projections onto PCA axis')
    axs[1].set_xlabel('Projection value')
    axs[1].set_ylabel('Counts')
    axs[1].legend()

    # Plot histogram of projections along TDR axis for each class in the third subplot
    axs[2].hist(proj_tdr1, bins=30, color='blue', alpha=0.5, label='Choice 1')
    axs[2].hist(proj_tdr2, bins=30, color='red', alpha=0.5, label='Choice 2')
    axs[2].set_title('Projections onto TDR-choice axis')
    axs[2].set_xlabel('Projection value')
    axs[2].set_ylabel('Counts')
    axs[2].legend()

    proj_input_tdr1 = np.dot(data1, betas_input_normalized_l2)

    # Plot scatter of projections against normalized distance
    axs[3].scatter(dist1_normalized, proj_input_tdr1, color='red', alpha=0.5)
    axs[3].set_title('Projections onto TDR-choice axis')
    axs[3].set_xlabel('Input 2 strength')
    axs[3].set_ylabel('Projection')

    plt.tight_layout()
    plt.show()




# Define your function to visualize correlated data with gradient colors based on distance from fixed points
def visualize_correlated_data_2classes(corr_coeff=0):
    # Create a 2D covariance matrix based on the correlation coefficient
    cov_matrix = [[1, corr_coeff], [corr_coeff, 1]]
    
    # Generate random data for two classes
    data1 = np.random.multivariate_normal([.5, .5], cov_matrix, size=500)
    data2 = np.random.multivariate_normal([-.5, -.5], cov_matrix, size=500)

    # Set fixed rightmost points for each class
    fixed_point1 = np.array([4, 2])
    fixed_point2 = np.array([-4, -2])

    # Calculate distances from the fixed points
    dist1 = np.linalg.norm(data1 - fixed_point1, axis=1)
    dist2 = np.linalg.norm(data2 - fixed_point2, axis=1)

    # Normalize distances to range [0, 1] for color mapping
    dist1_normalized = (dist1 - dist1.min()) / (dist1.max() - dist1.min())
    dist2_normalized = (dist2 - dist2.min()) / (dist2.max() - dist2.min())

    reg = LinearRegression().fit(data1, dist1_normalized)
    betas = reg.coef_
    betas_input_normalized_l2 = betas / np.linalg.norm(betas, ord=2)

    # Create a color gradient based on the normalized distances
    colors1 = cm.Blues(dist1_normalized)
    colors2 = cm.Reds(dist2_normalized)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the data with color gradients in the first subplot
    axs[0].scatter(data1[:, 0], data1[:, 1], alpha=0.5, color=colors1, label='Choice 1')
    axs[0].scatter(data2[:, 0], data2[:, 1], alpha=0.5, color=colors2, label='Choice 2')
    
    # Stack data and labels for further analysis
    data = np.vstack((data1, data2))
    labels = np.hstack((np.ones(data1.shape[0]), np.full(data2.shape[0], 2)))

    # Part 1: Run PCA on the stacked data
    pca = PCA(n_components=2)
    pca.fit(data)
    pcs = pca.components_

    norm2 = np.linalg.norm(pcs[0, :], ord=2)
    pcs1_normalized_l2 = pcs[0, :] / norm2

    # Part 2: Run linear regression on the stacked data
    reg = LinearRegression().fit(data, labels)
    betas = reg.coef_
    betas_choice_normalized_l2 = betas / np.linalg.norm(betas, ord=2)
    do_ortho = False
    if do_ortho:
        projection = np.dot(betas_input_normalized_l2, betas_choice_normalized_l2) / np.dot(betas_choice_normalized_l2, betas_choice_normalized_l2) * betas_choice_normalized_l2
        v2_orthogonal = betas_input_normalized_l2 - projection
        betas_input_normalized_l2 = v2_orthogonal / np.linalg.norm(v2_orthogonal, ord=2)

    # Plot the basis vectors in the first subplot
    axs[0].plot([0, betas_choice_normalized_l2[0]], [0, betas_choice_normalized_l2[1]], color='black', linewidth=3, label='TDR-choice')
    axs[0].plot([0, pcs1_normalized_l2[0]], [0, pcs1_normalized_l2[1]], color='green', linewidth=3, label='PCA')
    axs[0].plot([0, betas_input_normalized_l2[0]], [0, betas_input_normalized_l2[1]], color='red', linewidth=3, label='TDR-input 2 strength')
    
    axs[0].set_title(f'Data with Correlation Coefficient: {corr_coeff:.2f}')
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')
    axs[0].set_xlim([-5, 5])
    axs[0].set_ylim([-5, 5])
    axs[0].grid(True)
    axs[0].legend()

    # Compute projections of each class onto PCA and TDR axes
    proj_pca1 = np.dot(data1, pcs1_normalized_l2)
    proj_pca2 = np.dot(data2, pcs1_normalized_l2)
    proj_tdr1 = np.dot(data1, betas_choice_normalized_l2)
    proj_tdr2 = np.dot(data2, betas_choice_normalized_l2)

    # Plot histogram of projections along PCA axis for each class in the second subplot
    axs[1].hist(proj_pca1, bins=30, color='blue', alpha=0.5, label='Choice 1')
    axs[1].hist(proj_pca2, bins=30, color='red', alpha=0.5, label='Choice 2')
    axs[1].set_title('Projections onto PCA axis')
    axs[1].set_xlabel('Projection value')
    axs[1].set_ylabel('Counts')
    axs[1].legend()

    # Plot histogram of projections along TDR axis for each class in the third subplot
    axs[2].hist(proj_tdr1, bins=30, color='blue', alpha=0.5, label='Choice 1')
    axs[2].hist(proj_tdr2, bins=30, color='red', alpha=0.5, label='Choice 2')
    axs[2].set_title('Projections onto TDR-choice axis')
    axs[2].set_xlabel('Projection value')
    axs[2].set_ylabel('Counts')
    axs[2].legend()

    proj_input_tdr1 = np.dot(data1, betas_input_normalized_l2)

    # Plot histogram of projections along TDR axis for each class in the third subplot
    axs[3].scatter(dist1_normalized, proj_input_tdr1, color='red', alpha=0.5)
    axs[3].set_title('Projections onto TDR-choice axis')
    axs[3].set_xlabel('Input 2 strength')
    axs[3].set_ylabel('Projection')

    plt.tight_layout()
    plt.show()



