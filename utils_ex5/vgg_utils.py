import numpy as np
import torchvision
from torchvision.models import VGG11_Weights
import matplotlib.pyplot as plt
import torch
import math
import pickle


def load_cnn_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    return model

def load_image(path):
    return torchvision.io.read_image(path).float()


def interactive_grating(frequency, radius, contrast, orientation):

    # Set parameters
    size = (256, 256)
    center_x = 125
    center_y = 125 


    grating = create_grating(frequency, orientation, 0, 128, contrast, location=None, radius=radius)
    plot_image(grating.clip(0, 1))

def create_orientation_gratings(frequency, radius, contrast, orientations, size=256, center=(125, 125)):
    # Set parameters
    center_x, center_y = center
    contrast = 1
    gratings = [create_grating(frequency, o, 0, size, contrast, (center_x, center_y), radius) for o in orientations]
    #network_outputs = [layers(g) for g in gratings]
    #outs = torch.stack(network_outputs)
    return torch.stack(gratings)

def create_contrast_gratings(frequency, radius, contrasts, orientation, size=256, center=(125, 125)):
    # Set parameters
    center_x, center_y = center
    gratings = [create_grating(frequency, orientation, 0, size, c, (center_x, center_y), radius) for c in contrasts]
    #network_outputs = [layers(g) for g in gratings]
    #outs = torch.stack(network_outputs)
    return torch.stack(gratings)

def create_radius_gratings(frequency, radii, contrast, orientation, size=256, center=(125, 125)):
    # Set parameters
    center_x, center_y = center
    gratings = [create_grating(frequency, orientation, 0, size, contrast, (center_x, center_y), r) for r in radii]
    #network_outputs = [layers(g) for g in gratings]
    #outs = torch.stack(network_outputs)
    return torch.stack(gratings)

def apply_circle(gratings, radius, size, center):

    if radius is not None:

        # Default center location is the middle of the image
        if center is None:
            center = (size // 2, size // 2)

        # Get x and y coordinates, centered on the location
        x = np.arange(size) - center[0]
        y = np.arange(size) - center[1]
        x, y = np.meshgrid(x, y)
        
        # Distance from the center
        distance_from_center = np.sqrt(x**2 + y**2)
        
        # Create a Gaussian mask for smooth falloff
        #mask = np.exp(-((distance_from_center**2) / (2 * (radius**2))))
        mask = np.where(distance_from_center <= radius, 1, 0)

        # Apply the mask to the grating
        gratings *= mask
        gratings = gratings + 0.5

    return gratings.float()

def create_superimposed_gratings(frequency, radius, contrast_pref, contrasts_orth, pref_orientation, orth_orientation, size=256, center=(125, 125)):
    if len(contrast_pref) > 1:
        raise ValueError(f"Error: length of contrast_pref should be 1, but got {len(contrast_pref)}.")

    # Set parameters
    center_x, center_y = center

    gratings_orth = [create_grating(frequency, orth_orientation, 0, size, c, (center_x, center_y), radius) for c in contrasts_orth]
    grating_pref = [create_grating(frequency, pref_orientation, 0, size, c, (center_x, center_y), radius) for c in contrast_pref]
    #create_contrast_gratings(frequency, radius, contrast_pref, pref_orientation)

    gratings_superimposed = [apply_circle(grating_pref[0] + grating_orth, radius, size, center) for grating_orth in gratings_orth]     

    return torch.stack(gratings_superimposed)

def create_grating(sf, ori, phase, imsize, contrast=1.0, location=None, radius=None):
    """
    :param sf: spatial frequency (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param wave: type of wave ('sqr' or 'sin')
    :param imsize: image size (integer)
    :param contrast: contrast of the grating (float between 0 and 1)
    :param location: (x, y) tuple for the center of the grating, defaults to the center of the image
    :param radius: radius of the grating in pixels, with smooth falloff beyond this radius
    :return: numpy array of shape (imsize, imsize)
    """
    # Default center location is the middle of the image
    if location is None:
        location = (imsize // 2, imsize // 2)

    # Get x and y coordinates, centered on the location
    x = np.arange(imsize) - location[0]
    y = np.arange(imsize) - location[1]
    x, y = np.meshgrid(x, y)

    # Apply orientation to create the gradient
    gradient = np.sin(ori * math.pi / 180) * x - np.cos(ori * math.pi / 180) * y

    # Plug gradient into the chosen wave function
    grating = np.sin((2 * math.pi * gradient) / sf + (phase * math.pi) / 180)

    # Apply contrast by scaling the grating
    #print(np.max(grating, axis = 1))
    grating *= contrast
    #print(np.max(grating, axis = 1))

    # If radius is specified, create a smooth circular mask
    if radius is not None:
        # Distance from the center
        distance_from_center = np.sqrt(x**2 + y**2)
        
        # Create a Gaussian mask for smooth falloff
        #mask = np.exp(-((distance_from_center**2) / (2 * (radius**2))))
        mask = np.where(distance_from_center <= radius, 1, 0)

        # Apply the mask to the grating
        grating *= mask
        grating = grating + 0.5

    return torch.tensor(grating).unsqueeze(0).repeat(3, 1, 1).float()

def create_bar_lengths(size, contrast, location, lengths, width):
    # Set parameters
    center_x, center_y = location
    gratings = [create_solid_bar(size, contrast, (center_x, center_y), l, width) for l in lengths]
    #network_outputs = [layers(g) for g in gratings]
    #outs = torch.stack(network_outputs)
    return torch.stack(gratings)


def create_solid_bar(imsize, contrast=1.0, location=None, length=None, width=None):
    """
    Create a solid bar pattern without frequency (single color).

    :param ori: bar orientation (in degrees, [0-360])
    :param imsize: image size (integer)
    :param contrast: contrast of the bar (float between 0 and 1)
    :param location: (x, y) tuple for the center of the bar, defaults to the center of the image
    :param length: length of the bar in pixels along its orientation
    :param width: width of the bar in pixels
    :return: numpy array of shape (imsize, imsize)
    """
    # Default center location is the middle of the image
    if location is None:
        location = (imsize // 2, imsize // 2)

    ori = 0

    # Get x and y coordinates, centered on the location
    x = np.arange(imsize) - location[0]
    y = np.arange(imsize) - location[1]
    x, y = np.meshgrid(x, y)

    # Apply orientation to create the bar
    # Rotate the coordinate system according to the orientation
    gradient = np.sin(ori * math.pi / 180) * x - np.cos(ori * math.pi / 180) * y

    # Create the mask for the bar length and width
    if length is not None and width is not None:
        # Create two gradients to control length and width of the bar
        perpendicular_gradient = np.cos(ori * math.pi / 180) * x + np.sin(ori * math.pi / 180) * y

        # Create mask for the bar
        bar_mask = (np.abs(gradient) <= width / 2) & (np.abs(perpendicular_gradient) <= length / 2)

        # Create a solid bar image based on the mask
        bar = np.zeros((imsize, imsize))
        bar[bar_mask] = contrast

    else:
        raise ValueError("Both length and width of the bar must be specified.")

    return torch.tensor(bar).unsqueeze(0).repeat(3, 1, 1).float()

def get_center(len_x, len_y):
    return math.ceil(len_x/2), math.ceil(len_y/2)

def get_pref_and_orth_angle(angle):
    pref_orientation = angle
    orth_orientation = angle + 90
    if orth_orientation > 180:
        orth_orientation = angle - 90
    return pref_orientation, orth_orientation

def get_network_layers(network):
    return list(network.children())[0]
    

def plot_64_neuron_grid(neurons):
        # Plot the grid of images (8x8 grid for 64 images)
    fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(16, 8))  # 8x8 grid

    # Loop through the grid and plot each image
    for i, ax in enumerate(axes.flat):
        # Unnormalize if necessary (if you used transforms.Normalize earlier)
        # image = images[i] * std + mean

        ax.set_title(f'Neuron {i}')
        
        ax.imshow(neurons.detach()[i])  # Display image
        ax.axis('off')  # Turn off axis for cleaner visualization

    plt.tight_layout()
    plt.show()

def plot_alexnet_rfs(convolutional_neural_net):
    conv_layers = []
    for layer in convolutional_neural_net.modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append(layer)
            if len(conv_layers) == 5:
                break

    # Loop through the first 5 convolutional layers and plot 25 random filters
    for idx, conv_layer in enumerate(conv_layers):
    
        filters = conv_layer.weight.data.clone()
        num_filters = filters.shape[0]
    
        # Randomly select 25 filters
        random_indices = np.random.choice(num_filters, 25, replace=False)
    
        # Set grid size to 5x5 for 25 filters
        grid_size = 5
    
        # Plot the filters in a 5x5 grid
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1, grid_size * 1))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, filter_idx in enumerate(random_indices):
            ax = axes[i]
        
            # Normalize the filter for visualization
            filter_image = filters[filter_idx].cpu().numpy()
            filter_image = (filter_image - filter_image.min()) / (filter_image.max() - filter_image.min())

            ax.imshow(filter_image[0], cmap='gray')
        
            ax.axis('off')
    
        fig.suptitle(f'25 Random Filters of Convolutional Layer {idx + 1}', fontsize=16)
        plt.show()

def plot_image_row(imgs, titles=None):
    n_images = len(imgs)
    # Plot the grid of images (8x8 grid for 64 images)
    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(16, 8))  # 8x8 grid

    if titles is None:
        titles = [''] * n_images

    # Loop through the grid and plot each image
    for i, ax in enumerate(axes.flat):
        # Unnormalize if necessary (if you used transforms.Normalize earlier)
        # image = images[i] * std + mean

        ax.set_title(titles[i])
        ax.imshow(imgs[i].permute(1, 2, 0).clip(0, 1))  # Display image
        ax.axis('off')  # Turn off axis for cleaner visualization

    plt.tight_layout()
    plt.show()

def plot_image(img, title=''):

    plt.title(title)
    plt.imshow(img.permute(1, 2, 0), extent=[-1, 1, -1, 1])
    plt.axis('off')  # Hide axis
    plt.show()



class RaoBallard1999Model:
    def __init__(self, dt=1, sigma2=1, sigma2_td=10):
        self.dt = dt
        self.inv_sigma2 = 1/sigma2 # 1 / sigma^2        
        self.inv_sigma2_td = 1/sigma2_td # 1 / sigma_td^2
        
        self.k1 = 0.3 # k_1: update rate
        self.k2 = 0.2 # k_2: learning rate
        
        self.lam = 0.02 # sparsity rate
        self.alpha = 1
        self.alphah = 0.05
        
        self.num_units_level0 = 256
        self.num_units_level1 = 32
        self.num_units_level2 = 128
        self.num_level1 = 3
        
        U = np.random.randn(self.num_units_level0, 
                            self.num_units_level1)
        Uh = np.random.randn(int(self.num_level1*self.num_units_level1),
                             self.num_units_level2)
        self.U = U.astype(np.float32) * np.sqrt(2/(self.num_units_level0+self.num_units_level1))
        self.Uh = Uh.astype(np.float32) * np.sqrt(2/(int(self.num_level1*self.num_units_level1)+self.num_units_level2)) 
                
        self.r = np.zeros((self.num_level1, self.num_units_level1))
        self.rh = np.zeros((self.num_units_level2))
    
    def initialize_states(self, inputs):
        self.r = inputs @ self.U 
        self.rh = self.Uh.T @ np.reshape(self.r, (int(self.num_level1*self.num_units_level1)))
    
    def calculate_total_error(self, error, errorh):
        recon_error = self.inv_sigma2*np.sum(error**2) + self.inv_sigma2_td*np.sum(errorh**2)
        sparsity_r = self.alpha*np.sum(self.r**2) + self.alphah*np.sum(self.rh**2)
        sparsity_U = self.lam*(np.sum(self.U**2) + np.sum(self.Uh**2))
        return recon_error + sparsity_r + sparsity_U
        
    def __call__(self, inputs, training=False):
        # inputs : (3, 256)
        r_reshaped = np.reshape(self.r, (int(self.num_level1*self.num_units_level1))) # (96)

        fx = self.r @ self.U.T        
        #fx = np.tanh(self.r @ self.U.T) # (3, 256)

        fxh = self.Uh @ self.rh # (96, )
        #fxh = np.tanh(self.Uh @ self.rh) # (96, )
        
        #dfx = 1 - fx**2 # (3, 256)
        #dfxh = 1 - fxh**2 # (96,)
        
        # Calculate errors
        error = inputs - fx # (3, 256)
        errorh = r_reshaped - fxh # (96, ) 
        errorh_reshaped = np.reshape(errorh, (self.num_level1, self.num_units_level1)) # (3, 32)

        #dfx_error = dfx * error # (3, 256)
        #dfxh_errorh = dfxh * errorh # (96, )
        
        g_r = self.alpha * self.r / (1 + self.r**2) # (3, 32)
        g_rh = self.alphah * self.rh / (1 + self.rh**2) # (64, )
        #g_r = self.alpha * self.r  # (3, 32)
        #g_rh = self.alphah * self.rh # (64, )
        
        # Update r and rh
        dr = self.inv_sigma2 * error @ self.U - self.inv_sigma2_td * errorh_reshaped - g_r
        drh = self.inv_sigma2_td * self.Uh.T @ errorh - g_rh
        
        """
        dr = self.inv_sigma2 * dfx_error @ self.U - self.inv_sigma2_td * errorh_reshaped - g_r
        drh = self.inv_sigma2_td * self.Uh.T @ dfxh_errorh - g_rh
        """
        
        dr = self.k1 * dr
        drh = self.k1 * drh
        
        # Updates                
        self.r += dr
        self.rh += drh
        
        if training:  
            """
            dU = self.inv_sigma2 * dfx_error.T @ self.r - 3*self.lam * self.U
            dUh = self.inv_sigma2_td * np.outer(dfxh_errorh, self.rh) - self.lam * self.Uh
            """
            dU = self.inv_sigma2 * error.T @ self.r - 3*self.lam * self.U
            dUh = self.inv_sigma2_td * np.outer(errorh, self.rh) - self.lam * self.Uh
            
            self.U += self.k2 * dU
            self.Uh += self.k2 * dUh
            
        return error, errorh, dr, drh


def load_rao_ballard():
    with open('utils_ex5/rao_ballard_trained.pkl', 'rb') as f:
        obj = pickle.load(f)

    model = obj
    # Plot Receptive fields of level 1
    fig = plt.figure(figsize=(8, 4))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(np.reshape(model.U[:, i], (16, 16)), cmap="gray")
        plt.axis("off")

    fig.suptitle("Layer 1 receptive fields", fontsize=20)

    # Plot Receptive fields of level 2
    zero_padding = np.zeros((80, 32))
    U0 = np.concatenate((model.U, zero_padding, zero_padding))
    U1 = np.concatenate((zero_padding, model.U, zero_padding))
    U2 = np.concatenate((zero_padding, zero_padding, model.U))
    U_ = np.concatenate((U0, U1, U2), axis = 1)
    Uh_ = U_ @ model.Uh  

    fig = plt.figure(figsize=(8, 5))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(np.reshape(Uh_[:, i], (16, 26), order='F'), cmap="gray")
        plt.axis("off")

    fig.suptitle("Layer 2 receptive fields", fontsize=20)
    plt.subplots_adjust(top=0.9)
    plt.show()
    return obj
