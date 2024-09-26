from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm




def plot_2d_flowfield(f, space=np.linspace(-10, 10, 10), data=None, title=''):
        # Define a grid for the flow field
    if data is not None:
        space = np.linspace(-abs(data).max(), abs(data).max(), 10)

    x = space
    y = space
    X1, X2 = np.meshgrid(x, y)

    # Compute the flow field (vector field) at each point on the grid
    U = np.zeros_like(X1)  # To store dx/dt (U) values
    V = np.zeros_like(X2)  # To store dy/dt (V) values

    # For each point on the grid, calculate the direction of the flow
    for i in range(len(x)):
        for j in range(len(y)):
            X_vec = np.array([X1[i, j], X2[i, j]])  # Position vector [x, y]
            dX = f(X_vec)                # Compute the derivatives [dx/dt, dy/dt]
            U[i, j] = dX[0]                         # dx/dt
            V[i, j] = dX[1]     
    # Create a quiver plot to visualize the flow field
 
    plt.figure(figsize=(8, 8))

    if data is not None:
        plt.plot(data[..., 0], data[..., 1])
    plt.quiver(X1, X2, U, V, color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    ax = plt.gca()

    return ax
def interactive_eigval_plot(real1, imag1, real2, imag2, b):


    f, axes = plt.subplots(1, 2, figsize=(8,4))

    a1, b1 = real1, imag1
    a2, b2 = real2, imag2

    A = np.array([[a1, -b1],
                [b2, a2]])
    

    t = np.linspace(0,np.pi*2,100)
    axes[0].set_title('Eigenvalues of A')
    axes[0].plot(np.cos(t), np.sin(t), linewidth=1)

    axes[0].scatter(real1, imag1, label='eigenvalue 1')
    axes[0].scatter(real2, imag2, label='eigenvalue 2')
    axes[0].set_xlabel('real part')
    axes[0].set_ylabel('imaginary part')
    axes[0].legend()
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)

    # Define the system of ODEs
    def linear_system(X, t=0):
        return A @ X + b


    space=np.linspace(-10, 10, 10)
    x = space
    y = space
    X1, X2 = np.meshgrid(x, y)

    # Compute the flow field (vector field) at each point on the grid
    U = np.zeros_like(X1)  # To store dx/dt (U) values
    V = np.zeros_like(X2)  # To store dy/dt (V) values

    # For each point on the grid, calculate the direction of the flow
    for i in range(len(x)):
        for j in range(len(y)):
            X_vec = np.array([X1[i, j], X2[i, j]])  # Position vector [x, y]
            dX = linear_system(X_vec)                # Compute the derivatives [dx/dt, dy/dt]
            U[i, j] = dX[0]                         # dx/dt
            V[i, j] = dX[1]     
    # Create a quiver plot to visualize the flow field

    
    axes[1].quiver(X1, X2, U, V, color='black')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title("Dynamics")

    

def simulate_noisy_system(noise_level, system=None):

    
    if system is None:
            # Define the system matrix A
        A = np.array([[0.01, 0],
                    [0, -0.01]])

        # Define the system of ODEs
        def system(X):
            #linear system with noise
            return (X @ A) + np.random.randn(4, 2) * noise_level

    initial_conditions = np.array([[1.1, 3], [0.001, 1], [0.5, 2], [0.5, 0.2]])

    #simulate the system
    z = np.zeros((100, 4, 2))
    z[0] = initial_conditions
    for i in range(1, 100):
        z[i] = z[i-1] + system(z[i-1])

    return z

def simulate_noisy_3d_system(noise_level):

        # Define the system matrix A
    A = np.array([[1, -1, 0],[-1, 2, -1],[0, -1, 1]]) * 0.01

    # Define the system of ODEs
    def linear_system(X):
        #linear system with noise
        return (X @ A) + np.random.randn(50, 3) * noise_level

        # Find the eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Select the eigenvectors corresponding to non-zero eigenvalues
    nonzero_eigenvectors = eigenvectors[:, np.abs(eigenvalues) > 1e-6]

    # Number of initial conditions
    n_points = 50

    # Randomly generate coefficients (alphas and betas) for the linear combination of the two eigenvectors
    alphas = np.random.uniform(-5, 5, n_points)
    betas = np.random.uniform(-5, 5, n_points)

    # Generate initial conditions as a linear combination of the eigenvectors
    initial_conditions = np.array([alpha * nonzero_eigenvectors[:, 0] + beta * nonzero_eigenvectors[:, 1] 
                    for alpha, beta in zip(alphas, betas)])

    #simulate the system
    z = np.zeros((100, 50, 3))
    z[0] = initial_conditions
    for i in range(1, 100):
        z[i] = z[i-1] + linear_system(z[i-1])

    return z



def fit_dynamics_matrix_continuous(data, dt):
    """
    Fit a 2x2 dynamics matrix A to the time x trials x 2 data in continuous-time.
    
    Parameters:
    - data: numpy array of shape (time, trials, 2)
    - dt: time step between consecutive time points
    
    Returns:
    - A: fitted continuous dynamics matrix (2x2)
    """
    
    # Reshape the data: combine time and trials dimensions into one
    X_t = data[:-1].reshape(-1, 2)  # States at time t
    X_t1 = data[1:].reshape(-1, 2)  # States at time t+1
    
    # Solve the least-squares problem to fit e^(A * dt): X_t1 ≈ expm(A * dt) @ X_t
    # Use lstsq to fit the matrix exponential exp(A * dt)
    expA_dt, _, _, _ = np.linalg.lstsq(X_t, X_t1, rcond=None)
    
    # Now recover A by taking the matrix logarithm
    A_dt = logm(expA_dt)  # logm gives A * dt
    
    # Recover A by dividing by dt
    A = A_dt / 0.1
    return A


def pca(noisy_3d_data):

    data_flat = noisy_3d_data.reshape(-1, 3)
    data_flat = data_flat - data_flat.mean(0)[None, :]

    S, V, D = np.linalg.svd(data_flat)

    variance_explained = (V / V.sum()) 
    components = D
    return variance_explained, components


def compute_jacobian(system, point, h=1e-5):
    n = len(point)  # Dimension of the system
    J = np.zeros((n, n))  # Initialize the Jacobian matrix
    
    # Loop over each variable to compute partial derivatives
    for i in range(n):
        perturbed_point = np.copy(point)
        perturbed_point[i] += h  # Perturb variable i by a small amount h
        
        # Compute the system value at the perturbed point and original point
        f_perturbed = system(perturbed_point)
        f_original = system(point)
        
        # Approximate the partial derivative
        J[:, i] = (f_perturbed - f_original) / h  # Central difference approximation
    
    return J