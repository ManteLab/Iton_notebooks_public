import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import clear_output
from scipy.integrate import solve_ivp
from scipy.linalg import logm



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


def plot_2d_flowfield(f, space=np.linspace(-10, 10, 10), data=None, title="", ax=None):
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
 
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if data is not None:
        ax.plot(data[..., 0], data[..., 1])
    ax.quiver(X1, X2, U, V, color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    return ax
def interactive_eigval_plot(a11, a12, a21, a22, b):
    """Plot the eigenvalues and flow field of a real 2D dynamics matrix."""


    f, axes = plt.subplots(1, 2, figsize=(8,4))

    A = np.array([[a11, a12], [a21, a22]])
    eigenvalues = np.linalg.eigvals(A)
    

    axes[0].set_title("Eigenvalues of A")
    axes[0].axvline(0, color="grey", linewidth=1)
    axes[0].axhline(0, color="grey", linewidth=1)

    axes[0].scatter(eigenvalues[0].real, eigenvalues[0].imag, label="eigenvalue 1")
    axes[0].scatter(eigenvalues[1].real, eigenvalues[1].imag, label="eigenvalue 2")
    axes[0].set_xlabel("real part")
    axes[0].set_ylabel("imaginary part")
    axes[0].legend()
    eigenvalue_limit = max(2, np.abs(eigenvalues).max() * 1.2)
    axes[0].set_xlim(-eigenvalue_limit, eigenvalue_limit)
    axes[0].set_ylim(-eigenvalue_limit, eigenvalue_limit)
    axes[0].set_aspect("equal")

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

    
    axes[1].quiver(X1, X2, U, V, color="black")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Dynamics")

    plt.show()

    

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
    eigenvectors = np.real_if_close(eigenvectors)

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
    z[0] = initial_conditions + np.random.randn(*initial_conditions.shape) * noise_level
    for i in range(1, 100):
        z[i] = z[i-1] + linear_system(z[i-1])

    return z



def fit_dynamics_matrix_continuous(data, dt):
    """
    Fit a continuous-time matrix acting on row-vector observations.
    
    Parameters:
    - data: numpy array of shape (time, trials, 2)
    - dt: time step between consecutive time points
    
    Returns:
    - A: fitted continuous dynamics matrix (2x2), with dx/dt = x @ A.
    """
    
    # Reshape the data: combine time and trials dimensions into one
    X_t = data[:-1].reshape(-1, 2)  # States at time t
    X_t1 = data[1:].reshape(-1, 2)  # States at time t+1
    
    # Solve the least-squares problem X_t1 ≈ X_t @ expm(A * dt).
    # Use lstsq to fit the matrix exponential exp(A * dt)
    expA_dt, _, _, _ = np.linalg.lstsq(X_t, X_t1, rcond=None)
    
    # Now recover A by taking the matrix logarithm
    A_dt = logm(expA_dt)  # logm gives A * dt
    
    # Recover A by dividing by dt
    A = A_dt / dt
    return A


def pca(noisy_3d_data):
    """Return explained-variance fractions and PC directions as columns."""

    data_flat = noisy_3d_data.reshape(-1, 3)
    data_flat = data_flat - data_flat.mean(0)[None, :]

    _, singular_values, right_vectors = np.linalg.svd(data_flat, full_matrices=False)

    variance_explained = singular_values**2 / np.sum(singular_values**2)
    components = right_vectors.T
    return variance_explained, components


def compute_jacobian(system, point, h=1e-5):
    """Approximate a vector field's Jacobian using central differences."""
    point = np.asarray(point, dtype=float)
    n = len(point)  # Dimension of the system
    J = np.zeros((n, n))  # Initialize the Jacobian matrix
    
    # Loop over each variable to compute partial derivatives
    for i in range(n):
        forward_point = point.copy()
        backward_point = point.copy()
        forward_point[i] += h
        backward_point[i] -= h
        
        # Evaluate on both sides of the point for a central difference.
        f_forward = system(forward_point)
        f_backward = system(backward_point)
        
        # Approximate the partial derivative
        J[:, i] = (f_forward - f_backward) / (2 * h)
    
    return J


class Noisy3DExplorer:
    """Keep a rotatable population plot and its PCA views on the same sample."""

    def __init__(self, noise_level=0.1):
        self.noise_slider = widgets.FloatSlider(
            value=noise_level, min=0, max=1, step=0.05,
            description="Noise level", continuous_update=False,
        )
        self.resample_button = widgets.Button(description="New sample")
        self.plot_output = widgets.Output()
        self.widget = widgets.VBox([
            widgets.HBox([self.noise_slider, self.resample_button]),
            self.plot_output,
        ])
        self._views = {}
        self.noise_slider.observe(self._update, names="value")
        self.resample_button.on_click(self._update)
        self._update()

    def _update(self, change=None):
        self.data = simulate_noisy_3d_system(self.noise_slider.value)
        self.variance_explained, self.components = pca(self.data)
        self.projection = self.data @ self.components[:, :2]
        self.figure = go.Figure([
            go.Scatter3d(
                x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                mode="lines", line={"width": 2}, opacity=0.55,
                name=f"Trajectory {trial + 1}", showlegend=False,
            )
            for trial, trajectory in enumerate(self.data.transpose(1, 0, 2))
        ])
        self.figure.update_layout(
            title=f"Noisy three-dimensional trajectories (noise = {self.noise_slider.value:.2f})",
            height=550, margin={"l": 0, "r": 0, "b": 0, "t": 45},
            scene={
                "xaxis_title": "Neuron 1", "yaxis_title": "Neuron 2",
                "zaxis_title": "Neuron 3", "dragmode": "orbit", "aspectmode": "data",
            },
        )
        with self.plot_output:
            clear_output(wait=True)
            self.figure.show(
                renderer="colab" if is_colab else "plotly_mimetype",
                config={"scrollZoom": True, "displaylogo": False},
            )
        for _, draw in self._views.values():
            draw()

    def variance_view(self):
        """Return the explained-variance view, refreshed on every new sample."""
        if "variance" not in self._views:
            output = widgets.Output()

            def draw():
                with output:
                    clear_output(wait=True)
                    plt.figure()
                    plt.scatter(np.arange(1, 4), self.variance_explained)
                    plt.xticks([1, 2, 3])
                    plt.xlabel("Principal component")
                    plt.ylabel("Fraction of variance explained")
                    plt.title("Explained variance in the current three-dimensional sample")
                    plt.ylim(0, 1)
                    plt.show()

            self._views["variance"] = output, draw
            draw()
        return self._views["variance"][0]

    def projection_view(self):
        """Return projected trajectories, refreshed on every new sample."""
        if "projection" not in self._views:
            output = widgets.Output()

            def draw():
                with output:
                    clear_output(wait=True)
                    plt.figure(figsize=(6, 6))
                    plt.plot(self.projection[..., 0], self.projection[..., 1], alpha=0.35)
                    plt.scatter(*self.projection[0].T, marker="o", color="black", label="Initial states")
                    plt.scatter(*self.projection[-1].T, marker="x", color="black", label="Final states")
                    plt.scatter(0, 0, marker="+", s=120, color="red", label="Origin")
                    plt.xlabel("PC 1")
                    plt.ylabel("PC 2")
                    plt.title("Current trajectories projected onto two PC directions")
                    plt.axis("equal")
                    plt.legend()
                    plt.show()

            self._views["projection"] = output, draw
            draw()
        return self._views["projection"][0]


def minimize_speed(system, initial_guess, bounds=(-2.0, 2.0), max_steps=500, tolerance=1e-8):
    """Trace projected gradient descent on q(x) = ||f(x)||² / 2.

    Return accepted iterates and whether the final vector-field norm is small.
    A stationary point or an iteration limit is not reported as a fixed point.
    """
    point = np.clip(np.asarray(initial_guess, dtype=float), *bounds)
    path = [point.copy()]
    for _ in range(max_steps):
        residual = system(point)
        if np.linalg.norm(residual) < tolerance:
            break
        cost = 0.5 * np.dot(residual, residual)
        gradient = compute_jacobian(system, point).T @ residual
        step = 0.2
        for _ in range(40):
            candidate = np.clip(point - step * gradient, *bounds)
            displacement = candidate - point
            next_residual = system(candidate)
            next_cost = 0.5 * np.dot(next_residual, next_residual)
            if next_cost <= cost + 1e-4 * np.dot(gradient, displacement):
                break
            step *= 0.5
        else:
            break
        if np.linalg.norm(displacement) < 1e-14:
            break
        point = candidate
        path.append(point.copy())
    converged = np.linalg.norm(system(point)) < tolerance
    return np.array(path), converged


def compare_dynamics_and_search(system, initial_states):
    """Plot forward dynamics and accepted search iterates from matched starts."""
    initial_states = np.asarray(initial_states, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    titles = ["Forward dynamics: follow f(x)", "Fixed-point search: decrease ½‖f(x)‖²"]
    for ax, title in zip(axes, titles):
        plot_2d_flowfield(system, space=np.linspace(-2, 2, 15), title=title, ax=ax)
        ax.collections[0].set_alpha(0.25)
        ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal")

    dynamics_paths, search_paths, converged = [], [], []
    times = np.linspace(0, 8, 401)
    for index, start in enumerate(initial_states):
        color = plt.get_cmap("tab20")(index % 20)
        solution = solve_ivp(
            lambda time, state: system(state), (times[0], times[-1]), start,
            t_eval=times, rtol=1e-7, atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        dynamics = solution.y.T
        search, found = minimize_speed(system, start)
        dynamics_paths.append(dynamics)
        search_paths.append(search)
        converged.append(found)
        for ax, path in zip(axes, [dynamics, search]):
            ax.plot(*path.T, color=color, alpha=0.3, linewidth=1.3)
            if len(path) > 1:
                middle = len(path) // 2
                ax.annotate(
                    "", xy=path[middle], xytext=path[middle - 1],
                    arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.65},
                )
        axes[0].scatter(*dynamics[-1], color=color, marker="s", s=28)
        axes[1].scatter(*search[-1], color=color, marker="*" if found else "x", s=70)

    for ax in axes:
        ax.scatter(*initial_states.T, facecolors="none", edgecolors="black", s=35, label="Starting points")
    axes[0].scatter([], [], color="black", marker="s", label="State at t = 8")
    axes[1].scatter([], [], color="black", marker="*", label="Search endpoint: small residual")
    if not all(converged):
        axes[1].scatter([], [], color="black", marker="x", label="Search stopped: nonzero residual")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.show()
    endpoints = np.array([path[-1] for path, found in zip(search_paths, converged) if found])
    return {
        "dynamics_paths": dynamics_paths, "search_paths": search_paths,
        "converged": np.array(converged), "fixed_points": endpoints,
    }
