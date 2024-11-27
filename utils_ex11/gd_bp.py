import io
import itertools
import math
import sys

import jax
from IPython.display import display
from ipywidgets import FloatSlider, FloatLogSlider, Layout, IntSlider, Output, Button, VBox, SelectionSlider, widgets, \
    HBox

from IPython.display import clear_output, display

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import copy
import seaborn as sns
import time

class NeuralNetwork:

    def __init__(self, key, input_dim: int, hidden_units_1: int, hidden_units_2: int, output_dim: int = 1):
        key1, key2, key3 = jrandom.split(key, 3)

        # Initialize weights and biases for the first hidden layer
        self.W1 = jrandom.normal(key1, (input_dim, hidden_units_1)) * jnp.sqrt(2.0 / input_dim)
        self.b1 = jnp.zeros((1, hidden_units_1))

        # Initialize weights and biases for the second hidden layer
        self.W2 = jrandom.normal(key2, (hidden_units_1, hidden_units_2)) * jnp.sqrt(2.0 / hidden_units_1)
        self.b2 = jnp.zeros((1, hidden_units_2))

        # Initialize weights and biases for the output layer
        self.W3 = jrandom.normal(key3, (hidden_units_2, output_dim)) * jnp.sqrt(2.0 / hidden_units_2)
        self.b3 = jnp.zeros((1, output_dim))

    def forward(self, xs):
        # First hidden layer with ReLU activation
        self.xs = xs
        self.z1 = jnp.dot(xs, self.W1) + self.b1
        self.h1 = jnp.maximum(0, self.z1)  # ReLU activation

        # Second hidden layer with ReLU activation
        self.z2 = jnp.dot(self.h1, self.W2) + self.b2
        self.h2 = jnp.maximum(0, self.z2)  # ReLU activation

        # Output layer (no activation function)
        y_pred = jnp.dot(self.h2, self.W3) + self.b3
        return y_pred.flatten()

    def backward(self, dL_dy_pred, learning_rate=0.01):
        # Gradients for W3 and b3
        dL_dW3 = jnp.dot(self.h2.T, dL_dy_pred)
        dL_db3 = jnp.sum(dL_dy_pred, axis=0, keepdims=True)

        # Backpropagate to the second hidden layer
        dL_dh2 = jnp.dot(dL_dy_pred, self.W3.T)
        dL_dz2 = dL_dh2 * (self.z2 > 0)  # Derivative of ReLU

        dL_dW2 = jnp.dot(self.h1.T, dL_dz2)
        dL_db2 = jnp.sum(dL_dz2, axis=0, keepdims=True)

        # Backpropagate to the first hidden layer
        dL_dh1 = jnp.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_dh1 * (self.z1 > 0)  # Derivative of ReLU

        dL_dW1 = jnp.dot(self.xs.T, dL_dz1)
        dL_db1 = jnp.sum(dL_dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W3 -= learning_rate * dL_dW3
        self.b3 -= learning_rate * dL_db3


def iplot_gn_model():
    output_plot = Output()
    error_log = Output()

    # Update the plot (triggered by the button)

    import matplotlib.pyplot as plt
    import networkx as nx

    def plot_network(ax, nn: NeuralNetwork):
        input_dim = 1
        hidden_units_1 = nn.W1.shape[1]
        hidden_units_2 = nn.W2.shape[1]

        # Create a directed graph to represent the network
        G = nx.DiGraph()

        # Add nodes for each layer
        input_node = [f'Input']
        hidden1_nodes = [f'Hidden1_{i + 1}' for i in range(hidden_units_1)]
        hidden2_nodes = [f'Hidden2_{i + 1}' for i in range(hidden_units_2)]
        output_node = ['Output']

        # Add nodes to the graph
        G.add_node(input_node[0], layer='Input')
        G.add_nodes_from(hidden1_nodes, layer='Hidden1')
        G.add_nodes_from(hidden2_nodes, layer='Hidden2')
        G.add_node(output_node[0], layer='Output')

        # Connect nodes between layers to represent weights
        for hidden1_node in hidden1_nodes:
            G.add_edge(input_node[0], hidden1_node)

        for hidden1_node in hidden1_nodes:
            for hidden2_node in hidden2_nodes:
                G.add_edge(hidden1_node, hidden2_node)

        for hidden2_node in hidden2_nodes:
            G.add_edge(hidden2_node, output_node[0])

        # Define positions for each layer for visual clarity
        max_nodes = max(len(hidden1_nodes), len(hidden2_nodes))
        scale = 6
        max_height = max_nodes * scale
        layer_positions = {
            **{input_node[0]: (0, max_height / 2)},
            **{node: (1, (i+(max_nodes - len(hidden1_nodes))/2)*scale) for i, node in enumerate(hidden1_nodes)},
            **{node: (2, (i+(max_nodes - len(hidden2_nodes))/2)*scale) for i, node in enumerate(hidden2_nodes)},
            **{output_node[0]: (3, max_height / 2)},
        }

        # Draw the graph on the provided axis
        nx.draw(G, pos=layer_positions, with_labels=True, node_size=1000, node_color='lightblue', font_size=4, arrows=True, ax=ax)
        ax.set_title("Neural Network Graph Structure")

    def generate_plots(
            learning_rate: float,
            hidden_unit_1: int,
            hidden_unit_2: int,
            n_epochs: int,
            network_model_only: bool = False
    ):
        data_key, noise_key, model_key = jrandom.split(jrandom.PRNGKey(0), 3)

        def f(x):
            return x + 0.5 * jnp.square(x)

        # Generate two-dimensional data
        batch_size = 32
        n_data_points = 1280
        train_steps = n_data_points // batch_size
        X = (jrandom.uniform(data_key, (n_data_points, 1)) - 0.5) * 10
        y = f(X) + jrandom.normal(noise_key, (n_data_points, 1)) * 0.1

        nn = NeuralNetwork(key=model_key, input_dim=X.shape[1], hidden_units_1=hidden_unit_1, hidden_units_2=hidden_unit_2, output_dim=1)

        def train(nn, X, y, epochs=1000, learning_rate=0.01):
            models_per_epoch = []
            losses_per_epoch = []
            for epoch in range(epochs):
                models_per_iteration = []
                losses_per_iteration = []
                running_loss = None
                for X_batch, y_batch in zip(X.reshape(-1, batch_size, X.shape[1]),
                                            y.reshape(-1, batch_size, y.shape[1])):
                    y_pred = nn.forward(X_batch)

                    y_pred = np.expand_dims(y_pred, axis=1)

                    loss = np.mean((y_batch - y_pred) ** 2)

                    if not math.isfinite(loss):
                        break

                    models_per_iteration.append(copy.deepcopy(nn))
                    losses_per_iteration.append(float(loss))

                    if running_loss is None:
                        running_loss = loss
                    else:
                        running_loss = 0.95 * running_loss + 0.05 * loss
                    dL_dy_pred = 2 * (y_pred - y_batch) / y.shape[0]
                    nn.backward(dL_dy_pred, learning_rate)
                else:
                    models_per_epoch.append(copy.deepcopy(nn))
                    losses_per_epoch.append(float(running_loss))
                    continue
                # In case training failed:
                return models_per_epoch, losses_per_epoch, models_per_iteration, losses_per_iteration
            return models_per_epoch, losses_per_epoch, None, None

        def plot_data(ax, X, y):
            ax.scatter(X, y, s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('y')
            ax.set_title('Data')
            return ax

        def plot_model(ax, nn, color='green', alpha=1.0):
            xs = jnp.linspace(-5, 5, 100)[:, None]
            ax.plot(xs, nn.forward(xs), label='Model', color=color, alpha=alpha)
            ax.set_title('Model')
            return ax

        if not network_model_only:
            models_per_epoch, losses_per_epoch, models_per_iteration, losses_per_iteration = train(nn, X, y, epochs=n_epochs, learning_rate=learning_rate)
        else:
            models_per_epoch, losses_per_epoch, models_per_iteration, losses_per_iteration = [], [], [], []

        train_failed = losses_per_iteration is not None
        epochs = list(range(0, len(losses_per_epoch)))
        if train_failed:
            epochs += list(map(lambda x: len(losses_per_epoch) + (x+1) / train_steps, range(0, len(losses_per_iteration))))
        history = losses_per_epoch + (losses_per_iteration or [])
        models = models_per_epoch + (models_per_iteration or [])
        frames = list(range(0, len(models))) if not network_model_only else [0]
        max_epoch = max(epochs, default=n_epochs)
        assert len(models) == len(history) == len(epochs), f"{len(frames)} == {len(models)} == {len(history)} == {len(epochs)}"
        frames_in_memory = []
        for step in frames:
            output_plot.clear_output(wait=True)
            last_frame = step == frames[-1]
            if step % 10 == 0 or last_frame:
                fig = plt.figure(figsize=(12, 6))
                ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
                ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
                plot_data(ax1, X, y)
                for i, model in enumerate(models[:step]):
                    if model == models[:step][-1]:
                        plot_model(ax1, model, color='green')
                    else:
                        alpha = 0.1 - 0.1 * i / len(models[:step])
                        plot_model(ax1, model, color='orange', alpha=alpha)
                sns.lineplot(x=epochs[:step], y=history[:step], ax=ax2)
                if train_failed:
                    plt.suptitle(f'Training failed during epoch {len(losses_per_epoch)+1} - reduce learning rate!')
                    ax2.set_yscale('log')
                else:
                    ax2.set_ylim(0, None)
                ax2.set_title('Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_xlim(0, max_epoch)
                ax2.set_ylabel('Loss')
                ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
                plot_network(ax3, nn)
                plt.tight_layout()

                # Save to memory instead of a file
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)  # Rewind buffer to the beginning
                frames_in_memory.append(buf.read())  # Store raw image data
                buf.close()
                plt.close(fig)
        return frames_in_memory

    # Create sliders for synaptic weight, frequency, and length
    style = {'description_width': 'initial'}

    # lr_slider = SelectionSlider(description='learning rate:', options=['0.001', '0.01', '0.1', '1.0', '10.0'], value='0.1', style=style, layout=Layout(width='500px'))
    lr_slider = FloatLogSlider(min=-4, max=1, step=1, value=0.1, description='learning rate:', style=style, layout=Layout(width='500px'))
    hidden_units_1_slider = IntSlider(min=1, max=32, step=1, value=16, description='# hidden units (1st layer):', style=style, layout=Layout(width='500px'), continuous_update=False)
    hidden_units_2_slider = IntSlider(min=1, max=32, step=1, value=8, description='# hidden units (2nd layer):', style=style, layout=Layout(width='500px'), continuous_update=False)
    epoch_slide = IntSlider(min=10, max=200, step=1, value=100, description='# epochs:', style=style, layout=Layout(width='500px'))

    def generate_plots_by_values(network_model_only=False):
        return generate_plots(
            learning_rate=float(lr_slider.value),
            hidden_unit_1=hidden_units_1_slider.value,
            hidden_unit_2=hidden_units_2_slider.value,
            n_epochs=epoch_slide.value,
            network_model_only=network_model_only,
        )

    frames_in_memory = []

    play = widgets.Play(
        value=0, min=0, max=1, step=1, interval=200, description="Play"
    )
    frame_slider = IntSlider(min=0, max=1, step=1, description="Frame")
    widgets.jslink((play, "value"), (frame_slider, "value"))
    prev_button = widgets.Button(description="<", layout=widgets.Layout(width="30px", height="30px"))
    next_button = widgets.Button(description=">", layout=widgets.Layout(width="30px", height="30px"))

    def on_prev_click(_):
        if frame_slider.value > frame_slider.min:
            frame_slider.value -= frame_slider.step
    def on_next_click(_):
        if frame_slider.value < frame_slider.max:
            frame_slider.value += frame_slider.step

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)

    image_widget = widgets.Image(format='png', layout=widgets.Layout(visibility='hidden'))

    with output_plot:
        display(image_widget)

    def simulate_frames(_, network_model_only=False):
        global frames_in_memory

        desc = simulation_button.description
        image_widget.layout.visibility = 'hidden'
        simulation_button.description = "Generating Frames..."
        simulation_button.disabled = True

        try:
            frames_in_memory = generate_plots_by_values(network_model_only)
            play.max = len(frames_in_memory) - 1
            frame_slider.max = len(frames_in_memory) - 1

            image_widget.value = frames_in_memory[0]
        except Exception as e:
            with error_log:
                print(f"Error: {e}")
        finally:
            simulation_button.disabled = False
            simulation_button.description = desc
            image_widget.layout.visibility = 'visible'

    def update_output(change):
        global frames_in_memory
        if frames_in_memory and len(frames_in_memory) > change["new"]:
            image_widget.value = frames_in_memory[change["new"]]


    from functools import partial
    hidden_units_1_slider.observe(partial(simulate_frames, network_model_only=True), names='value', type='change')
    hidden_units_2_slider.observe(partial(simulate_frames, network_model_only=True), names='value', type='change')

    simulation_button = Button(description="Generate Animation", button_style='success')
    simulation_button.on_click(simulate_frames)

    play.observe(update_output, names="value")

    ui = VBox(
        [lr_slider, epoch_slide, hidden_units_1_slider, hidden_units_2_slider,
         simulation_button, HBox([HBox([prev_button, play, next_button]), frame_slider]), output_plot, error_log])

    display(ui)


"""
BACKPROPAGATION
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Node:
    """ stores a single scalar value and its gradient
    The base structure is taken from Andrej Karpathy's Micrograd project which is published on Github.
    For more details, see the `Micrograd <https://github.com/karpathy/micrograd>`_.

    It was then extended by some more functions and added more parameters to the constructor.
    """

    def __init__(self, data, children=(), op='', label='', trainable=False, loss_related=False):
        """
        Initializes a new instance of the Node class.

        Parameters:
        -----------
            data (float): The scalar value to store in the node.
            children (tuple of Node, optional): Child nodes linked to this node. Defaults to an empty tuple.
            op (str, optional): The operation associated with this node, if any. Defaults to an empty string.
            label (str, optional): A human-readable label for identifying the node. Defaults to an empty string.
            trainable (bool, optional): Indicates if the node is trainable. Defaults to False.
            loss_related (bool, optional): Indicates if the node is related to the loss calculation. Defaults to False.
        """
        self.data = data
        self.label = label
        self.grad = 0
        self.trainable = trainable
        # internal variables used for autograd graph construction
        self.backward = lambda: None
        self.prev = set(children)
        self.loss_related = loss_related
        self.op = op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), '+', f"{self.label}+{other.label}")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*', f"{self.label}*{other.label}")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward = backward

        return out

    def add(self, other):
        return self + other

    def mult(self, other):
        return self * other

    def relu(self):
        out = Node(0 if self.data < 0 else self.data, (self,), 'ReLU', f"relu({self.label})")

        def backward():
            local_grad = 1 if out.data > 0 else 0
            self.grad += local_grad * out.grad

        out.backward = backward

        return out

    def sig(self):
        out = Node(sigmoid(self.data), (self,), 'Sig', f"sig({self.label})")

        def backward():
            local_grad = sigmoid(self.data) * (1 - sigmoid(self.data))
            self.grad += local_grad * out.grad

        out.backward = backward

        return out

    def squared_error(self, y):
        y = y if isinstance(y, Node) else Node(y)
        out = Node((self.data - y.data) ** 2, (self, y), op='Squared Error', label="SE", loss_related=True)

        def backward():
            local_grad = 2 * (self.data - y.data)
            self.grad += local_grad * out.grad
            # Y is non-trainable
            y.grad = 0.0

        out.backward = backward
        out.grad = 1.0

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Node(self.data ** other, (self,), f'**{other}')

        def backward():
            local_grad = (other * self.data ** (other - 1))
            self.grad += local_grad * out.grad

        out.backward = backward

        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.data, self.label, self.trainable, self.op) == (
                other.data, other.label, other.trainable, other.op)
        return False

    def __hash__(self):
        return hash((self.data, self.label, self.trainable, self.op))

    def __repr__(self):
        return f"Value(op={self.op}, label={self.label}, data={self.data}, grad={self.grad})"


from graphviz import Digraph

import matplotlib.pyplot as plt
from collections import deque


def reset_gradients(loss):
    """
    Can be used to reset the gradients before a new epoch

    Parameters
    ----------
        loss : Last node in our neural network
    """

    # Topsort is basically not required but already available to get all nodes
    nodes = topsort(loss)
    for node in nodes:
        node.grad = 0.0


def bfs(loss):
    """
    Does a BFS and returns the nodes in order they appear. If there are multiple nodes at the same distance,
    it will place those nodes with less children first.

    Parameters
    ----------
        loss : Last node in our neural network

    Returns
    ----------
        List with nodes in BFS order
    """

    bfs = []
    queue = deque([loss])

    while queue:
        current = queue.popleft()
        if current not in bfs:
            bfs.append(current)
        children = sorted(
            (node for node in current.prev if node not in bfs),
            key=lambda node: (
                len(node.prev) if hasattr(node, 'prev') and node.prev is not None else 0,
                node.label
            )
        )
        queue.extend(children)  # deque.extend() is used to add multiple elements at the end

    return bfs


def reversed_topsort(loss) -> []:
    """
    Parameters
    ----------
        loss : Last node in our neural network

    Returns
    ----------
        Reversed topsort order
    """
    return reversed(topsort(loss))


def topsort(loss) -> []:
    """
    Topological sort of the graph

    Parameters
    ----------
        loss : Last node in our neural network

    Returns
    ----------
        Nodes in topological order
    """
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v.prev:
                build_topo(child)
            topo.append(v)

    build_topo(loss)
    return topo


def trainable_nodes(loss):
    """
    Returns all trainable nodes

    Parameters
    ----------
        loss : Last node in our neural network

    Returns
    ----------
        list with all trainable nodes
    """
    all_nodes = topsort(loss)
    return [n for n in all_nodes if n.trainable == True]


def print_trainable_weights(root):
    print("The values of the trainable nodes are:")
    print(*(sorted(trainable_nodes(root), key=lambda node: node.label)), sep='\n')


def print_loss_history(loss_history):
    print(f"Loss history: {loss_history}")


def plot_history(loss_history):
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def _trace(loss):
    """
    Creates a trace of the entire network and returns all nodes and edges

    Parameters:
    -----------
        loss: Loss node which is the latest node in the network

    Return
    -----------
        (nodes, edges) in the network
    """
    nodes, edges = list(), list()

    def build(v):
        if v not in nodes:
            nodes.append(v)
            for child in sorted(v.prev, key=lambda n: n.label):
                edges.append((child, v))
                build(child)

    build(loss)
    return nodes, edges


def draw_nn(loss):
    """
    Returns a DiGraph representation of the neural network with given loss node

    Parameters:
    -----------
        loss: Loss node which is the latest node in the network

    Return
    -----------
        Digraph showing the neural network from left to right. Trainable nodes have a red border.
    """
    #dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'size': '6,6'})  # LR = left to right
    trainable = trainable_nodes(loss)
    nodes, edges = _trace(loss)
    for n in nodes:
        uid = str(id(n))
        border = "black"
        penwidth = '1'
        if n in trainable:
            border = "red"
            penwidth = '4'

        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ %s | data %.6f | grad %.6f}" % (n.label, n.data, n.grad), shape='record',
                 fillcolor='white', style='filled', color=border, penwidth=penwidth)
        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n.op, label=n.op)
            style = 'solid'
            if n.loss_related:
                style = 'dotted'
            # and connect this node to it
            dot.edge(uid + n.op, uid, style=style)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        style = 'solid'
        if n2.loss_related:
            style = 'dotted'
        dot.edge(str(id(n1)), str(id(n2)) + n2.op, style=style)

    return dot


def draw_nn_emphasizing_nodes(loss, currently_updated_node, already_visited_forward=None, already_visited_backward=None, backward_mode=True):
    if already_visited_backward is None:
        already_visited_backward = []

    already_visited = already_visited_backward if backward_mode else already_visited_forward

    title = "Forward pass" if not backward_mode else "Backward pass"
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'size': '11,11', 'label': title})  # LR = left to right
    trainable = trainable_nodes(loss)
    nodes, edges = _trace(loss)

    if backward_mode and already_visited_forward is None:
        already_visited_forward = nodes  # all visited

    for n in nodes:
        uid = str(id(n))
        fill = 'grey'
        border = "black"
        penwidth = '1'
        if n == currently_updated_node:
            fill = 'yellow'
        elif n in already_visited:
            fill = 'lightblue'
        if n in trainable:
            border = "red"
            if not backward_mode:
                fill = 'lightblue'
            penwidth = '2'

        # for any value in the graph, create a rectangular ('record') node for it
        data_v = round(n.data, 2) if n in already_visited_forward or (n == currently_updated_node and not backward_mode) or n in trainable else '?'
        grad_v = round(n.grad, 2) if n in already_visited_backward or (n == currently_updated_node and backward_mode) else '?'
        dot.node(name=uid, label=f"{n.label} | data {data_v} | grad {grad_v}", shape='record',
                 fillcolor=fill, style='filled', color=border, penwidth=penwidth)
        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n.op, label=n.op)
            style = 'solid'
            if n.loss_related:
                style = 'dotted'
            # and connect this node to it
            dot.edge(uid + n.op, uid, style=style)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        style = 'solid'
        if n2.loss_related:
            style = 'dotted'
        dot.edge(str(id(n1)), str(id(n2)) + n2.op, style=style)

    return dot

def backprop():

    output = Output()
    print_log = Output()
    next_btn = Button(description="Next", button_style='success')

    def sim_step():
        from IPython.display import clear_output, display

        y = Node(5.4, label="y", loss_related=True)  # expected output
        x1 = Node(1.5, label="x1")  # input

        # Trainable parameters
        w1 = Node(0.5, label="w1", trainable=True)
        w2 = Node(0.5, label="w2", trainable=True)
        b1 = Node(2.0, label="b1", trainable=True)
        b2 = Node(-2.0, label="b2", trainable=True)

        def forward(interactive=True):
            h1 = x1 * w1 + b1  # Hidden layer
            y_hat = h1 * w2 + b2  # Output layer
            loss = y_hat.squared_error(y)  # Loss function
            topsorted = list(reversed(bfs(loss)))
            already_visited = {x1}
            for n in topsorted:
                if n.trainable:
                    topsorted.remove(n)  # parameters care constants in forward pass
                    # already_visited.add(n)

            already_visited.add(y)
            topsorted.remove(y)

            for i, node in enumerate(topsorted):
                if interactive:
                    already_visited.add(node)
                    currently_updated_node = topsorted[i + 1] if i + 1 < len(topsorted) else None
                    with output:
                        clear_output(wait=True)
                        display(draw_nn_emphasizing_nodes(loss, currently_updated_node, already_visited_forward=already_visited, backward_mode=False))
                    if i < len(topsorted) - 1:
                        yield "Press Next to continue forward pass..."

            yield loss

        def backward(loss, interactive=True):  # backprop

            topsorted = bfs(loss)
            topsorted.remove(y)  # y is constant
            topsorted.remove(x1)  # x1 is constant
            reset_gradients(loss)
            loss.grad = 1.0

            topsorted.insert(0, loss)
            already_visited = {y, x1}

            for i, node in enumerate(topsorted):
                node.backward()
                if interactive:
                    already_visited.add(node)
                    currently_updated_node = topsorted[i + 1] if i + 1 < len(topsorted) else None
                    with output:
                        clear_output(wait=True)
                        display(draw_nn_emphasizing_nodes(loss, currently_updated_node, already_visited_backward=already_visited))
                    if i < len(topsorted) - 1:
                        yield "Press Next to continue backward pass..."

        interactive = True
        yield "Press Next to start simulation, by starting forward pass: "
        clear_output(wait=True)

        forward_gen = forward(interactive=interactive)
        for message in forward_gen:
            yield message
            loss = message

        yield "Press Next to start backward pass (backprop): "
        clear_output(wait=True)

        backward_gen = backward(loss, interactive=interactive)
        for message in backward_gen:
            yield message

        if not interactive:
            draw_nn(loss)

        yield "Simulation finished. Press Next to restart."

    def inf_sim():
        while True:
            step = sim_step()
            for x in step:
                yield x
            with output:
                clear_output(wait=False)

    sim = inf_sim()

    def inf_sim_step():
        message = next(sim)
        if isinstance(message, str):
            with print_log:
                # print_log.clear_output(wait=True)
                clear_output(wait=True)
                print(message)

    next_btn.on_click(lambda _: inf_sim_step())
    inf_sim_step()

    display(VBox([next_btn, print_log, output]))