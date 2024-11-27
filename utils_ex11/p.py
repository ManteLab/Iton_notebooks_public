# Original source: https://raw.githubusercontent.com/christianversloot/rosenblatts-perceptron/refs/heads/master/p.py
import time

import numpy as np
from IPython.display import display, clear_output
from ipywidgets import Output, FloatSlider, Button, VBox, IntSlider, Layout, FloatLogSlider
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


# Basic Rosenblatt Perceptron implementation
class RBPerceptron:

    # Constructor
    def __init__(self, number_of_epochs=100, learning_rate=0.1, bias = True, feature_engineering=False):
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.bias = bias
        self.feature_engineering = feature_engineering

    def train(self, X, D):
        # Initialize weights vector with zeroes
        X2 = X.copy()
        if self.feature_engineering:
            X2 = np.c_[X2, X2[:, 0] * X2[:, 1]]
        if self.bias:
            X2 = np.c_[np.ones(X2.shape[0]), X2]
        num_features = X2.shape[1]
        np.random.seed(2)
        self.w = np.random.uniform(-0.01, 0.01, num_features)
        # Perform the epochs
        for i in range(self.number_of_epochs):
            changed = False
            # For every combination of (X_i, D_i)
            for sample_orig, sample, desired_outcome in zip(X, X2, D):
                # Generate prediction and compare with desired outcome
                prediction = self.predict(sample_orig.reshape(1, -1))[0]
                difference = (desired_outcome - prediction)
                changed = changed or difference != 0
                # Compute weight update via Perceptron Learning Rule
                weight_update = self.learning_rate * difference
                self.w += weight_update * sample
                yield i, sample_orig
            if not changed:
                break
        return self

    def _predict(self, sample):
        outcome = np.dot(sample, self.w)
        return np.where(outcome > 0, 1, 0)


    def predict(self, sample):
        if self.feature_engineering:
            sample = np.c_[sample, sample[:, 0] * sample[:, 1]]
        if self.bias:
            sample = np.c_[np.ones(sample.shape[0]), sample]
        return self._predict(sample)


def generate_linear_separable_data(scale: float):
    n_samples = 15
    zeros = np.zeros(n_samples)
    ones = zeros + 1
    targets = np.concatenate((zeros, ones))

    np.random.seed(42)
    small = np.random.normal(-1, scale, (n_samples, 2))
    large = np.random.normal(1, scale, (n_samples, 2))

    X = np.concatenate((small, large))
    D = targets

    # Standardize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    D = D[idx]

    return X, D


def generate_nonlinear_xor_data(scale: float):
    n_samples = 6
    zeros = np.zeros(n_samples * 2)
    ones = zeros + 1
    targets = np.concatenate((zeros, ones))

    np.random.seed(42)
    d_00 = np.random.normal([-1, -1], scale, (n_samples, 2))
    d_10 = np.random.normal([1, -1], scale, (n_samples, 2))
    d_01 = np.random.normal([-1, 1], scale, (n_samples, 2))
    d_11 = np.random.normal([1, 1], scale, (n_samples, 2))

    X = np.concatenate((d_00, d_11, d_10, d_01))
    D = targets

    # Standardize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    D = D[idx]

    return X, D


def show_linear_model(interactive: bool = True):
    show_model(generate_linear_separable_data, RBPerceptron, interactive)


def show_nonlinear_model(interactive: bool = True, feature_engineering=False):
    show_model(generate_nonlinear_xor_data, lambda e, lr: RBPerceptron(e, lr, bias=False, feature_engineering=feature_engineering), interactive, data_scale=0.1)


def show_model(data_f, model_f, interactive: bool = True, data_scale=0.8):
    output = Output()
    log = Output()

    def update(simulation_speed: int, alpha: float, data_scale: float):
        def plot_model(model: RBPerceptron, X, D, sample=None, title: str = 'Perceptron'):
            with output:
                clear_output(wait=True)  # Clear the output widget
                plt.figure(figsize=(6, 4))  # Ensure a consistent figure size
                plot_decision_regions(X, D.astype(np.integer), clf=model)
                plt.title(title)
                plt.xlabel('X1')
                plt.ylabel('X2')
                if sample is not None:
                    plt.scatter(sample[0], sample[1], color='red', s=80, facecolors='none', edgecolors='r')
                plt.show()

        X, D = data_f(data_scale)

        number_of_epochs = 3
        perceptron = model_f(number_of_epochs, alpha)
        for epoch, sample in perceptron.train(X, D):
            if interactive:
                plot_model(perceptron, X, D, sample, title=f'Perceptron - Epoch {epoch+1}/{number_of_epochs}')
                if simulation_speed != 0:
                    plt.pause(simulation_speed / 1000 * 100)
        plot_model(perceptron, X, D)

    style = {'description_width': 'initial'}
    simulation_speed_slider = FloatSlider(min=0, max=10, step=0.1, value=0, description='simulation speed:',
                                          style=style, layout=Layout(width='500px'))
    alpha_slider = FloatLogSlider(value=0.001, base=10, min=-4, max=0, step=0.1, description='Learning Rate',
                                  style=style, layout=Layout(width='500px'))
    data_scale = FloatSlider(value=data_scale, min=0.1, max=1.0, step=0.1, description='data scale (std)', style=style,
                             layout=Layout(width='500px'))
    simulate_btn = Button(description='Simulate')

    simulate_btn.on_click(lambda _: update(simulation_speed_slider.value, alpha_slider.value, data_scale.value))

    if interactive:
        display(VBox([
            alpha_slider,
            simulation_speed_slider,
            data_scale,
            simulate_btn,
            output,
            log
        ]))
    else:
        update(simulation_speed_slider.value, alpha_slider.value, data_scale.value)
        display(output)


def show_non_linear_data():
    X, D = generate_nonlinear_xor_data(0.1)
    y = D.astype(np.integer)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='s', edgecolors='k', label='0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='^', edgecolors='k', label='1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend(loc='upper right')
    plt.show()