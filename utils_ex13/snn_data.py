import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Layout, FloatSlider, Output, VBox, Button
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils_ex13.ann_data import plot_data, get_dataloader as get_ann_dataloader, plot_loader


class YinYangPoissonDataset(Dataset):
    """
    A PyTorch dataset for generating Poisson-distributed spike data
    based on the Yin-Yang pattern, with three distinct classes ('yin', 'yang', 'dot').
    Attributes:
    -----------
    rng : np.random.RandomState - random number generator initialized with seed.
    transform : callable, optional - optional transform to apply to samples.
    r_small : float - radius of the small inner circles in the Yin-Yang pattern.
    r_big : float - radius of the larger encompassing circle in the Yin-Yang pattern.
    __vals : List[np.ndarray] - list of Poisson-distributed spike train tensors (T, 4).
    __cs : List[int] - list of class labels (0 for 'yin', 1 for 'yang', 2 for 'dot').
    class_names : List[str] - names of the classes ('yin', 'yang', 'dot').
    T : int - number of timesteps for spike trains.
    fmin : int - minimum frequency for Poisson spike generation (in Hz).
    fmax : int - maximum frequency for Poisson spike generation (in Hz).
    """

    def __init__(self, size: int, seed: int,
                 r_small: float = .1, r_big: float = .5,
                 T: int = 200, fmin: int = 1, fmax: int = 100,
                 homogeneous_poisson: bool = False,
                 poisson_to_dots: bool = False):
        super(YinYangPoissonDataset, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']
        self.T = T
        self.fmin = fmin
        self.fmax = fmax
        self.homogeneous_poisson = homogeneous_poisson
        self.poisson_to_dots = poisson_to_dots
        for i in range(size):
            ## get coordinates of the i-th datapoint
            ## according to Yin-Yang dataset
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            x_flipped = 1. - x
            y_flipped = 1. - y
            val = np.array([x, y, x_flipped, y_flipped])
            ## convert coordinates into spike trains
            poisson_val = self.get_poisson(val)
            self.__vals.append(poisson_val)
            self.__cs.append(c)

    def get_poisson(self, val: np.ndarray):
        ## convert coordinates into Homogeneous Poisson firing rates
        if self.homogeneous_poisson == True:
            freq = (self.fmin) * np.ones_like(val) + ((self.fmax - self.fmin)) * val
            # respect T as interval over time
            prob = freq / self.T  # probability of firing at each time step
            poisson_val = (np.random.rand(self.T, 4) < (prob[np.newaxis, :])).astype(np.float32)
        ## convert coordinates into the exact number of spikes per second
        ## equally spaced in time (with 2 ms jitter)
        else:
            freq = (self.fmin) * np.ones_like(val) + ((self.fmax - self.fmin)) * val
            num_spikes = np.floor(freq * (self.T / 1e3)).astype(int)
            poisson_val = np.zeros((len(freq), self.T), dtype=int)
            for i, n in enumerate(num_spikes):
                if n > 0:
                    event_indices = np.linspace(0, self.T, n, endpoint=False).astype(int)
                    poisson_val[i, event_indices + np.random.randint(-2, 2 + 1, size=len(event_indices))] = 1
            poisson_val = poisson_val.T.astype(np.float32)
        ## convert back to coordinates of a given dot
        if self.poisson_to_dots == True:
            poisson_val = poisson_val.sum(axis=0) * (1e3 / self.T)
        return poisson_val

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        return sample

    def __len__(self):
        return len(self.__cs)


def get_dataloader(train_size=5000, val_size=1000, test_size=1000, seed=42, batch_size=1000, shuffle_train=True):
    dataset_train = YinYangPoissonDataset(size=train_size, seed=seed)
    dataset_validation = YinYangPoissonDataset(size=val_size, seed=seed)
    dataset_test = YinYangPoissonDataset(size=test_size, seed=seed)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_sample_slider(dataset: YinYangPoissonDataset):

    style = {'description_width': 'initial'}

    x_slider = FloatSlider(min=0, max=1, step=0.01, value=0.5, description='x:', style=style, layout=Layout(width='500px'))
    y_slider = FloatSlider(min=0, max=1, step=0.01, value=0.5, description='y:', style=style, layout=Layout(width='500px'))

    refresh = Button(description="Resample", button_style='success')

    output = Output()

    def refresh_output():
        with output:
            output.clear_output(wait=True)
            x = x_slider.value
            y = y_slider.value
            sample = dataset.get_poisson(np.array([x, y, 1. - x, 1. - y]))
            plot_sample(sample, title=f"{x} {y}")
            train_loader, _, _= get_ann_dataloader()
            plot_loader(train_loader, highlight=[[x, y, -1]])
            plt.show()

    x_slider.observe(lambda _: refresh_output(), names='value')
    y_slider.observe(lambda _: refresh_output(), names='value')
    refresh.on_click(lambda _: refresh_output())

    refresh_output()

    display(VBox([x_slider, y_slider, refresh, output]))



def plot_sample(sample, title: str = "", fsize=(15, 6)):
    features = ["x", "y", "1-x", "1-y"]

    time_steps = np.arange(sample.shape[0])
    num_features = sample.shape[1]

    # Create the plot
    plt.figure(figsize=fsize)
    for i in range(num_features):
        # Offset each feature for better visibility
        plt.step(time_steps, sample[:, i] + i * 1.5, where='mid', label=features[i])

    # Customizing the plot
    plt.xlabel("Time Steps")
    plt.ylabel("Feature Value")
    plt.yticks([1.5 * i for i in range(num_features)], [f for f in features])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()