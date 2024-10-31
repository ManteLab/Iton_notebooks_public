from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ipywidgets import interact, Layout
from ipywidgets.widgets import IntSlider
from mpl_toolkits.axes_grid1 import host_subplot


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


def load_cnn_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    return model


def load_imagenet_sample(path='./imagenet/', limit=None):

    # normalization for alexnet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataset_loader = DataLoader(
        dataset, batch_size=1, shuffle=False
    )
    
    batch_images = [(img, label) for img, label in islice(dataset_loader, limit)]
    return (
        torch.concatenate([img for img, label in batch_images]),
        torch.concatenate([label for img, label in batch_images])
    )


def plot_images_grid(images, labels, cols=10):
    rows = np.unique(labels).shape[0]
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 8))

    for i, label in enumerate(np.unique(labels)):
        for j, image in zip(range(cols), images[labels == label]):
            axes[i, j].imshow(np.einsum('ijk -> jki', image).clip(0, 1))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()


def compute_redness(batch):
    redness = batch.mean(axis=(-2, -1))[:, 0] - batch.mean(axis=(-2, -1))[:, 1:].mean(axis=-1)
    return redness


def pick_most_red(batch, nr_images):
    redness = compute_redness(batch)
    return batch[np.argsort(-redness)[:nr_images]]


def pick_least_red(batch, nr_images):
    redness = compute_redness(batch)
    return batch[np.argsort(redness)[:nr_images]]


def pick_redness_extrema(images, labels, sample=40, nr_images=5):
    return torch.concatenate([
        torch.concatenate([
            pick_most_red(images[ labels == label ][:sample], nr_images),
            pick_least_red(images[ labels == label ][:sample], nr_images),
        ])
        for label in labels.unique()
    ]), labels.unique().repeat_interleave(nr_images * 2)


from torch.nn.functional import softmax

def get_activations_with_names(model, batch, subsample=None):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    activations = {}
    x = batch

    def subsample_activation(activation):
        if subsample is None:
            return activation
        else:
            batch_size = activation.shape[0]
            activation_flat = activation.view(batch_size, -1)
            total_elements = activation_flat.shape[1]
            subsample_size = min(subsample, total_elements)
            indices = torch.randperm(total_elements)[:subsample_size]
            sampled_activation = activation_flat[:, indices]
            return sampled_activation

    for idx, layer in enumerate(model.features):
        x = layer(x)
        activation = x.detach()
        layer_name = f'({idx})_{layer.__class__.__name__}'
        activations[f'features_{layer_name}'] = subsample_activation(activation)

    x = model.avgpool(x)
    activation = x.detach()
    activations['avgpool'] = subsample_activation(activation)
    x = torch.flatten(x, 1)

    for idx, layer in enumerate(model.classifier):
        x = layer(x)
        activation = x.detach()
        layer_name = f'({idx})_{layer.__class__.__name__}'
        activations[f'classifier_{layer_name}'] = subsample_activation(activation)

    
    # activations[f'readout'] = softmax(subsample_activation(activation))
    return activations


def plot_similarity_matrices_col(similarity_matrices):
    fig, axes = plt.subplots(nrows=len(similarity_matrices), ncols=1, figsize=(16, 30))

    for (n, sm), ax in zip(similarity_matrices.items(), axes.flat, strict=True):
        ax.set_title(n, size=10)
        ax.imshow(sm)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_second_order_similarity_matrix(second_order_similarity_matrix, layer_names):
    plt.imshow(second_order_similarity_matrix)

    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    
    plt.gca().set_xticks(np.arange(len(layer_names)))
    plt.gca().set_yticks(np.arange(len(layer_names)))
    
    plt.gca().set_xticklabels(layer_names)
    plt.gca().set_yticklabels(layer_names)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor");

    plt.show()


def plot_variance_explained(pcs):
    get_ipython().run_line_magic('matplotlib', 'inline' if is_colab else 'widget')

    def update_plot(layer_nr):
        plt.clf()
        layer_name = list(pcs.keys())[layer_nr]
        _, var_explained, cum_var_explained = pcs[layer_name]
        
        host = host_subplot(111)
        parasite = host.twinx()
        axes = [host, parasite]
        
        axes[0].plot(cum_var_explained, color='C0', label='cumulative variance explained')
        axes[0].set_xlabel('# component')
        axes[0].set_ylabel('cumulative variance explained', color='C0')
        axes[0].set_ylim([0., 1.05])
        
        axes[1].plot(var_explained / var_explained.sum(), color='C2', label='variance explained')
        axes[1].set_ylabel('variance explained', color='C2')
        axes[1].set_yscale('log')
        axes[1].set_ylim([1e-8, 1e-0])

        host.set_title(f'Layer: {layer_name}')
        host.legend()
        plt.tight_layout()
        plt.show()

    layout = Layout(width='600px')
    style = {'description_width': '150px'}
    options = list(pcs.keys())
    slider = IntSlider(
        min=0,
        max=len(pcs) - 1,
        description='Layer #',
        layout=layout,
        style=style,
        continuous_update=continuous_update
    )
    
    interact(update_plot, layer_nr=slider)
