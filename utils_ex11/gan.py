import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from tqdm.autonotebook import tqdm
from IPython.display import display
import pandas as pd

import ipywidgets as widgets
from tqdm.notebook import tqdm

TRAIN_BUF = 60000
BATCH_SIZE = 512
TEST_BUF = 10000
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)
N_Z = 64

class GAN(tf.keras.Model):
    """ a basic GAN class
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(GAN, self).__init__()
        self.__dict__.update(kwargs)

        self.gen = self.gen
        self.disc = self.disc

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def compute_loss(self, x):
        """ passes through the network and computes loss
        """
        # generating noise from a uniform distribution
        z_samp = tf.random.normal([x.shape[0], self.n_Z])

        # run noise through generator
        x_gen = self.generate(z_samp)
        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)
        ### losses
        # losses of real with label "1"
        disc_real_loss = gan_loss(logits=logits_x, is_real=True)
        # losses of fake with label "0"
        disc_fake_loss = gan_loss(logits=logits_x_gen, is_real=False)
        disc_loss = disc_fake_loss + disc_real_loss

        # losses of fake with label "1"
        gen_loss = gan_loss(logits=logits_x_gen, is_real=True)

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):

        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )
    @tf.function
    def train(self, train_x):
        gen_gradients, disc_gradients = self.compute_gradients(train_x)
        self.apply_gradients(gen_gradients, disc_gradients)


def gan_loss(logits, is_real=True):
    """Computes standard gan loss between logits and labels
    """
    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits
    )


import io


# exampled data for plotting results
def plot_reconstruction(model, nex=8, zm=2, seed=42):
    samples = model.generate(tf.random.normal(shape=(BATCH_SIZE, N_Z), seed=seed))
    fig, axs = plt.subplots(ncols=nex, nrows=1, figsize=(zm * nex, zm))
    for axi in range(nex):
        axs[axi].matshow(
            samples.numpy()[axi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
        )
        axs[axi].axis('off')

    # Save the figure as a PNG byte stream
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)  # Close the figure after saving
    buf.seek(0)  # Go to the beginning of the byte stream

    return buf.read()  # Return the PNG byte data


def get_models():
    """
    Use functional style here as otherwise input layer won't show up in model.summary()...
    """
    input = tf.keras.layers.Input(shape=(N_Z, ))
    x = input
    for l in [
        tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ),
    ]:
        x = l(x)

    generator = Model(inputs=input, outputs=x)

    input = tf.keras.layers.Input(shape=DIMS)
    x = input
    for l in [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1, activation=None),
    ]:
        x = l(x)
    discriminator = Model(inputs=input, outputs=x)

    return generator, discriminator

def plot_gan(generator, discriminator, N_Z=64):
    # load dataset
    (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

    # split dataset
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    ) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

    # batch datasets
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(TRAIN_BUF)
        .batch(BATCH_SIZE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(TEST_BUF)
        .batch(BATCH_SIZE)
    )

    # optimizers
    gen_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.005)  # train the model
    # model
    model = GAN(
        gen=generator,
        disc=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        n_Z=N_Z
    )

    losses = pd.DataFrame(columns = ['disc_loss', 'gen_loss'])


    # Initialize storage for images and slider widget
    images = []
    slider = widgets.IntSlider(value=0, min=0, max=50, step=1, description='Epoch:')
    image_widget = widgets.Image(
        width=1000,
        height=200,
    )
    train_log = widgets.Output()

    # Function to update the displayed image based on the slider value
    def update_image(change):
        epoch = change['new']
        image_widget.value = images[epoch]

    # Attach the update function to the slider
    slider.observe(update_image, names='value')

    def add_to_slider(image):
        images.append(image)

        slider.max = len(images)-1
        slider.value = len(images)-1

    # Display the slider and image widget together
    display(widgets.VBox([slider, image_widget, train_log]))

    image_array = plot_reconstruction(model)
    add_to_slider(image_array)

    n_epochs = 50
    slider.disabled = True  # Slider doesn't work during training => Disable it to prevent confusion
    for epoch in range(n_epochs):
        with train_log:
            # train
            for batch, train_x in tqdm(
                zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
            ):
                model.train(train_x)
            # test on holdout
            loss = []
            for batch, test_x in tqdm(
                zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
            ):
                loss.append(model.compute_loss(train_x))
            losses.loc[len(losses)] = np.mean(loss, axis=0)
            print(
                "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
                    epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
                )
            )

        # Generate plot for this epoch
        image_array = plot_reconstruction(model)
        add_to_slider(image_array)

    slider.disabled = False

    return losses


def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses.index, losses['disc_loss'], label='Discriminator Loss')
    plt.plot(losses.index, losses['gen_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses Over Epochs')
    plt.legend()
    plt.show()