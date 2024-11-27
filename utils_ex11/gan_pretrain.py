import numpy as np
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import nltk
from nltk.corpus import wordnet, words
import random

from ipywidgets import Text, Output, widgets

from utils_ex11.imagenet_classes import IMAGENET_CLASSES


logging.basicConfig(level=logging.ERROR)


def gan_pretrain():

    nltk.download('wordnet')
    nltk.download('words')

    text = Text(value='soap bubble')
    random_btn = widgets.Button(description='Generate random word')
    submit = widgets.Button(description='Generate')
    output_plot = Output()

    def update():
        try:
            output_plot.clear_output(wait=False)

            text_value = text.value

            if torch.backends.mps.is_available():
                DEVICE = 'mps'
            elif torch.cuda.is_available():
                DEVICE = 'cuda'
            else:
                DEVICE = 'cpu'

            # Load pre-trained model tokenizer (vocabulary)
            model = BigGAN.from_pretrained('biggan-deep-256')

            # Prepare a input
            truncation = 0.4
            class_vector = one_hot_from_names([text_value], batch_size=1)
            noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)

            # All in tensors
            noise_vector = torch.from_numpy(noise_vector)
            class_vector = torch.from_numpy(class_vector)

            # If you have a GPU, put everything on cuda
            noise_vector = noise_vector.to(DEVICE)
            class_vector = class_vector.to(DEVICE)
            model.to(DEVICE)

            # Generate an image
            with torch.no_grad():
                output = model(noise_vector, class_vector, truncation)

            # If you have a GPU put back on CPU
            output = output.to('cpu')

            # If you have a sixtel compatible terminal you can display the images in the terminal
            # (see https://github.com/saitoha/libsixel for details)
            output = output.numpy()
            output = np.moveaxis(output, 1, -1)
            assert output.shape == (1, 256, 256, 3)
            with output_plot:
                img = output[0]
                plt.imshow(img)
                plt.axis('off')
                plt.show()
        except Exception as e:
            with output_plot:
                print("FAILED", e)

    def random_word():
        imagenet_classes = IMAGENET_CLASSES
        random_label = random.choice(imagenet_classes)
        text.value = random_label
        update()

    submit.on_click(lambda x: update())
    random_btn.on_click(lambda x: random_word())
    random_word()
    display(widgets.VBox([text, random_btn, submit, output_plot]))
