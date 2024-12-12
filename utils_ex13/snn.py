import matplotlib.pyplot as plt
from utils_ex13.ann_data import plot_data as plot_data_ann
from utils_ex13.ann_data import get_dataloader as get_dataloader_ann
from utils_ex13.snn_data import get_dataloader

import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)


def step_fn(x):
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 10.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[torch.Tensor(input > 0)] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
surrogate_fn = SurrGradSpike.apply

import torch.nn as nn


class SNNRecurrentLayer(nn.Module):
    def __init__(self, input_size, output_size, alpha, beta, spike_fn=surrogate_fn):
        super(SNNRecurrentLayer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.spike_fn = spike_fn

        # Linear layer for synaptic weights
        self.linear = nn.Linear(input_size, output_size, bias=False)
        # torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)

    def forward(self, inputs, nb_steps: int):
        batch_size = inputs.size(0)
        output_size = self.linear.weight.size(0)

        # Initialize states
        syn = torch.zeros((batch_size, output_size), dtype=torch.float32, device=inputs.device)
        mem = torch.zeros((batch_size, output_size), dtype=torch.float32, device=inputs.device)

        mem_rec = []
        spk_rec = []

        # calculate all linear transformations in one go
        h = self.linear(inputs.reshape(-1, inputs.size(-1))).reshape(batch_size, nb_steps, -1)

        # Iterate over time steps
        for t in range(int(nb_steps)):
            # Spiking neuron update
            mthr = mem - 1.0
            out = self.spike_fn(mthr)
            rst = out.detach()  # No gradient through reset

            # Update synaptic current and membrane potential
            new_syn = self.alpha * syn + h[:, t]
            new_mem = (self.beta * mem + syn) * (1.0 - rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        return spk_rec


class SNNReadoutLayer(nn.Module):
    def __init__(self, input_size, output_size, alpha, beta):
        super(SNNReadoutLayer, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # Linear layer for readout weights
        self.linear = nn.Linear(input_size, output_size, bias=False)
        # torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)

    def forward(self, inputs, nb_steps: int):
        batch_size = inputs.size(0)
        output_size = self.linear.weight.size(0)

        # Initialize states
        flt = torch.zeros((batch_size, output_size), dtype=torch.float32, device=inputs.device)
        out = torch.zeros((batch_size, output_size), dtype=torch.float32, device=inputs.device)

        out_rec = []

        # calculate all linear transformations in one go
        h = self.linear(inputs.reshape(-1, inputs.size(-1))).reshape(batch_size, nb_steps, -1)

        # Iterate over time steps
        for t in range(int(nb_steps)):
            # Low-pass filtering for readout dynamics
            new_flt = self.alpha * flt + h[:, t]
            new_out = self.beta * out + flt

            # Record output
            out_rec.append(new_out)

            # Update states
            flt = new_flt
            out = new_out

        # Stack output recordings
        out_rec = torch.stack(out_rec, dim=1)

        return out_rec


class SNN(nn.Module):
    def __init__(self, network_layout, freeze_first_layer=False, spike_fn=surrogate_fn):
        super(SNN, self).__init__()
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = len(network_layout['layer_sizes'])
        self.layer_sizes = network_layout['layer_sizes']
        self.spike_fn = spike_fn

        tau_mem = 10e-3
        tau_syn = 5e-3
        time_step = 1e-3

        alpha = float(np.exp(-time_step / tau_syn))
        beta = float(np.exp(-time_step / tau_mem))

        self.layers = torch.nn.ModuleList()
        layer = SNNRecurrentLayer(self.n_inputs, self.layer_sizes[0], alpha, beta, spike_fn=self.spike_fn)
        self.layers.append(layer)

        for i in range(self.n_layers - 2):
            layer = SNNRecurrentLayer(self.layer_sizes[i], self.layer_sizes[i + 1], alpha, beta, spike_fn=self.spike_fn)
            self.layers.append(layer)

        readout_layer = SNNReadoutLayer(self.layer_sizes[-2], self.layer_sizes[-1], alpha, beta)
        self.layers.append(readout_layer)

        if freeze_first_layer:
            for param in self.layers[0].parameters():
                param.requires_grad = False

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x, nb_steps=x.shape[1])
        return x


class Trainer:

    def __init__(
            self,
            network,
            train_loader,
            valid_loader,
            test_loader,
            n_epochs=300,
            learning_rate = 1e-2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.network = network
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.device = device

        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.AdamW(self.network .parameters(), lr=learning_rate, weight_decay=1e-8)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20)
        self.log_softmax_fn = nn.LogSoftmax(dim=1)

        self.network.to(self.device)

    def training_epoch(self):
        num_correct = 0
        num_shown = 0
        self.network.train()

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device).float()
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.network(x)
            y_pred_agg, _ = y_pred.max(dim=1)
            log_y_pred = self.log_softmax_fn(y_pred_agg)
            loss = self.criterion(log_y_pred, y)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            loss.backward()
            self.optimizer.step()

            predicted = y_pred_agg.argmax(dim=1)
            num_correct += (predicted == y).sum().item()
            num_shown += len(y)

        accuracy = float(num_correct) / num_shown
        return accuracy

    def validation_epoch(self, loader):
        predictions = []
        num_correct = 0
        num_shown = 0
        self.network.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device).float()
                y = y.to(self.device)
                y_pred = self.network(x)
                y_pred_agg, _ = y_pred.max(dim=1)
                predicted = y_pred_agg.argmax(dim=1)
                num_correct += (predicted == y).sum().item()
                num_shown += len(y)

                predictions.extend(predicted.cpu().numpy().tolist())

        accuracy = float(num_correct) / num_shown
        return accuracy, predictions


    def train(self):
        train_accs = []
        valid_accs = []
        for epoch in range(self.n_epochs):
            train_acc = self.training_epoch()
            valid_acc, _ = self.validation_epoch(self.valid_loader)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            if (epoch+1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs} - Training accuracy: {train_acc:.4f} - Validation accuracy: {valid_acc:.4f}')
            self.scheduler.step(train_acc)
        return train_accs, valid_accs

    def test(self):
        test_acc, _ = self.validation_epoch(self.test_loader)
        print(f'Test accuracy: {test_acc:.4f}')
        return test_acc

    def plot_training_accuracies(self, train_accuracies, val_accuracies, test_accuracies):
        plt.figure(figsize=(10, 8))
        plt.plot(train_accuracies, label='train acc')
        plt.plot(val_accuracies, label='val acc')
        plt.axhline(test_accuracies, ls='--', color='grey', label='test acc')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.ylim(0.3, 1.05)
        plt.grid()
        plt.legend()

    def visualize_predictions(self):
        # create dataloaders that are not shuffling the data
        train_loader, val_loader, test_loader = get_dataloader(shuffle_train=False)

        train_predictions = self.validation_epoch(train_loader)[1]
        val_predictions = self.validation_epoch(val_loader)[1]
        test_predictions = self.validation_epoch(test_loader)[1]

        ann_train_loader, ann_val_loader, ann_test_loader = get_dataloader_ann()
        plot_data_ann(ann_train_loader, ann_val_loader, ann_test_loader, train_predictions, val_predictions, test_predictions)