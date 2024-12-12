import matplotlib.pyplot as plt
from utils_ex13.ann_data import get_dataloader, plot_data

import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

class ANN(torch.nn.Module):
    def __init__(self, network_layout, freeze_first_layer=False):
        super(ANN, self).__init__()
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = len(network_layout['layer_sizes'])
        self.layer_sizes = network_layout['layer_sizes']

        self.layers = torch.nn.ModuleList()
        layer = torch.nn.Linear(self.n_inputs, self.layer_sizes[0], bias=True)
        self.layers.append(layer)

        for i in range(self.n_layers-1):
            layer = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1], bias=True)
            self.layers.append(layer)

        if freeze_first_layer:
            for param in self.layers[0].parameters():
                param.requires_grad = False


    def forward(self, x):
        x_hidden = []
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if not i == (self.n_layers-1):
                x = torch.nn.functional.relu(x)
                x_hidden.append(x)
        return x


class Trainer:

    def __init__(
            self,
            network,
            train_loader,
            valid_loader,
            test_loader,
            n_epochs=300,
            learning_rate = 0.05,
            device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.network = network
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.device = device

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.network .parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20)

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
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            winner = y_pred.argmax(1)
            num_correct += len(y_pred[winner == y])
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
                winner = y_pred.argmax(1)
                num_correct += len(y_pred[winner == y])
                num_shown += len(y)

                predictions.extend(y_pred.argmax(1).cpu().numpy().tolist())

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
            if (epoch+1) % 25 == 0:
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

        plot_data(self.train_loader, self.valid_loader, self.test_loader, train_predictions, val_predictions, test_predictions)