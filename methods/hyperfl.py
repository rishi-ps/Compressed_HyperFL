from copy import deepcopy
import copy
import numpy as np
from numpy import random
from numpy.core.shape_base import stack
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import time


# ---------------------------------------------------------------------------- #

class LocalUpdate_HyperFL(object):
    def __init__(self, idx, args, train_set, test_set, model):
        self.idx = idx
        self.args = args
        self.num_classes = args.num_classes
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.agg_weight = self.aggregate_weight()

    def aggregate_weight(self):
        data_size = len(self.train_data)
        w = torch.tensor(data_size).to(self.device)
        return w

    def local_test(self, test_loader):
        model = self.local_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader)
        loss_test = []
        with torch.no_grad():
            for inputs, labels in DataLoader(test_loader, batch_size=self.args.local_bs, shuffle=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.predict(inputs)
                loss = self.criterion(outputs, labels)
                loss_test.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        return acc, sum(loss_test) / len(loss_test)

    def update_hypernetwork(self, global_weight):
        self.local_model.hypernetwork.load_state_dict(global_weight)

    def local_training(self, local_epoch, round=0):
        model = self.local_model
        model.train()
        round_loss = []
        iter_loss = []
        model.zero_grad()

        # Evaluate before training
        acc0, _ = self.local_test(self.test_data)

        local_ep_rep = local_epoch
        epoch_classifier = 1

        # ---------- Update classifier ----------
        optimizer = torch.optim.SGD(
            model.target_model.fc2.parameters(),
            lr=self.args.lr_g,
            momentum=0.5,
            weight_decay=0.0005
        )

        train_loader = DataLoader(
            self.train_data, batch_size=self.args.local_bs, shuffle=True
        )

        for ep in range(epoch_classifier):
            for images, labels in train_loader:
                optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                _, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            round_loss.append(sum(iter_loss) / len(iter_loss))
            iter_loss = []

        acc1, _ = self.local_test(self.test_data)

        # ---------- Update hypernetwork ----------
        optimizer = torch.optim.SGD(
            [{'params': model.hypernetwork.parameters()},
             {'params': model.client_embedding.parameters()}],
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=0.0005
        )

        for ep in range(local_ep_rep):
            for images, labels in train_loader:
                optimizer.zero_grad()
                model.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                _, output = model(images)
                loss = self.criterion(output, labels)

                weights = model.generate_weight()
                params_to_grads = []
                for name, param in model.target_model.named_parameters():
                    if 'fc2' not in name:
                        params_to_grads.append(param)
                grads = torch.autograd.grad(loss, params_to_grads)

                hypernetwork_grads = torch.autograd.grad(
                    list(weights.values()), model.hypernetwork.parameters(),
                    grad_outputs=list(grads), retain_graph=True
                )
                client_embedding_grads = torch.autograd.grad(
                    list(weights.values()), model.client_embedding.parameters(),
                    grad_outputs=list(grads), retain_graph=True
                )

                for p, g in zip(model.hypernetwork.parameters(), hypernetwork_grads):
                    p.grad = g
                for p, g in zip(model.client_embedding.parameters(), client_embedding_grads):
                    p.grad = g

                torch.nn.utils.clip_grad_norm_(model.hypernetwork.parameters(), 50)
                torch.nn.utils.clip_grad_norm_(model.client_embedding.parameters(), 50)
                optimizer.step()
                iter_loss.append(loss.item())

            round_loss.append(sum(iter_loss) / len(iter_loss))
            iter_loss = []

        round_loss1 = round_loss[0]
        round_loss2 = round_loss[-1]
        acc2, _ = self.local_test(self.test_data)

        return model.hypernetwork.state_dict(), round_loss1, round_loss2, acc0, acc2
