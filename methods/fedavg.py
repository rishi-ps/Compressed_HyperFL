import copy
import torch
from torch import nn

class LocalUpdate_FedAvg:
    def __init__(self, idx, args, train_loader, test_loader, model):
        """
        Local client for FedAvg training
        """
        self.idx = idx
        self.args = args
        self.train_loader = train_loader  # DataLoader for this client's train data
        self.test_loader = test_loader    # DataLoader for evaluation (full test set)
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = copy.deepcopy(model)

        # Initialize aggregation weight as the number of samples for this client
        self.agg_weight = torch.tensor(len(train_loader.dataset), dtype=torch.float32, device=self.device)

        # Optional: keys for classifier fine-tuning
        self.w_local_keys = getattr(model, 'classifier_weight_keys', None)

    def local_test(self, test_loader=None, test_model=None):
        test_loader = self.test_loader if test_loader is None else test_loader
        model = self.local_model if test_model is None else test_model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        return acc

    def update_local_model(self, global_weight):
        self.local_model.load_state_dict(global_weight)

    def local_training(self, local_epoch=None, local_iter=None):
        """
        Standard FedAvg local training.
        Removed 'round' argument to match calling code.
        Returns:
            model.state_dict(), loss_start, loss_end, acc_before, acc_after
        """
        model = self.local_model
        model.train()
        local_epoch = self.args.local_epoch if local_epoch is None else local_epoch
        local_iter = self.args.local_iter if local_iter is None else local_iter

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=self.args.lr,
                                    momentum=0.5,
                                    weight_decay=5e-4)

        acc_before = self.local_test()
        iter_loss = []

        if local_epoch > 0:
            for ep in range(local_epoch):
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    features, outputs = model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
        else:
            data_iter = iter(self.train_loader)
            for it in range(local_iter):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    images, labels = next(data_iter)
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                features, outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())

        loss_start = iter_loss[0] if iter_loss else 0.0
        loss_end = iter_loss[-1] if iter_loss else 0.0
        acc_after = self.local_test()

        return model.state_dict(), loss_start, loss_end, acc_before, acc_after

    

    