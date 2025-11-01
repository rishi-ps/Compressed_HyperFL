# methods/hyperfl_sparse.py
from copy import deepcopy
import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import the original HyperFL class to reuse utilities if needed
from .hyperfl import LocalUpdate_HyperFL

class LocalUpdate_HyperFL_Sparse(LocalUpdate_HyperFL):
    """
    LocalUpdate for HyperFL + Top-k gradient sparsification.
    Inherits from LocalUpdate_HyperFL and overrides the hypernetwork update step
    to sparsify gradients prior to applying them.
    """

    def __init__(self, idx, args, train_set, test_set, model):
        super().__init__(idx, args, train_set, test_set, model)
        # top_k fraction (keep fraction) — configurable via args. Default 0.1 (keep 10%).
        self.top_k_frac = getattr(self.args, 'sparsify_frac', 0.1)

    @staticmethod
    def _sparsify_tensor_grad(grad, keep_frac):
        """
        Keep top (by absolute value) fraction of elements in grad; zero out rest.
        grad: torch.Tensor (same shape as parameter.grad)
        return: sparsified grad (new tensor)
        """
        if grad is None:
            return None
        numel = grad.numel()
        if numel == 0:
            return grad
        k = max(1, int(math.ceil(keep_frac * numel)))
        # Flatten and get threshold as kth largest absolute value
        flat = grad.view(-1).abs()
        if k >= numel:
            return grad
        # torch.kthvalue finds the k-th smallest — for largest, index = numel-k+1
        # but simpler: use topk
        vals, _ = torch.topk(flat, k, largest=True, sorted=False)
        threshold = vals.min()
        # build mask
        mask = (grad.abs() >= threshold).to(dtype=grad.dtype)
        return grad * mask

    def local_training(self, local_epoch, round=0):
        """
        Override to sparsify hypernetwork and client_embedding gradients just before optimizer.step()
        Returns hypernetwork.state_dict(), round_loss1, round_loss2, acc0, acc2, sparsity_fraction
        sparsity_fraction: fraction of *kept* elements averaged across hypernetwork+embedding grads
        """
        model = self.local_model
        model.train()
        round_loss = []
        iter_loss = []
        model.zero_grad()

        # Evaluate before training
        acc0, _ = self.local_test(self.test_data)

        local_ep_rep = local_epoch
        epoch_classifier = 1

        # ---------- Update classifier (same as original HyperFL) ----------
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

        # ---------- Update hypernetwork (with sparsified grads) ----------
        optimizer = torch.optim.SGD(
            [{'params': model.hypernetwork.parameters()},
             {'params': model.client_embedding.parameters()}],
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=0.0005
        )

        # keep track of total elements and kept elements to compute sparsity
        total_elems = 0
        kept_elems = 0

        for ep in range(local_ep_rep):
            for images, labels in train_loader:
                optimizer.zero_grad()
                model.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                _, output = model(images)
                loss = self.criterion(output, labels)

                # generate target_model weights (these are used to compute grads via autograd)
                weights = model.generate_weight()

                # collect parameters to compute grads for (exclude fc2)
                params_to_grads = []
                for name, param in model.target_model.named_parameters():
                    if 'fc2' not in name:
                        params_to_grads.append(param)

                # compute grads of loss wrt params_to_grads
                grads = torch.autograd.grad(loss, params_to_grads, retain_graph=True, create_graph=False)

                # compute grads wrt hypernetwork and client_embedding
                hypernetwork_grads = torch.autograd.grad(
                    list(weights.values()), model.hypernetwork.parameters(),
                    grad_outputs=list(grads), retain_graph=True, create_graph=False
                )
                client_embedding_grads = torch.autograd.grad(
                    list(weights.values()), model.client_embedding.parameters(),
                    grad_outputs=list(grads), retain_graph=True, create_graph=False
                )

                # sparsify each gradient tensor by top-k fraction
                sparsified_hyper_grads = []
                for g in hypernetwork_grads:
                    if g is None:
                        sparsified_hyper_grads.append(None)
                        continue
                    s = self._sparsify_tensor_grad(g, self.top_k_frac)
                    sparsified_hyper_grads.append(s)
                    total_elems += g.numel()
                    kept_elems += (s.abs() > 0).sum().item()

                sparsified_emb_grads = []
                for g in client_embedding_grads:
                    if g is None:
                        sparsified_emb_grads.append(None)
                        continue
                    s = self._sparsify_tensor_grad(g, self.top_k_frac)
                    sparsified_emb_grads.append(s)
                    total_elems += g.numel()
                    kept_elems += (s.abs() > 0).sum().item()

                # set gradients on model.hypernetwork and model.client_embedding
                for p, g in zip(model.hypernetwork.parameters(), sparsified_hyper_grads):
                    if g is None:
                        p.grad = None
                    else:
                        p.grad = g.clone().detach()

                for p, g in zip(model.client_embedding.parameters(), sparsified_emb_grads):
                    if g is None:
                        p.grad = None
                    else:
                        p.grad = g.clone().detach()

                # gradient clipping as original
                torch.nn.utils.clip_grad_norm_(model.hypernetwork.parameters(), 50)
                torch.nn.utils.clip_grad_norm_(model.client_embedding.parameters(), 50)
                optimizer.step()
                iter_loss.append(loss.item())

            round_loss.append(sum(iter_loss) / len(iter_loss) if iter_loss else 0.0)
            iter_loss = []

        # compute sparsity fraction (kept / total)
        sparsity_fraction = (kept_elems / total_elems) if total_elems > 0 else 1.0

        round_loss1 = round_loss[0] if round_loss else 0.0
        round_loss2 = round_loss[-1] if round_loss else 0.0
        acc2, _ = self.local_test(self.test_data)

        # Return hypernetwork state_dict (same as HyperFL), plus sparsity
        return model.hypernetwork.state_dict(), round_loss1, round_loss2, acc0, acc2, sparsity_fraction
