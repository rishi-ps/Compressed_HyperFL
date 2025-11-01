import torch
import copy
import numpy as np
from tools import average_weights_weighted

# ----------------- Dispatcher ----------------- #
def one_round_training(rule):
    """
    Returns the function to execute one federated training round
    """
    Train_Round = {
        'FedAvg': train_round_fedavg,
        'HyperFL': train_round_hyperfl,
        'HyperFL_Sparse': train_round_hyperfl_sparse
    }
    if rule not in Train_Round:
        raise NotImplementedError(f"Training rule '{rule}' not implemented.")
    return Train_Round[rule]

# ----------------- FedAvg Training ----------------- #
def train_round_fedavg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    idx_users = sorted(np.random.choice(range(num_users), m, replace=False))

    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1, local_acc2 = [], []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        client = local_clients[idx]
        agg_weight.append(client.agg_weight)            # client data size
        client.update_local_model(global_weight)        # sync global weights
        w, loss1, loss2, acc1, acc2 = client.local_training(local_epoch=args.local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # Aggregate global weights
    global_weight_new = average_weights_weighted(local_weights, torch.stack(agg_weight).to(args.device))
    global_model.load_state_dict(global_weight_new)

    # Compute average metrics
    loss_avg1 = np.mean(local_losses1)
    loss_avg2 = np.mean(local_losses2)
    acc_avg1 = np.mean(local_acc1)
    acc_avg2 = np.mean(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

# ----------------- HyperFL Training ----------------- #
def train_round_hyperfl(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- HyperFL Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    idx_users = sorted(np.random.choice(range(num_users), m, replace=False))

    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1, local_acc2 = [], []
    agg_weight = []

    # Each client performs local training
    for idx in idx_users:
        client = local_clients[idx]
        w, loss1, loss2, acc1, acc2 = client.local_training(local_epoch=args.local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
        agg_weight.append(client.agg_weight)

    # Aggregate global weights via hypernetwork
    agg_weight_tensor = torch.stack(agg_weight).to(args.device)
    global_weight_new = average_weights_weighted(local_weights, agg_weight_tensor)

    # Update hypernetwork in all clients
    for client in local_clients:
        if hasattr(client, 'update_hypernetwork'):
            client.update_hypernetwork(global_weight=global_weight_new)

    # Compute average metrics
    loss_avg1 = np.mean(local_losses1)
    loss_avg2 = np.mean(local_losses2)
    acc_avg1 = np.mean(local_acc1)
    acc_avg2 = np.mean(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

# ----------------- HyperFL_Sparse Training ----------------- #
def train_round_hyperfl_sparse(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- HyperFL_Sparse Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    idx_users = sorted(np.random.choice(range(num_users), m, replace=False))

    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1, local_acc2 = [], []
    agg_weight = []
    local_sparsities = []

    # Each client performs local training (sparse)
    for idx in idx_users:
        client = local_clients[idx]
        # Expect sparse local_training to return an extra sparsity value
        result = client.local_training(local_epoch=args.local_epoch)
        # result can be (w, loss1, loss2, acc1, acc2, sparsity) for sparse clients
        if len(result) == 6:
            w, loss1, loss2, acc1, acc2, sparsity = result
        else:
            # fallback: if client is a regular hyperfl return, assume no sparsity
            w, loss1, loss2, acc1, acc2 = result
            sparsity = 1.0
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
        agg_weight.append(client.agg_weight)
        local_sparsities.append(sparsity)

    # Aggregate global weights via hypernetwork
    agg_weight_tensor = torch.stack(agg_weight).to(args.device)
    global_weight_new = average_weights_weighted(local_weights, agg_weight_tensor)

    # Update hypernetwork in all clients
    for client in local_clients:
        if hasattr(client, 'update_hypernetwork'):
            client.update_hypernetwork(global_weight=global_weight_new)

    # Compute average metrics
    loss_avg1 = np.mean(local_losses1)
    loss_avg2 = np.mean(local_losses2)
    acc_avg1 = np.mean(local_acc1)
    acc_avg2 = np.mean(local_acc2)
    sparsity_avg = np.mean(local_sparsities) if local_sparsities else 1.0

    # Return sparsity as well (placed in 5th position to avoid breaking callers expecting 4 outputs)
    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, sparsity_avg
