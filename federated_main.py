import numpy as np
import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import Subset, DataLoader

from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CNN_MNIST, CNN_MNIST_Hyper
from options import args_parser

torch.set_num_threads(4)

if __name__ == '__main__':
    # ----------------- Args and device ----------------- #
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # ----------------- Reproducibility ----------------- #
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args.dataset = 'mnist'
    args.num_classes = 10

    # ----------------- Dataset ----------------- #
    train_datasets, global_test_dataset = get_dataset(args)

    # ----------------- Global model ----------------- #
    if args.train_rule in ['HyperFL', 'HyperFL_Sparse']:
        global_model = CNN_MNIST_Hyper(args=args).to(device)
    elif args.train_rule == 'FedAvg':
        global_model = CNN_MNIST(num_classes=args.num_classes).to(device)
    else:
        raise NotImplementedError("Only FedAvg, HyperFL and HyperFL_Sparse are supported for MNIST.")

    print(f"\n=== Training {args.train_rule} on MNIST ===")
    print(args)

    # ----------------- LocalUpdate / One-round functions ----------------- #
    LocalUpdate = local_update(args.train_rule)            # returns the class
    train_round_parallel = one_round_training(args.train_rule)

    train_loss, local_accs1, local_accs2 = [], [], []
    local_clients = []
    sparsity_per_round = []

    # ----------------- Initialize local clients ----------------- #
    for idx in range(args.num_users):
        train_set_i = train_datasets[idx]
        test_set_i = global_test_dataset

        if args.train_rule in ['HyperFL', 'HyperFL_Sparse']:
            # HyperFL uses datasets (creates loaders internally)
            local_clients.append(
                LocalUpdate(
                    idx=idx,
                    args=args,
                    train_set=train_set_i,
                    test_set=test_set_i,
                    model=copy.deepcopy(global_model)
                )
            )
        else:
            # FedAvg uses explicit DataLoaders
            train_loader = DataLoader(train_set_i, batch_size=args.local_bs, shuffle=True)
            test_loader = DataLoader(test_set_i, batch_size=args.local_bs, shuffle=False)

            local_clients.append(
                LocalUpdate(
                    idx=idx,
                    args=args,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model=copy.deepcopy(global_model)
                )
            )

    # ----------------- Federated training ----------------- #
    for rnd in range(args.epochs):
        print(f"\n========== Round {rnd + 1} / {args.epochs} ==========")
        results = train_round_parallel(
            args, global_model, local_clients, rnd
        )

        # For HyperFL_Sparse, running.train_round returns 5 values including sparsity
        if args.train_rule == 'HyperFL_Sparse':
            loss1, loss2, local_acc1, local_acc2, sparsity_avg = results
            sparsity_per_round.append(sparsity_avg)
        else:
            loss1, loss2, local_acc1, local_acc2 = results

        train_loss.append(loss1)
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)

        if args.train_rule == 'HyperFL_Sparse':
            print(f"Train Loss: {loss1:.4f}, {loss2:.4f} | Avg Sparsity kept: {sparsity_avg*100:.2f}%")
        else:
            print(f"Train Loss: {loss1:.4f}, {loss2:.4f}")
        print(f"Local Accuracy: {local_acc1:.2f}%, {local_acc2:.2f}%")

    print("\n=== Training Complete ===")
    
    # Optionally save sparsity_per_round for plotting later
    if sparsity_per_round:
        np.save('sparsity_per_round.npy', np.array(sparsity_per_round))

        # ----------------- Save results for plotting ----------------- #
    np.save(f"{args.train_rule}_train_loss.npy", np.array(train_loss))
    np.save(f"{args.train_rule}_accs2.npy", np.array(local_accs2))
    if args.train_rule == 'HyperFL_Sparse':
        np.save(f"{args.train_rule}_sparsity.npy", np.array(sparsity_per_round))

