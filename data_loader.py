import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch

# ----------------- Dataset Loader ----------------- #
def get_dataset(args):
    """
    Prepare federated MNIST datasets with IID or non-IID splits.
    Returns:
        train_subsets: list of Subset objects for each client
        global_test_dataset: full MNIST test dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Split training dataset for each client using IID or non-IID split
    if args.iid == 1:
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        user_groups = mnist_noniid(train_dataset, args.num_users)

    # Instead of DataLoaders, return dataset subsets
    train_subsets = []
    for i in range(args.num_users):
        indices = user_groups[i]
        train_subset = Subset(train_dataset, indices)
        train_subsets.append(train_subset)

    # Return subsets (datasets) instead of loaders
    return train_subsets, test_dataset


# ----------------- IID Split ----------------- #
def mnist_iid(dataset, num_users):
    """Split dataset into IID partitions for each client."""
    num_items = len(dataset) // num_users
    dict_users, all_idxs = {}, np.arange(len(dataset))
    np.random.seed(2021)
    for i in range(num_users):
        select_idxs = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = select_idxs
        all_idxs = np.setdiff1d(all_idxs, select_idxs)
    return dict_users


# ----------------- Non-IID Split (Safe) ----------------- #
def mnist_noniid(dataset, num_users, shard_per_user=3):
    """
    Split dataset into non-IID subsets for each client using shards per class.
    """
    np.random.seed(2022)
    num_classes = len(np.unique(dataset.targets))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Group indices by class and shuffle
    idxs_dict = {}
    for i in range(len(dataset)):
        label = int(dataset.targets[i])
        idxs_dict.setdefault(label, []).append(i)
    for k in idxs_dict:
        np.random.shuffle(idxs_dict[k])

    # Compute shards per class
    num_shards_total = num_users * shard_per_user
    imgs_per_shard = len(dataset) // num_shards_total

    # Assign shards to users
    for user in range(num_users):
        selected_labels = np.random.choice(range(num_classes), shard_per_user, replace=False)
        shards = []
        for label in selected_labels:
            available = len(idxs_dict[label])
            n = min(imgs_per_shard, available)
            if n == 0:
                continue
            shard_idxs = idxs_dict[label][:n]
            shards.append(shard_idxs)
            idxs_dict[label] = idxs_dict[label][n:]  # remove assigned
        dict_users[user] = np.concatenate(shards)

    # Sanity check: each user should have at least one sample per shard
    for key, value in dict_users.items():
        if len(value) == 0:
            raise ValueError(f"User {key} received no samples. Reduce shard_per_user or num_users.")

    return dict_users




