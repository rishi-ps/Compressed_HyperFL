import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Federated Learning with FedAvg / HyperFL / HyperFL-Sparse")

    # ---------- Federated learning settings ----------
    parser.add_argument('--epochs', type=int, default=5,
                        help="number of global communication rounds")
    parser.add_argument('--num_users', type=int, default=3,
                        help="number of clients participating in FL")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='fraction of clients participating per round (1.0 = all)')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help="number of local epochs per client per round")
    parser.add_argument('--local_iter', type=int, default=1,
                        help="alternative to local_epoch (used if local_epoch=0)")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size per client")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for main model / hypernetwork')
    parser.add_argument('--lr_g', type=float, default=0.05,
                        help='learning rate for classifier/global model')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum for SGD optimizer')

    # ---------- Training method ----------
    parser.add_argument('--train_rule', type=str, default='FedAvg',
                        choices=['FedAvg', 'HyperFL', 'HyperFL_Sparse'],
                        help='federated learning rule')

    # ---------- HyperFL model hyperparameters ----------
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='dimension of client embedding (smaller for faster runs)')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='hidden dimension of hypernetwork (smaller for faster runs)')

    # ---------- Sparsification ----------
    parser.add_argument('--sparsify_frac', type=float, default=0.1,
                        help='fraction of gradients to KEEP during sparsification (e.g. 0.1 = top 10%)')

    # ---------- Dataset ----------
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="dataset name (MNIST supported)")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")

    # ---------- Data partitioning ----------
    parser.add_argument('--iid', type=int, default=1,
                        help='set 1 for IID, 0 for non-IID split')

    # ---------- Device ----------
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use (ignored if no CUDA)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device string: cuda or cpu')

    args = parser.parse_args()
    return args
