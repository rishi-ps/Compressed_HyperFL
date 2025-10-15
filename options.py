import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # ---------- Federated learning settings ----------
    parser.add_argument('--epochs', type=int, default=5,  # fewer federated rounds
                        help="number of training epochs")
    parser.add_argument('--num_users', type=int, default=5,  # fewer clients
                        help="number of users/clients")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='fraction of clients participating per round')
    parser.add_argument('--local_epoch', type=int, default=1,  # fewer local epochs
                        help="number of local epochs per client")
    parser.add_argument('--local_iter', type=int, default=1,
                        help="number of local iterations")
    parser.add_argument('--local_bs', type=int, default=32,  # smaller batch size
                        help="local batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for local models')
    parser.add_argument('--lr_g', type=float, default=0.05,
                        help='learning rate for classifier/global model')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='optimizer for local training')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum for SGD')
    
    # ---------- Training method ----------
    parser.add_argument('--train_rule', type=str, default='FedAvg',
                        choices=['FedAvg', 'HyperFL'],
                        help='training rule for personalized FL')
    
    # ---------- Client model hyperparameters ----------
    parser.add_argument('--local_size', type=int, default=600,  # smaller local dataset
                        help='number of samples per client')
    parser.add_argument('--embed_dim', type=int, default=32,  # smaller embedding
                        help='dimension of client embedding')
    parser.add_argument('--hidden_dim', type=int, default=64,  # smaller hidden dimension
                        help='hidden dimension of hypernetwork')
    
    # ---------- Dataset ----------
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="dataset name")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    
    # ---------- Device ----------
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device id')
    parser.add_argument('--device', default='cuda:0',
                        help="device to use: cuda or cpu")
    
    # ---------- Data partitioning ----------
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for IID, 0 for non-IID')
    parser.add_argument('--noniid_s', type=int, default=20,
                        help='default shard size for non-IID partitioning')
    
    args = parser.parse_args()
    return args
