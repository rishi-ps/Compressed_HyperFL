import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of training epochs")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: n")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_epoch', type=int, default=5,
                        help="the number of local epochs")
    parser.add_argument('--local_iter', type=int, default=1,
                        help="the number of local iterations")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: b")
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--lr_g', type=float, default=0.1,
                        help='learning rate for classifier')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='training optimizer')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--train_rule', type=str, default='FedAvg',
                        help='the training rule for personalized FL')
    parser.add_argument('--local_size', type=int, default=600,
                        help='number of samples for each client')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='The dimension of client embedding.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='The hidden dimension of hypernetwork.')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device id')
    parser.add_argument('--device', default='cuda:0',
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to non-IID. Set to 1 for IID.')
    parser.add_argument('--noniid_s', type=int, default=20,
                        help='Default set to 20. Set to 100 for IID.')
    args = parser.parse_args()
    return args