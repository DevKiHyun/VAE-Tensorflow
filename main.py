import argparse
import sys

sys.path.append(".")
import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=20, help='-')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--training_epoch', type=int, default=100, help='-')
    parser.add_argument('--batch_size', type=int, default=64, help='-')
    args, unknown = parser.parse_known_args()

    train.run_train(args)
