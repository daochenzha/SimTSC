import os
import argparse

import numpy as np

from src.utils import read_dataset, read_multivariate_dataset

dataset_dir = './datasets/UCRArchive_2018'
multivariate_dir = './datasets/multivariate'
output_dir = './tmp'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']

def argsparser():
    parser = argparse.ArgumentParser("SimTSC data creator")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--shot', help='How many labeled time-series per class', type=int, default=1)

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)

    # Create dirs
    if args.dataset in multivariate_datasets:
        output_dir = os.path.join(output_dir, 'multivariate_datasets_'+str(args.shot)+'_shot')
    else:
        output_dir = os.path.join(output_dir, 'ucr_datasets_'+str(args.shot)+'_shot')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_multivariate_dataset(multivariate_dir, args.dataset, args.shot)
    else:
        X, y, train_idx, test_idx = read_dataset(dataset_dir, args.dataset, args.shot)
    data = {
                'X': X,
                'y': y,
                'train_idx': train_idx,
                'test_idx': test_idx
            }
    np.save(os.path.join(output_dir, args.dataset), data)
