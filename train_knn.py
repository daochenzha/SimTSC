import os
import argparse

import numpy as np

from src.utils import read_dataset_from_npy, Logger

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']

def knn(X, y, train_idx, test_idx, distances, logger):
    correct, total = 0, 0
    for idx in test_idx:
        pred = None
        nearest_distance = None
        for _idx in train_idx:
            d = distances[idx][_idx]
            if nearest_distance == None or d < nearest_distance:
                nearest_distance = d
                pred = y[_idx]
        if pred == y[idx]:
            correct += 1
        total += 1
    acc = float(correct) / total
    return acc

def argsparser():
    parser = argparse.ArgumentParser("KNN")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')
    parser.add_argument('--shot', help='shot', type=int, default=1)

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    if args.dataset in multivariate_datasets:
        dtw_dir = os.path.join('datasets/multivariate') 
    else:
        dtw_dir = os.path.join(data_dir, 'ucr_datasets_dtw') 
    out_dir = os.path.join(log_dir, 'knn_log_'+str(args.shot)+'_shot')
    out_path = os.path.join(out_dir, args.dataset+'.txt')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, 'w') as f:
        logger = Logger(f)
        # Read dtw
        if args.dataset in multivariate_datasets:
            distances = np.load(os.path.join(dtw_dir, args.dataset+'_dtw.npy'))
        else:
            distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))
        if args.dataset in multivariate_datasets:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))
        else:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'ucr_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))

        acc = knn(X, y, train_idx, test_idx, distances, logger)

        logger.log(str(acc))
