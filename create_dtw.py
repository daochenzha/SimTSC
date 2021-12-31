import os
import argparse

import numpy as np

from src.utils import read_X

import dtw

dataset_dir = './datasets/UCRArchive_2018'
output_dir = './tmp'

def argsparser():
    parser = argparse.ArgumentParser("SimTSC dtw creator")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')

    return parser

def get_dtw(X):
    X = X.copy(order='C').astype(np.float64)
    X[np.isnan(X)] = 0
    distances = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for i in range(len(X)):
        for j in range(len(X)):
            data = X[i]
            query = X[j]
            distances[i][j] = dtw.query(data, query, r=min(len(data)-1, len(query)-1, 100))['value']
    return distances

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    result_dir = os.path.join(output_dir, 'ucr_datasets_dtw')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    X = read_X(dataset_dir, args.dataset)

    dtw_arr = get_dtw(X)
    np.save(os.path.join(result_dir, args.dataset), dtw_arr)
