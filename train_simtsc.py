import os
import argparse

import numpy as np
import torch

from src.utils import read_dataset_from_npy, Logger
from src.simtsc.model import SimTSC, SimTSCTrainer

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']

def train(X, y, train_idx, test_idx, distances, device, logger, K, alpha):
    nb_classes = len(np.unique(y, axis=0))

    input_size = X.shape[1]

    model = SimTSC(input_size, nb_classes)
    model = model.to(device)
    trainer = SimTSCTrainer(device, logger)

    model = trainer.fit(model, X, y, train_idx, distances, K, alpha)
    acc = trainer.test(model, test_idx)

    return acc


def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--shot', help='shot', type=int, default=1)
    parser.add_argument('--K', help='K', type=int, default=3)
    parser.add_argument('--alpha', help='alpha', type=float, default=0.3)

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Setup the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset in multivariate_datasets:
        dtw_dir = os.path.join('datasets/multivariate') 
        distances = np.load(os.path.join(dtw_dir, args.dataset+'_dtw.npy'))
    else:
        dtw_dir = os.path.join(data_dir, 'ucr_datasets_dtw') 
        distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))

    out_dir = os.path.join(log_dir, 'simtsc_log_'+str(args.shot)+'_shot'+str(args.K)+'_'+str(args.alpha))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(log_dir, args.dataset+'_'+str(args.seed)+'.txt')

    with open(out_path, 'w') as f:
        logger = Logger(f)
        # Read data
        if args.dataset in multivariate_datasets:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))
        else:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'ucr_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))

        # Train the model
        acc = train(X, y, train_idx, test_idx, distances, device, logger, args.K, args.alpha)

        logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, acc))
        logger.log(str(acc))
