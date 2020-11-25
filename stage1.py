import pickle
import time
import os
import torch
import numpy as np
import sys
import argparse

from classifier import multilabel_classifier
from load_data import *

# Example use case:
# python stage1.py --dataset COCOStuff --epochs 100 --gpus 2

def run_train(args):
    dataset = args.dataset[0]
    nepochs = args.nepochs
    modelpath = args.modelpath
    
    # Create data loader
    trainset = create_dataset(dataset, labels='labels_train.pkl', B=200) # instead of 200
    valset = create_dataset(dataset, labels='labels_val.pkl', B=500)
    
    print('Created train and val datasets \n')

    if dataset == 'COCOStuff':
        unbiased_classes_mapped = pickle.load(open('{}/unbiased_classes_mapped.pkl'.format(dataset), 'rb'))

    # Create output directory
    outdir = '{}/save/stage1'.format(dataset)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        print('Created', outdir, '\n')

    # Initialize classifier
    if dataset == 'COCOStuff':
        num_categs = 171
        learning_rate = 0.1
    elif dataset == 'AwA':
        num_categs = 85
        learning_rate = 0.01
    elif dataset == 'DeepFashion':
        num_categs = 250
        learning_rate = 0.1
    else:
        num_categs = 0
        learning_rate = 0.01
        print('Invalid dataset: {}'.format(dataset))

    classifier = multilabel_classifier(torch.device('cuda'), torch.float32, 
                                       num_categs=num_categs, learning_rate=learning_rate, 
                                       modelpath=modelpath) # cuda

    # Start stage 1 training
    start_time = time.time()
    for i in range(classifier.epoch, nepochs):

        if dataset in ['COCOStuff', 'DeepFashion'] and i == 60: # Reduce learning rate from 0.1 to 0.01
            classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.01, momentum=0.9)

        classifier.train(trainset)
        if (i + 1) % 5 == 0:
            classifier.save_model('{}/stage1_{}.pth'.format(outdir, i))

        APs, mAP = classifier.test(valset)
        if dataset == 'COCOStuff':
            mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
            print('Validation mAP: all {} {:.5f}, unbiased 60 {:.5f}'.format(num_categs, mAP, mAP_unbiased))
        else:
            print('Validation mAP: all {} {:.5f}'.format(num_categs, mAP))

        print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
        print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, 
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, 
                        help='Number of gpus per node')
    parser.add_argument('--nepochs', default=2, type=int, 
                        metavar='N', help='Number of epochs to train until')
    parser.add_argument('--dataset', default='COCOStuff', type=str,
                        required=True, nargs=1, dest='dataset',
                        help='Dataset used to train this model')
    parser.add_argument('--modelpath', default=None, type=str,
                        nargs=1, dest='modelpath')
    args = parser.parse_args()
    run_train(args)

if __name__ == '__main__':
    main()

