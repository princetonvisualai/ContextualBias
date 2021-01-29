import pickle, time, argparse, os
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from PIL import Image
import matplotlib.pyplot as plt

from classifier import multilabel_classifier
from load_data import *
from recall import recall3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nclasses', type=int, default=171)
    parser.add_argument('--standard_modelpath', type=str, default=None)
    parser.add_argument('--fs_modelpath', type=str, default=None)
    parser.add_argument('--b', type=str, default=None)
    parser.add_argument('--labels_test', type=str, default='COCOStuff/labels_test.pkl')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--hs', type=int, default=2048)
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    arg = vars(parser.parse_args())
    print('\n', arg, '\n')

    # Make the directory
    fs_but_not_standard_outdir = '{}/fs_but_not_standard'.format(arg['outdir'])
    if not os.path.exists(fs_but_not_standard_outdir):
        os.makedirs(fs_but_not_standard_outdir)
    not_fs_not_standard_outdir = '{}/not_fs_not_standard'.format(arg['outdir'])
    if not os.path.exists(not_fs_not_standard_outdir):
        os.makedirs(not_fs_not_standard_outdir)

    # Load utility files
    biased_classes_mapped = pickle.load(open('{}/biased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
    if arg['dataset'] == 'COCOStuff':
        unbiased_classes_mapped = pickle.load(open('{}/unbiased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
    humanlabels_to_onehot = pickle.load(open('{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

    # Create dataloader
    testset = create_dataset(arg['dataset'], arg['labels_test'], biased_classes_mapped, B=arg['batchsize'], removecimages=True, train=False, splitbiased=False)

    # Load models
    standard_classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['standard_modelpath'], hidden_size=arg['hs'], attribdecorr=False)
    fs_classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['fs_modelpath'], hidden_size=arg['hs'], attribdecorr=False)

    # Do inference with the models
    b = humanlabels_to_onehot[arg['b']]
    c = biased_classes_mapped[b]
    print('Running for pair ({}, {})'.format(arg['b'], onehot_to_humanlabels[c]), flush=True)
    print('Standard classifier')
    standard_success, standard_failures = standard_classifier.get_prediction_examples(testset, b)
    print('Feature-split classifier')
    fs_success, fs_failures = fs_classifier.get_prediction_examples(testset, b)

    fs_but_not_standard = list(fs_success.intersection(standard_failures))
    not_fs_not_standard = list(fs_failures.intersection(standard_failures))
    print('Feature-split successful, standard failed:', len(fs_but_not_standard), flush=True)
    print('Both feature-split and standard failed:', len(not_fs_not_standard), flush=True)

    arg['num_examples'] = min(arg['num_examples'], min(len(fs_but_not_standard), len(not_fs_not_standard)))
    for i in range(arg['num_examples']):
        # Feature-split is correct but standard fails
        img_path = fs_but_not_standard[i]
        img = Image.open(img_path).convert('RGB')
        img_name = img_path.split('/')[-1][:-4]
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title('{}, {}'.format(arg['b'], onehot_to_humanlabels[c]))
        plt.savefig('{}/{}'.format(fs_but_not_standard_outdir, img_name))
        plt.show()
        plt.close()

        # Both feature-split and standard fail
        img_path = not_fs_not_standard[i]
        img = Image.open(img_path).convert('RGB')
        img_name = img_path.split('/')[-1][:-4]
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title('{}, {}'.format(arg['b'], onehot_to_humanlabels[c]))
        plt.savefig('{}/{}'.format(not_fs_not_standard_outdir, img_name))
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
