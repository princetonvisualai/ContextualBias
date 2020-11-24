import pickle
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compute alpha weights.')
parser.add_argument('--dataset', default='COCOStuff', nargs=1, type=str, required=True,
                    help='Specifies the dataset to process',
                    dest='dataset')
parser.add_argument('--val', action='store_true', default=False,
                    help='Compute validation set weights',
                    dest='val')
parser.add_argument('--train', action='store_true', default=False,
                    help='Compute train set weights',
                    dest='train')

dataset = parser.parse_args().dataset[0]
run_val = parser.parse_args().val
run_train = parser.parse_args().train

if dataset == 'COCOStuff':
    num_categs = 171
elif dataset == 'AwA':
    num_categs = 85
elif dataset == 'DeepFashion':
    num_categs = 250
else:
    num_categs = 0
    print('Invalid dataset: {}'.format(dataset))

datatypes = []
if run_val:
    datatypes.append('train')
if run_train:
    datatypes.append('val')

for datatype in datatypes:
    labels = pickle.load(open('{}/labels_{}.pkl'.format(dataset, datatype), 'rb'))
    biased_classes = pickle.load(open('{}/biased_classes.pkl'.format(dataset), 'rb'))
    biased_classes_mapped = pickle.load(open('{}/biased_classes_mapped.pkl'.format(dataset), 'rb'))
    humanlabels_to_onehot = pickle.load(open('{}/humanlabels_to_onehot.pkl'.format(dataset), 'rb'))

    # Calculate alphas
    alphas = {}
    for b in biased_classes_mapped.keys():
        exclusive_positive = 0
        cooccur_positive = 0

        for key in labels.keys():
            label = labels[key]

            if label[b]==1 and label[biased_classes_mapped[b]]==1:
                cooccur_positive += 1
            elif label[b]==1 and label[biased_classes_mapped[b]]==0:
                exclusive_positive += 1

        alpha = np.sqrt(cooccur_positive/exclusive_positive)

        # Print how many images are in each set
        b_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(b)]
        c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(biased_classes_mapped[b])]
        print()
        print('{} - {}'.format(b_human, c_human))
        print('  co-occur: {} images'.format(cooccur_positive))
        print('  exclusive: {} images'.format(exclusive_positive))
        print('  alpha: {}'.format(alpha))
        alphas[b_human] = alpha

    # Calculate and save weights
    w = torch.ones(num_categs)
    for key in alphas.keys():
        index = humanlabels_to_onehot[key]
        if alphas[key] > 0:
            w[index] = alphas[key]

    with open('{}/weight_{}.pkl'.format(dataset, datatype), 'wb+') as handle:
        pickle.dump(w, handle)
