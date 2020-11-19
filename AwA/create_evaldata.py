import pickle
from os import path, mkdir
import torch
import numpy as np
from PIL import Image
import argparse 

# python create_evaldata.py --test --train (to run without saving .pkl files)
# python create_evaldata.py --test --train --save (to run and save .pkl files)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--test', action='store_true', default=True,
                      help='Run on test set',
                      dest='val')
parser.add_argument('--train', action='store_true', default=False,
                      help='Run on training set',
                      dest='train')
parser.add_argument('--save', action='store_true', default=False,
                      help='Save dictionaries of exclusive and co-occur image filepaths',
                      dest='save')

run_test = parser.parse_args().test
run_train = parser.parse_args().train
do_save = parser.parse_args().save

datasets = []
if run_test:
    datasets.append('labels_test.pkl')
if run_train:
    datasets.append('labels_train.pkl')

# 20 most biased classes identified in the original paper
biased_classes = {}
biased_classes['white'] = 'ground'
biased_classes['longleg'] = 'domestic'
biased_classes['forager'] = 'nestspot'
biased_classes['lean'] = 'stalker'
biased_classes['fish'] = 'timid'
biased_classes['hunter'] = 'big'
biased_classes['plains'] = 'stalker'
biased_classes['nocturnal'] = 'white'
biased_classes['nestspot'] = 'meatteeth'
biased_classes['jungle'] = 'muscle'
biased_classes['muscle'] = 'black'
biased_classes['meat'] = 'fish'
biased_classes['mountains'] = 'paws'
biased_classes['tree'] = 'tail'
biased_classes['domestic'] = 'inactive'
biased_classes['spots'] = 'longleg' 
biased_classes['bush'] = 'meat'
biased_classes['buckteeth'] = 'smelly'
biased_classes['slow'] = 'strong'
biased_classes['blue'] = 'coastal'
with open('biased_classes.pkl', 'wb+') as handle:
    pickle.dump(biased_classes, handle)

# Map human-readable labels to [0-84] label space used for training classifiers
predicatelabels_to_onehot = pickle.load(open('predicatelabels_to_onehot.pkl', 'rb'))
biased_classes_mapped = dict((predicatelabels_to_onehot[key], predicatelabels_to_onehot[value]) for (key, value) in biased_classes.items())
with open('biased_classes_mapped.pkl', 'wb+') as handle:
    pickle.dump(biased_classes_mapped, handle)

# LEFT OFF HERE

# Save non-biased object classes (80 - 20 things) used in the appendiix
unbiased_classes_mapped = [i for i in list(np.arange(80)) if i not in biased_classes_mapped.keys()]
with open('unbiased_classes_mapped.pkl', 'wb+') as handle:
    pickle.dump(unbiased_classes_mapped, handle)
    
for dataset_filename in datasets:
    # Construct 'exclusive' and 'co-occur' test distributions fom the dataset
    labels = pickle.load(open(dataset_filename, 'rb'))
    print('{} images in the dataset {}'.format(len(labels), dataset_filename))

    exclusive_set = {}
    cooccur_set = {}

    # Loop over K biased categories
    for b in biased_classes_mapped.keys():
        exclusive = []
        cooccur = []
        exclusive_positive = 0
        cooccur_positive = 0
        b0c0 = 0
        b0c1 = 0

        # Loop over all images in the test set
        for key in labels.keys():
            label = labels[key]

            # Co-occur
            if label[b]==1 and label[biased_classes_mapped[b]]==1:
                cooccur.append(key)
                cooccur_positive += 1
            # Exclusive
            elif label[b]==1 and label[biased_classes_mapped[b]]==0:
                exclusive.append(key)
                exclusive_positive += 1
            # Other
            elif label[b]==0 and label[biased_classes_mapped[b]]==1:
                cooccur.append(key)
                exclusive.append(key)
                b0c1 += 1
            else:
                cooccur.append(key)
                exclusive.append(key)
                b0c0 += 1

        exclusive_set[b] = exclusive
        cooccur_set[b] = cooccur

        # Print how many images are in each set
        b_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(b)]
        c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(biased_classes_mapped[b])]
        print('\n{} - {}'.format(b_human, c_human))
        print('  exclusive: {}+{}={} images'.format(exclusive_positive, b0c1+b0c0, len(exclusive_set[b])))
        print('  co-occur: {}+{}={} images'.format(cooccur_positive, b0c1+b0c0, len(cooccur_set[b])))

    # Save exclusive and co-occur sets
    if do_save:
        if dataset_filename == 'labels_test.pkl':
            outdir = 'evaldata/test'
        elif dataset_filename == 'labels_train.pkl':
            outdir = 'evaldata/train'
        else:
            outdir = 'evaldata/misc'
        if not path.isdir(outdir):
            mkdir(outdir)

        for b in biased_classes_mapped.keys():
            b_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(b)]
            c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(biased_classes_mapped[b])]

            b_exclusive = exclusive_set[b]
            b_cooccur = cooccur_set[b]

            # exclusive: save image file paths and one-hot-encoded labels
            exclusive_test = {}
            for key in b_exclusive:
                exclusive_test[key] = labels[key]
            with open('{}/exclusive_{}_{}.pkl'.format(outdir, b_human, c_human), 'wb+') as handle:
                pickle.dump(exclusive_test, handle)

            # cooccur: save image file paths and one-hot-encoded labels
            cooccur_test = {}
            for key in b_cooccur:
                cooccur_test[key] = labels[key]
            with open('{}/cooccur_{}_{}.pkl'.format(outdir, b_human, c_human), 'wb+') as handle:
                pickle.dump(cooccur_test, handle)
