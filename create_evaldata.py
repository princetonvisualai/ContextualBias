import pickle
from os import path, mkdir
import torch
import numpy as np
from PIL import Image
import argparse 

# python create_evaldata.py --val --train (to run without saving .pkl files)
# python create_evaldata.py --val --train --save (to run and save .pkl files)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--val', action='store_true', default=True,
                      help='Run on validation set',
                      dest='val')
parser.add_argument('--train', action='store_true', default=False,
                      help='Run on training set',
                      dest='train')
parser.add_argument('--save', action='store_true', default=False,
                      help='Save dictionaries of exclusive and co-occur image filepaths',
                      dest='save')

run_val = parser.parse_args().val
run_train = parser.parse_args().train
do_save = parser.parse_args().save

datasets = []
if run_val:
    datasets.append('labels_val.pkl')
if run_train:
    datasets.append('labels_train.pkl')

# 20 most biased classes identified in the original paper
biased_classes = {}
biased_classes['cup'] = 'dining table'
biased_classes['wine glass'] = 'person'
biased_classes['handbag'] = 'person'
biased_classes['apple'] = 'fruit'
biased_classes['car'] = 'road'
biased_classes['bus'] = 'road'
biased_classes['potted plant'] = 'vase'
biased_classes['spoon'] = 'bowl'
biased_classes['microwave'] = 'oven'
biased_classes['keyboard'] = 'mouse'
biased_classes['skis'] = 'person'
biased_classes['clock'] = 'building-other'
biased_classes['sports ball'] = 'person'
biased_classes['remote'] = 'person'
biased_classes['snowboard'] = 'person'
biased_classes['toaster'] = 'ceiling-other' # unclear from the paper
#biased_classes['toaster'] = 'ceiling-tile' # unclear from the paper
biased_classes['hair drier'] = 'towel'
biased_classes['tennis racket'] = 'person'
biased_classes['skateboard'] = 'person'
biased_classes['baseball glove'] = 'person'
with open('biased_classes.pkl', 'wb+') as handle:
    pickle.dump(biased_classes, handle)

# Map human-readable labels to [0-170] label space used for training classifiers
humanlabels_to_onehot = pickle.load(open('humanlabels_to_onehot.pkl', 'rb'))
biased_classes_mapped = dict((humanlabels_to_onehot[key], humanlabels_to_onehot[value]) for (key, value) in biased_classes.items())
with open('biased_classes_mapped.pkl', 'wb+') as handle:
    pickle.dump(biased_classes_mapped, handle)

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

        # Loop over all 40504 images in the validation set
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
        if dataset_filename == 'labels_val.pkl':
            outdir = 'evaldata/val'
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
            exclusive_val = {}
            for key in b_exclusive:
                exclusive_val[key] = labels[key]
            with open('{}/exclusive_{}_{}.pkl'.format(outdir, b_human, c_human), 'wb+') as handle:
                pickle.dump(exclusive_val, handle)

            # cooccur: save image file paths and one-hot-encoded labels
            cooccur_val = {}
            for key in b_cooccur:
                cooccur_val[key] = labels[key]
            with open('{}/cooccur_{}_{}.pkl'.format(outdir, b_human, c_human), 'wb+') as handle:
                pickle.dump(cooccur_val, handle)
