import pickle, argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default=None)
parser.add_argument('--labels', type=str, default='UnRel/labels_unrel.pkl')
parser.add_argument('--splitbiased', default=False, action="store_true")
parser.add_argument('--batchsize', type=int, default=170)
parser.add_argument('--nclasses', type=int, default=171)
parser.add_argument('--hs', type=int, default=2048)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--dtype', default=torch.float32)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

if arg['splitbiased'] == True:
    arg['nclasses'] = arg['nclasses'] + 20

# Load utility files
humanlabels_to_onehot = pickle.load(open('COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
biased_classes_mapped = pickle.load(open('COCOStuff/biased_classes_mapped.pkl', 'rb'))

print(sorted(list(biased_classes_mapped.keys())))

# Create dataloader
testset = create_dataset(None, arg['labels'], biased_classes_mapped, B=arg['batchsize'], train=False, splitbiased=arg['splitbiased'])

# Load model
classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'], hidden_size=arg['hs'])

# Do inference with the model
labels_list, scores_list, val_loss_list = classifier.test(testset)

# Calculate AP for car (2), bus (5), skateboard (36); person (0), road (137)
APs = []
for category in ['car', 'bus', 'skateboard']:
    b = humanlabels_to_onehot[category]
    c = biased_classes_mapped[b]

    if arg['splitbiased']:

        if b == 2: b_cooccur = 171 + 0
        if b == 5: b_cooccur = 171 + 1
        if b == 36: b_cooccur = 171 + 7

        # Identify co-occur, exclusive, other images
        cooccur = (labels_list[:,b_cooccur]==1)
        exclusive = (labels_list[:,b]==1)
        other = (~exclusive) & (~cooccur)
        print('cooccur, exclusive, other', cooccur.sum(), exclusive.sum(), other.sum())

        # Concatenate labels and scores
        labels = np.concatenate((labels_list[cooccur, b_cooccur],
            labels_list[exclusive, b],
            labels_list[other, b]))
        scores = np.concatenate((scores_list[cooccur, b_cooccur],
            scores_list[exclusive, b],
            (scores_list[other, b]+scores_list[other, b_cooccur])/2))

        AP = average_precision_score(labels, scores)

    else:

        AP = average_precision_score(labels_list[:,b], scores_list[:,b])

    APs.append(AP)
    print('AP ({}): {:.2f}'.format(category, AP*100.))

# Calculate mAP
mAP = np.nanmean(APs)
print('\nmAP (3 classes): {:.2f}'.format(mAP*100.))
