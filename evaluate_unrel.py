import pickle, argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default=None)
parser.add_argument('--labels', type=str, default='/n/fs/context-scr/UnRel/labels_unrel.pkl')
parser.add_argument('--splitbiased', type=bool, default=False)
parser.add_argument('--batchsize', type=int, default=170)
parser.add_argument('--nclasses', type=int, default=171)
parser.add_argument('--hs', type=int, default=2048)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--dtype', default=torch.float32)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load utility files
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
biased_classes_mapped = pickle.load(open('/n/fs/context-scr/COCOStuff/biased_classes_mapped.pkl', 'rb'))

# Create dataloader
testset = create_dataset(None, arg['labels'], biased_classes_mapped, B=arg['batchsize'], train=False, splitbiased=arg['splitbiased'])

# Load model
classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'], hidden_size=arg['hs'])

# Do inference with the model
labels_list, scores_list, val_loss_list = classifier.test(testset)

# Calculate AP for car (2), bus (5), skateboard (36)
APs = []
for category in ['car', 'bus', 'skateboard']:
    k = humanlabels_to_onehot[category]

    if arg['splitbiased']:

        if k == 2: k_cooccur = 171 + 0
        if k == 5: k_cooccur = 171 + 1
        if k == 36: k_cooccur = 171 + 7

        # Identify co-occur, exclusive, other images
        cooccur = (labels_list[:,k_cooccur]==1)
        exclusive = (labels_list[:,k]==1)
        other = (~exclusive) & (~cooccur)

        # Concatenate labels and scores
        labels = np.concatenate((labels_list[cooccur, k_cooccur],
            labels_list[exclusive, k],
            labels_list[other, k]))
        scores = np.concatenate((scores_list[cooccur, k_cooccur],
            scores_list[exclusive, k],
            (scores_list[other, k]+scores_list[other, k_cooccur])/2))

        AP = average_precision_score(labels, scores)

    else:
        
        AP = average_precision_score(labels_list[:,k], scores_list[:,k])

    APs.append(AP)
    print('AP ({}): {:.2f}'.format(category, AP*100.))

# Calculate mAP
mAP = np.nanmean(APs)
print('\nmAP (3 classes): {:.2f}'.format(mAP*100.))
