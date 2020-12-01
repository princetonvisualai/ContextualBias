import pickle, time, argparse
from os import path, mkdir
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--modelpath', type=str, default=None)
parser.add_argument('--labels', type=str, default='/n/fs/context-scr/COCOStuff/labels_val.pkl')
parser.add_argument('--batchsize', type=int, default=200)
parser.add_argument('--nclasses', type=int, default=171)
parser.add_argument('--hs', type=int, default=2048)
parser.add_argument('--splitbiased', type=bool, default=False)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--dtype', default=torch.float32)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load utility files
biased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/biased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
if arg['dataset'] == 'COCOStuff':
    unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/unbiased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# Create dataloader
valset = create_dataset(arg['dataset'], arg['labels'], biased_classes_mapped, B=arg['batchsize'], train=False)

# Load model
Classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'], hidden_size=arg['hs'])

# Do inference with the model
labels_list, scores_list, val_loss_list = Classifier.test(valset)

# Calculate and record mAP
APs = []
for k in range(arg['nclasses']):
    APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
mAP = np.nanmean(APs)
print('mAP (all): {:.2f}'.format(mAP*100.))
if arg['dataset'] == 'COCOStuff':
    mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
    print('mAP (unbiased): {:.2f}\n'.format(mAP_unbiased*100.))

# Calculate exclusive/co-occur AP for each biased category
exclusive_AP_list = []
cooccur_AP_list = []
biased_classes_list = sorted(list(biased_classes_mapped.keys()))
for k in range(len(biased_classes_list)):
    b = biased_classes_list[k]
    c = biased_classes_mapped[b]

    # Categorize the images into co-occur/exclusive/other
    if arg['splitbiased']:
        cooccur = (labels_list[:,arg['nclasses']+k]==1)
        exclusive = (labels_list[:,b]==1)
    else:
        cooccur = (labels_list[:,b]==1) & (labels_list[:,c]==1)
        exclusive = (labels_list[:,b]==1) & (labels_list[:,c]==0)
    other = (~exclusive) & (~cooccur)

    # Calculate AP for co-occur/exclusive sets
    if arg['splitbiased']:
        cooccur_AP = average_precision_score(labels_list[cooccur+other, arg['nclasses']+k],
            scores_list[cooccur+other, arg['nclasses']+k])
    else:
        cooccur_AP = average_precision_score(labels_list[cooccur+other, b],
            scores_list[cooccur+other, b])
    exclusive_AP = average_precision_score(labels_list[exclusive+other ,b],
        scores_list[exclusive+other, b])

    # Record and print
    cooccur_AP_list.append(cooccur_AP)
    exclusive_AP_list.append(exclusive_AP)
    print('{:>10} - {:>10}: exclusive {:5.2f}, co-occur {:5.2f}'.format(onehot_to_humanlabels[b], onehot_to_humanlabels[c], exclusive_AP*100., cooccur_AP*100.))

print('\nMean: exclusive {:.2f}, co-occur {:.2f}'.format(np.mean(exclusive_AP_list)*100., np.mean(cooccur_AP_list)*100.))
