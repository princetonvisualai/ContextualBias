import pickle, time, collections
import torch
import numpy as np
import os
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve
import sys
import torch.nn as nn

from classifier import multilabel_classifier
from load_data import *

# Example usage:
# python evaluate.py COCOStuff cam 99
#
# This will compute mAPs for the model stored at cam_99.pth

dataset = sys.argv[1]
model = sys.argv[2]
epoch = int(sys.argv[3])

# Specify the model to evaluate
if dataset == 'UnRel':
    modelpath = 'COCOStuff/save/{}/{}_{}.pth'.format(model, model, epoch)
else:
    modelpath = '{}/save/{}/{}_{}.pth'.format(dataset, model, model, epoch)
print('Loaded model from', modelpath)
#indir = '/n/fs/context-scr/{}/evaldata/train/'.format(dataset)
outdir = '{}/evalresults/{}/'.format(dataset, model)
if not os.path.exists(outdir):
    os.makedirs(outdir)
print('Save evaluation results in', outdir)

device = torch.device('cuda') # cuda
dtype = torch.float32

# Load useful files
biased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/biased_classes_mapped.pkl'.format(dataset), 'rb'))
if dataset == 'COCOStuff':
    unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/unbiased_classes_mapped.pkl'.format(dataset), 'rb'))

if dataset == 'UnRel': # use COCOStuff labels
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
else:
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(dataset), 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# Load dataset to evaluate and create a data loader
datapath = 'labels_val.pkl'
loader = create_dataset(dataset, labels=datapath, B=100)
labels = pickle.load(open('/n/fs/context-scr/{}/{}'.format(dataset, datapath), 'rb'))

# Set number of categoriews in dataset
if dataset == 'COCOStuff' or dataset == 'UnRel':
    num_categs = 171
elif dataset == 'AwA':
    num_categs = 85
elif dataset == 'DeepFashion':
    num_categs = 250
else:
    num_categs = 0
    print('Invalid dataset: {}'.format(dataset))

# Load model and set it in evaluation mode
classifier = multilabel_classifier(device, dtype, num_categs=num_categs, modelpath=modelpath)
classifier = classifier
print('Loaded model from', modelpath)
classifier.model.cuda()
classifier.model.eval()

# Get scores for all images
with torch.no_grad():

    labels_list = np.array([], dtype=np.float32).reshape(0, num_categs)
    scores_list = np.array([], dtype=np.float32).reshape(0, num_categs)

    for i, (images, labels, ids) in enumerate(loader):

        images, labels = images.to(device=device, dtype=dtype), labels.to(device=device, dtype=dtype)
        scores, _ = classifier.forward(images)
        scores = torch.sigmoid(scores).squeeze()

        labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
        scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

# Calculate AP for each category and mAP for all/unbiased categories
APs = []
for k in range(num_categs):
    APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
mAP = np.nanmean(APs)
if dataset == 'COCOStuff':
    mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
    print('mAP: all {} {:.5f}, unbiased 60 {:.5f}'.format(num_categs, mAP, mAP_unbiased))
else:
    print('mAP: all {} {:.5f}'.format(num_categs, mAP))

    # Calculate exclusive/co-occur AP for each biased category
    exclusive_AP_list = []
    cooccur_AP_list = []
    for b in biased_classes_mapped.keys():
        c = biased_classes_mapped[b]

        # Put the images into 3 categories
        exclusive = (labels_list[:,b]==1) & (labels_list[:,c]==0)
        cooccur = (labels_list[:,b]==1) & (labels_list[:,c]==1)
        other = (~exclusive) & (~cooccur)

        # Calculate AP for exclusive and cooccur sets
        exclusive_AP = average_precision_score(labels_list[exclusive+other,b], scores_list[exclusive+other,b])
        cooccur_AP = average_precision_score(labels_list[cooccur+other,b], scores_list[cooccur+other,b])
        exclusive_AP_list.append(exclusive_AP)
        cooccur_AP_list.append(cooccur_AP)

        print('\n{} - {}'.format(onehot_to_humanlabels[b], onehot_to_humanlabels[c]))
        print('   exclusive: AP {:.5f}'.format(exclusive_AP*100.))
        print('   co-occur: AP {:.5f}'.format(cooccur_AP*100.))

    print('Mean: exclusive {:.5f}, co-occur {:.5f}'.format(np.mean(exclusive_AP_list)*100., np.mean(cooccur_AP_list)*100.))
