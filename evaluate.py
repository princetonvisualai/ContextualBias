import pickle, time, collections
import torch
import numpy as np
import os
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve
import sys

from classifier import multilabel_classifier
from load_data import *

dataset = sys.argv[1]

# Specify the model to evaluate
modelpath = '{}/save/stage2_cam/stage2_99.pth'.format(dataset)
print('Loaded model from', modelpath)
indir = '/n/fs/context-scr/{}/evaldata/train/'.format(dataset)
outdir = '{}/evalresults/stage2_cam/'.format(dataset)
if not os.path.exists(outdir):
    os.makedirs(outdir)
print('Save evaluation results in', outdir)

device = torch.device('cuda')
dtype = torch.float32

# Load useful files
biased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/biased_classes_mapped.pkl'.format(dataset), 'rb'))
unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/unbiased_classes_mapped.pkl'.format(dataset), 'rb'))
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(dataset), 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# Load dataset to evaluate and create a data loader
datapath = 'labels_val.pkl'
loader = create_dataset(dataset, labels=datapath, B=100)
labels = pickle.load(open('/n/fs/context-scr/{}/{}'.format(dataset, datapath), 'rb'))

# Load model and set it in evaluation mode
Classifier = multilabel_classifier(device, dtype, modelpath=modelpath)
print('Loaded model from', modelpath)
Classifier.model.cuda()
Classifier.model.eval()

# Get scores for all images
with torch.no_grad():
    if dataset == 'COCOStuff':
        num_categs = 171
    elif dataset == 'AwA':
        num_categs = 85
    else:
        num_categs = 0
        print('Invalid dataset: {}'.format(dataset))

    labels_list = np.array([], dtype=np.float32).reshape(0, num_categs)
    scores_list = np.array([], dtype=np.float32).reshape(0, num_categs)

    for i, (images, labels, ids) in enumerate(loader):

        images, labels = images.to(device=device, dtype=dtype), labels.to(device=device, dtype=dtype)
        scores, _ = Classifier.forward(images)
        scores = torch.sigmoid(scores).squeeze()

        labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
        scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

# Calculate AP for each category and mAP for all/unbiased categories
APs = []
for k in range(num_categs):
    APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
mAP = np.nanmean(APs)
mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
print('mAP: all {} {:.5f}, unbiased 60 {:.5f}'.format(num_categs, mAP, mAP_unbiased))

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
