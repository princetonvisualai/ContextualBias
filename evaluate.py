import pickle
import time
import torch
import numpy as np
import os
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve

from classifier import multilabel_classifier
from load_data import *

modelpath = 'save/stage2_cam/stage2_99.pth'
print('Loaded model from', modelpath)
indir = '/n/fs/context-scr/evaldata/train/'
outdir = 'evalresults/stage2_cam/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
print('Save evaluation results in', outdir)

device = torch.device('cuda')
dtype = torch.float32

Classifier = multilabel_classifier(device, dtype, modelpath=modelpath)
Classifier.model.cuda()
Classifier.model.eval()

biased_classes = pickle.load(open('/n/fs/context-scr/biased_classes.pkl', 'rb'))
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/humanlabels_to_onehot.pkl', 'rb'))

start_time = time.time()

# Evaluate on exclusive and co-occur sets
exclusive_AP_list = []; exclusive_mAP_list = []
cooccur_AP_list = []; cooccur_mAP_list = []

# Loop over each of the K biased categories
for b in biased_classes.keys():

    print('\n{} - {}'.format(b, biased_classes[b]))

    # Exclusive
    exclusive_labels = '{}/{}_{}_{}.pkl'.format(indir, 'exclusive', b, biased_classes[b])
    exclusiveset = create_dataset(COCOStuff, labels=exclusive_labels, B=500)

    with torch.no_grad():
        labels_list = np.array([], dtype=np.float32).reshape(0, 171)
        scores_list = np.array([], dtype=np.float32).reshape(0, 171)

        for i, (images, labels) in enumerate(exclusiveset):
            images, labels = images.to(device=Classifier.device, dtype=Classifier.dtype), labels.to(device=Classifier.device, dtype=Classifier.dtype)
            scores, _ = Classifier.forward(images)
            scores = torch.sigmoid(scores).squeeze()
            labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
            scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)
            #print('     exclusive batch {}/{}, {} seconds'.format(i, len(exclusiveset), time.time()-start_time))

        with open('{}/labels_exclusive_{}.pkl'.format(outdir, b), 'wb+') as handle:
            pickle.dump(labels_list, handle)
        with open('{}/scores_exclusive_{}.pkl'.format(outdir, b), 'wb+') as handle:
            pickle.dump(scores_list, handle)

        exclusive_APs = []
        for k in range(171):
            exclusive_APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
        exclusive_mAP = np.nanmean(exclusive_APs)


    # Co-occur
    cooccur_labels = '{}/{}_{}_{}.pkl'.format(indir, 'cooccur', b, biased_classes[b])
    cooccurset = create_dataset(COCOStuff, labels=cooccur_labels, B=500)

    with torch.no_grad():
        labels_list = np.array([], dtype=np.float32).reshape(0, 171)
        scores_list = np.array([], dtype=np.float32).reshape(0, 171)

        for i, (images, labels) in enumerate(cooccurset):
            images, labels = images.to(device=Classifier.device, dtype=Classifier.dtype), labels.to(device=Classifier.device, dtype=Classifier.dtype)
            scores, _ = Classifier.forward(images)
            scores = torch.sigmoid(scores).squeeze()
            labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
            scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)
            #print('     cooccur batch {}/{}, {} seconds'.format(i, len(cooccurset), time.time()-start_time))

        with open('{}/labels_cooccur_{}.pkl'.format(outdir, b), 'wb+') as handle:
            pickle.dump(labels_list, handle)
        with open('{}/scores_cooccur_{}.pkl'.format(outdir, b), 'wb+') as handle:
            pickle.dump(scores_list, handle)

        cooccur_APs = []
        for k in range(171):
            cooccur_APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
        cooccur_mAP = np.nanmean(cooccur_APs)


    # Save values in a list and print
    exclusive_AP_list.append(exclusive_APs[humanlabels_to_onehot[b]])
    cooccur_AP_list.append(cooccur_APs[humanlabels_to_onehot[b]])
    exclusive_mAP_list.append(exclusive_mAP)
    cooccur_mAP_list.append(cooccur_mAP)

    print('\n{} - {}'.format(b, biased_classes[b]))
    print('   exclusive: AP {:.5f}'.format(exclusive_APs[humanlabels_to_onehot[b]]))
    print('   co-occur: AP {:.5f}'.format(cooccur_APs[humanlabels_to_onehot[b]]))


print()
print('Validation mean of biased class APs: exclusive {:.5f}, co-occur {:.5f}'.format(np.mean(exclusive_AP_list), np.mean(cooccur_AP_list)))
