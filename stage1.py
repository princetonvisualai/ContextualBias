import pickle
import time
from os import path, mkdir
import torch
import numpy as np

from classifier import multilabel_classifier
from load_data import *

nepochs = 100
modelpath = None # None if we're training from scratch. Otherwise put the previous model checkpoint.

# Create data loader
trainset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_train.pkl', B=200) # instead of 200
valset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_val.pkl', B=500)
print('Created train and val datasets \n')

unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/unbiased_classes_mapped.pkl', 'rb'))

# Create output directory
outdir = '/n/fs/context-scr/save/stage1'
if not path.isdir(outdir):
    mkdir(outdir)
    print('Created', outdir, '\n')

# Start stage 1 training
start_time = time.time()

# Initialize classifier
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath)

# Start stage 1 training
start_time = time.time()
for i in range(Classifier.epoch, nepochs):

    if i == 60: # Reduce learning rate from 0.1 to 0.01
        Classifier.optimizer = torch.optim.SGD(Classifier.model.parameters(), lr=0.01, momentum=0.9)

    Classifier.train(trainset)
    if i%5 == 0:
        Classifier.save_model('{}/stage1_{}.pth'.format(outdir, i))

    APs, mAP = Classifier.test(valset)
    mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
    print('Validation mAP: all 171 {:.5f}, unbiased 60 {:.5f}'.format(mAP, mAP_unbiased))

    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
