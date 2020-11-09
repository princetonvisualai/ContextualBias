import pickle
import time
import os
import torch
import numpy as np

from classifier import multilabel_classifier
from load_data import *

nepochs = 100
modelpath = '/n/fs/context-scr/save/stage1/stage1_99.pth'
outdir = 'save/stage2_weightedloss'
if not os.path.exists(outdir):
    os.makedirs(outdir)
print('Start stage2 weighted loss baseline training from {}'.format(modelpath))
print('Model parameters will be saved in {}'.format(outdir))

weight = pickle.load(open('/n/fs/context-scr/weight_train.pkl', 'rb'))
weight = torch.Tensor(weight).cuda()
biased_classes_mapped = pickle.load(open('/n/fs/context-scr/biased_classes_mapped.pkl', 'rb')

# Create data loader
trainset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_train.pkl', B=200) # instead of 200
valset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_val.pkl', B=500)
print('Created train and val datasets \n')

# Initialize classifier
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
Classifier.epoch = 0
Classifier.optimizer = torch.optim.SGD(Classifier.model.parameters(), lr=0.01, momentum=0.9)
loss_weight = 10.0

# Start stage 2 training
start_time = time.time()
for epoch in range(Classifier.epoch, nepochs):

    # Specialized train()
    train_loss = 0
    Classifier.model = Classifier.model.to(device=Classifier.device, dtype=Classifier.dtype)
    for i, (images, labels) in enumerate(trainset):
        
        # Identify exclusive instances and separate the batch into exclusive and non-exclusive
        exclusive_list = []
        for b in biased_classes_mapped.keys():
            exclusive_list.append(np.logical_and(labels[:,b]==1, labels[:,biased_classes_mapped[b]]==0))
        print(exclusive_list)
        exclusive = torch.stack(exclusive_list).sum(0) > 0
        print(exclusive)

        images_non = images[~exclusive].to(device=Classifier.device, dtype=Classifier.dtype)
        images_exc = images[exclusive].to(device=Classifier.device, dtype=Classifier.dtype)
        labels_non = labels[~exclusive].to(device=Classifier.device, dtype=Classifier.dtype)
        labels_exc = labels[exclusive].to(device=Classifier.device, dtype=Classifier.dtype)

        # Get loss for exclusive samples
        if exclusive.sum() > 0:
            Classifier.optimizer.zero_grad()
            _, x_exc = Classifier.forward(images_exc)
            out_exc = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_exc)))
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
            loss_exc = loss_weight * criterion(out_exc, labels_exc)
            
        # Get loss for non-exclusive samples (co-occur or neither b nor c appears)
        if (~exlusive).sum() > 0:
            Classifier.optimizer.zero_grad()
            _, x_non = Classifier.forward(images_non)
	    out_non = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_non)))
	    criterion = torch.nn.BCEWithLogitsLoss()
	    loss_non = criterion(out_non, labels_non)

	# Use total loss for batch (exclusive + non-exclusive) and perform optimization step
	loss = loss_exc + loss_non
	loss.backward()
	Classifier.optimizer.step()
	l = loss.item()

	if (i+1) % 100 == 0:
	    print('Training epoch {} [{}|{}] non-exclusive({}/{}), exclusive({}/{}) {}'.format(Classifier.epoch, i+1, len(trainset), (~exclusive).sum(), len(exclusive), (exclusive).sum(), len(exclusive), l), flush=True)

    if (epoch+1) % 5 == 0:
        Classifier.save_model('{}/stage1_{}.pth'.format(outdir, i))
    Classifier.epoch += 1

    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
