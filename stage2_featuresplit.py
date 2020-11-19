import pickle
import time
from os import path, makedirs
import torch
import numpy as np
import sys

from classifier import multilabel_classifier
from load_data import *

dataset = sys.argv[1]
nepochs = int(sys.argv[2])

# Feature Split Constant
FS_CONST = 1024
modelpath = '/n/fs/context-scr/{}/save/stage1/stage1_4.pth'.format(dataset)

print('Start stage2 feature-split training from {}'.format(modelpath))
outdir = '{}/save/stage2_featuresplit'.format(dataset)
if not path.isdir(outdir):
    makedirs(outdir)
print('Model parameters will be saved in {}'.format(outdir))

weight = pickle.load(open('/n/fs/context-scr/{}/weight_train.pkl'.format(dataset), 'rb'))
weight = torch.Tensor(weight).cuda()
biased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/biased_classes_mapped.pkl'.format(dataset), 'rb'))

# Create data loader
trainset = create_dataset(dataset, labels='labels_train.pkl'.format(dataset), B=100)
valset = create_dataset(dataset, labels='labels_val.pkl'.format(dataset), B=500)
print('Created train and val datasets \n')

# Start stage 2 training
start_time = time.time()
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
Classifier.epoch = 0
Classifier.optimizer = torch.optim.SGD(Classifier.model.parameters(), lr=0.01, momentum=0.9)

xs_prev_ten = []
for epoch in range(nepochs):

    # Specialized train()
    train_loss = 0
    Classifier.model = Classifier.model.to(device=Classifier.device, dtype=Classifier.dtype)
    for i, (images, labels) in enumerate(trainset):

        # Identify exclusive instances and separate the batch into exclusive and non-exclusive
        exclusive_list = []
        for b in biased_classes_mapped.keys():
            exclusive_list.append(np.logical_and(labels[:,b]==1, labels[:,biased_classes_mapped[b]]==0))
        exclusive = torch.stack(exclusive_list).sum(0) > 0

        images_non = images[~exclusive].to(device=Classifier.device, dtype=Classifier.dtype)
        images_exc = images[exclusive].to(device=Classifier.device, dtype=Classifier.dtype)
        labels_non = labels[~exclusive].to(device=Classifier.device, dtype=Classifier.dtype)
        labels_exc = labels[exclusive].to(device=Classifier.device, dtype=Classifier.dtype)

        # Update parameters with non-exclusive samples (co-occur or neither b nor c appears)
        if (~exclusive).sum() > 0:
            Classifier.optimizer.zero_grad()
            _, x_non = Classifier.forward(images_non)
            out_non = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_non)))
            criterion = torch.nn.BCEWithLogitsLoss()
            loss_non = criterion(out_non, labels_non)
            loss_non.backward()
            Classifier.optimizer.step()

            # Keep track of xs
            xs_prev_ten.append(x_non[:, FS_CONST:].detach())
            if len(xs_prev_ten) > 10:
                xs_prev_ten.pop(0)

            l_non = loss_non.item()
        else:
            l_non = 'NA'

        # Update parameters with exclusive samples
        if exclusive.sum() > 0:
            Classifier.optimizer.zero_grad()
            _, x_exc = Classifier.forward(images_exc)

            # Replace the second half of the features with xs_mean
            if len(xs_prev_ten) > 0:
                xs_mean = torch.cat(xs_prev_ten).mean(0)
                x_exc[:, FS_CONST:] = xs_mean.detach()

            out_exc = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_exc)))
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
            loss_exc = criterion(out_exc, labels_exc)
            loss_exc.backward()
            Classifier.model.fc.weight.grad[:, FS_CONST:] = 0.
            Classifier.optimizer.step()
            '''
            if (Classifier.model.fc.weight.grad[:, 1024:] != 0.).sum() > 0:
                print('Warning: Ws was updated (Ws.grad is non-zero)')
            '''
            assert not (Classifier.model.fc.weight.grad[:, FS_CONST:] != 0.).sum() > 0

            l_exc = loss_exc.item()
        else:
            l_exc = 'NA'

        if (i+1)%5 == 0: # CHANGE BACK TO 100
            print('Training epoch {} [{}|{}] non-exclusive({}/{}) {}, exclusive({}/{}) {}'.format(Classifier.epoch, i+1, len(trainset), (~exclusive).sum(), len(exclusive), l_non, (exclusive).sum(), len(exclusive),  l_exc), flush=True)
    
    if (epoch+1) % 5 == 0:
        print('Saving model')
        Classifier.save_model('{}/stage2_{}.pth'.format(outdir, Classifier.epoch))
    Classifier.epoch += 1
    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
