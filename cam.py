import pickle
import time
#from os import path, mkdir
import os

import torch
from torchvision import transforms
import numpy as np

from get_cams import returnCAM
from classifier import multilabel_classifier
from load_data import *

nepochs = 100
modelpath = '/n/fs/context-scr/save/stage1/stage1_4.pth'
outdir = 'save/cam'
if not os.path.exists(outdir):
    os.makedirs(outdir)
print('Start CAM training from {}'.format(modelpath))
print('Model parameters will be saved in {}'.format(outdir))

biased_classes_mapped = pickle.load(open('/n/fs/context-scr/biased_classes_mapped.pkl', 'rb'))

# Create data loader
trainset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_train.pkl', B=100)
valset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_val.pkl', B=500)
print('Created train and val datasets \n')

# Set up the CAM pre-trained network
stage1_net = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
stage1_net.model.cuda()
stage1_net.model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

# Start stage 2 training
start_time = time.time()
classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
classifier.epoch = 0
classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(classifier.epoch, nepochs):

    # Specialized training
    classifier.model = classifier.model.to(device=classifier.device, dtype=classifier.dtype)
    for i, (images, labels) in enumerate(trainset):

        # Identify co-occurring instances and separate the batch into co-occurring and non-co-occurring
        cooccur = []
        cooccur_classes = []
        for m in range(labels.shape[0]):
            cooccur_class = None
            for b in biased_classes_mapped.keys():
                if np.logical_and(labels[m,b]==1, labels[m,biased_classes_mapped[b]]==1):
                    cooccur_class = [b, biased_classes_mapped[b]]
            if cooccur_class is not None:
                cooccur.append(m)
                cooccur_classes.append(cooccur_class)

        noncooccur = [x for x in np.arange(images.shape[0]) if x not in cooccur]
        images_coc = images[cooccur].to(device=classifier.device, dtype=classifier.dtype)
        images_non = images[noncooccur].to(device=classifier.device, dtype=classifier.dtype)
        labels_coc = labels[cooccur].to(device=classifier.device, dtype=classifier.dtype)
        labels_non = labels[noncooccur].to(device=classifier.device, dtype=classifier.dtype)

        ### Co-occur (both b and c appears)
        if len(cooccur) > 0:

            # Hook the feature extractor
            classifier_features = []
            def hook_classifier_features(module, input, output):
                classifier_features.append(output)
            classifier.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
            classifier_params = list(classifier.model.parameters())
            classifier_softmax_weight = classifier_params[-2].squeeze(0)

            classifier.optimizer.zero_grad()
            _, x_coc = classifier.forward(images_coc)
            out_coc = classifier.model.fc(classifier.model.dropout(classifier.model.relu(x_coc)))

            criterion = torch.nn.BCEWithLogitsLoss()
            l_coc_bce = criterion(out_coc, labels_coc)

            CAMs = torch.Tensor(0, 2, 7, 7).cuda()
            for k in range(len(cooccur)):
                CAM = returnCAM(classifier_features[0][k].unsqueeze(0), classifier_softmax_weight, cooccur_classes[k])
                CAMs = torch.cat((CAMs, CAM.unsqueeze(0)), 0)

            # Get CAM from the pre-trained stage 1 network
            stage1_features = []
            def hook_stage1_feature(module, input, output):
                stage1_features.append(output)
            stage1_net.model._modules['resnet'].layer4.register_forward_hook(hook_stage1_feature)
            stage1_params = list(stage1_net.model.parameters())
            stage1_softmax_weight = np.squeeze(stage1_params[-2])
            _ = stage1_net.model(images_coc)

            CAMs_pretrained = torch.Tensor(0, 2, 7, 7).cuda()
            for k in range(len(cooccur)):
                CAM_pretrained = returnCAM(stage1_features[0][k].unsqueeze(0), stage1_softmax_weight, cooccur_classes[k])
                CAMs_pretrained = torch.cat((CAMs_pretrained, CAM_pretrained.unsqueeze(0)), 0)

            # CAM maps at 7x7 resolution
            l_o = (CAMs[:,0] * CAMs[:,1]).mean()
            l_r = torch.abs(CAMs - CAMs_pretrained).mean()

            loss_coc = 0.1*l_o + 0.01*l_r + l_coc_bce
            loss_coc.backward()
            classifier.optimizer.step()
            l_coc = loss_coc.item()
        else: 
            l_coc = "NA"

        ### All other cases
        if len(noncooccur) > 0:
            classifier.optimizer.zero_grad()
            _, x_non = classifier.forward(images_non)
            out_non = classifier.model.fc(classifier.model.dropout(classifier.model.relu(x_non)))
            criterion = torch.nn.BCEWithLogitsLoss()
            loss_non = criterion(out_non, labels_non)
            loss_non.backward()
            classifier.optimizer.step()
            l_non = loss_non.item()
        else: 
            l_non = "NA"

        if (i+1)%100 == 0:
            print('Training epoch {} [{}|{}] co-occur({}/{}) {}, other({}/{}) {}'.format(classifier.epoch, i+1, len(trainset), len(cooccur), images.shape[0], l_coc, len(noncooccur), images.shape[0],  l_non), flush=True)

    if (epoch+1) % 5 == 0:
        print('Saving model')
        classifier.save_model('{}/cam_{}.pth'.format(outdir, classifier.epoch))
    classifier.epoch += 1
    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
