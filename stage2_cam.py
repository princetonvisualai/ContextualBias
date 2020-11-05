import pickle
import time
#from os import path, mkdir
import os

import torch
from torchvision import transforms
import numpy as np

from classifier import multilabel_classifier
from load_data import *

nepochs = 100
modelpath = '/n/fs/context-scr/save/stage1/stage1_99.pth'
outdir = 'save/stage2_cam'
if not os.path.exists(outdir):
    os.makedirs(outdir)
print('Start stage2 CAM training from {}'.format(modelpath))
print('Model parameters will be saved in {}'.format(outdir))

biased_classes_mapped = pickle.load(open('/n/fs/context-scr/biased_classes_mapped.pkl', 'rb'))

# Create data loader
trainset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_train.pkl', B=64)
valset = create_dataset(COCOStuff, labels='/n/fs/context-scr/labels_val.pkl', B=500)
print('Created train and val datasets \n')

# Set up the CAM pre-trained network
stage1_net = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
stage1_net.model.cuda()
stage1_net.model.eval()

# CAM utils
def returnCAM(feature_conv, weight_softmax, class_idx):
    #size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = torch.Tensor(0, 7, 7).cuda()
    for idx in class_idx:
        cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - cam.min()
        cam_img = cam / cam.max()
        #output_cam.append(cv2.resize(cam_img, size_upsample))
        output_cam = torch.cat((output_cam, cam_img.unsqueeze(0)), 0)

    return output_cam

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

# Start stage 2 training
start_time = time.time()
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
Classifier.epoch = 0
Classifier.optimizer = torch.optim.SGD(Classifier.model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(Classifier.epoch, nepochs):

    # Specialized training
    Classifier.model = Classifier.model.to(device=Classifier.device, dtype=Classifier.dtype)
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
        images_coc = images[cooccur].to(device=Classifier.device, dtype=Classifier.dtype)
        images_non = images[noncooccur].to(device=Classifier.device, dtype=Classifier.dtype)
        labels_coc = labels[cooccur].to(device=Classifier.device, dtype=Classifier.dtype)
        labels_non = labels[noncooccur].to(device=Classifier.device, dtype=Classifier.dtype)

        ### Co-occur (both b and c appears)
        if len(cooccur) > 0:

            # Hook the feature extractor
            Classifier_features = []
            def hook_classifier_features(module, input, output):
                Classifier_features.append(output)
            Classifier.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
            Classifier_params = list(Classifier.model.parameters())
            Classifier_softmax_weight = Classifier_params[-2].squeeze(0)

            Classifier.optimizer.zero_grad()
            _, x_coc = Classifier.forward(images_coc)
            out_coc = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_coc)))

            criterion = torch.nn.BCEWithLogitsLoss()
            l_coc_bce = criterion(out_coc, labels_coc)

            CAMs = torch.Tensor(0, 2, 7, 7).cuda()
            for k in range(len(cooccur)):
                CAM = returnCAM(Classifier_features[0][k].unsqueeze(0), Classifier_softmax_weight, cooccur_classes[k])
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
            Classifier.optimizer.step()
            l_coc = loss_coc.item()

        ### All other cases
        if len(noncooccur) > 0:
            Classifier.optimizer.zero_grad()
            _, x_non = Classifier.forward(images_non)
            out_non = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_non)))
            criterion = torch.nn.BCEWithLogitsLoss()
            loss_non = criterion(out_non, labels_non)
            loss_non.backward()
            Classifier.optimizer.step()
            l_non = loss_non.item()

        if (i+1)%100 == 0:
            print('Training epoch {} [{}|{}] co-occur({}/{}) {}, other({}/{}) {}'.format(Classifier.epoch, i+1, len(trainset), len(cooccur), images.shape[0], l_coc, len(noncooccur), images.shape[0],  l_non), flush=True)

    if epoch+1 % 5 == 0:
        Classifier.save_model('{}/stage2_{}.pth'.format(outdir, Classifier.epoch))
    Classifier.epoch += 1
    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
