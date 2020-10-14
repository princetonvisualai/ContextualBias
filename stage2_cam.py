import pickle
import time
from os import path, mkdir

import torch
from torchvision import transforms
import numpy as np

from classifier import multilabel_classifier
from loaddata import *

nepochs = 100
modelpath = 'save/stage1/stage1_23.pth'
outdir = 'save/stage2_cam'
if not path.isdir(outdir):
    mkdir(outdir)
print('Start stage2 CAM training from {}'.format(modelpath))
print('Model parameters will be saved in {}'.format(outdir))

biased_classes_mapped = pickle.load(open('biased_classes_mapped.pkl', 'rb'))

# Create data loader
trainset = create_dataset(COCOStuff, labels='labels_train.pkl', B=64)
valset = create_dataset(COCOStuff, labels='labels_val.pkl', B=500)
print('Created train and val datasets \n')

# Set up the CAM pre-trained network
stage1_net = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
stage1_net.model.cuda()
stage1_net.model.eval()

# CAM utils
def returnCAM(feature_conv, weight_softmax, class_idx):
    #size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        #cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        #cam = cam - np.min(cam)
        cam = cam - cam.min()
        #cam_img = cam / np.max(cam)
        cam_img = cam / cam.max()
        #cam_img = np.uint8(255 * cam_img)
        cam_img = cam_img * 255
        #output_cam.append(cv2.resize(cam_img, size_upsample))
        output_cam.append(cam_img)
    return output_cam

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

# Start stage 2 training
start_time = time.time()
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
Classifier.epoch = 0
Classifier.optimizer = torch.optim.SGD(Classifier.model.parameters(), lr=0.01, momentum=0.9)

xs_prev_ten = []
for epoch in range(nepochs):

    if epoch > 1:
        break

    # Specialized train()
    train_loss = 0
    Classifier.model = Classifier.model.to(device=Classifier.device, dtype=Classifier.dtype)
    for i, (images, labels) in enumerate(trainset):

        if i > 1:
            break

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
                #Classifier_features.append(output.data.cpu().numpy())
                Classifier_features.append(output)
            Classifier.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
            Classifier_params = list(Classifier.model.parameters())
            #Classifier_softmax_weight = np.squeeze(Classifier_params[-2].data.numpy())
            Classifier_softmax_weight = Classifier_params[-2].squeeze(0)

            Classifier.optimizer.zero_grad()
            _, x_coc = Classifier.forward(images_coc)
            out_coc = Classifier.model.fc(Classifier.model.dropout(Classifier.model.relu(x_coc)))
            #logit = Classifier.model(images_coc)

            criterion = torch.nn.BCEWithLogitsLoss()
            l_coc_bce = criterion(out_coc, labels_coc)

            print('Classifier_features', len(Classifier_features))
            print('Classifier_features[0]', Classifier_features[0].shape)

            CAMs = []
            for k in range(len(cooccur)):
                #print(k)
                #print('Classifier_features[0][k].unsqueeze(0)', Classifier_features[0][k].unsqueeze(0).shape)
                #print('Classifier_softmax_weight', Classifier_softmax_weight.shape)
                #print('cooccur_classes[k]', cooccur_classes[k])
                #CAMs.append(returnCAM(Classifier_features[0][k][np.newaxis, :], Classifier_softmax_weight, cooccur_classes[k]))
                CAMs.append(returnCAM(Classifier_features[0][k].unsqueeze(0), Classifier_softmax_weight, cooccur_classes[k]))
            print('CAMs', len(CAMs))
            print('CAMs[0]', len(CAMs[0]), CAMs[0])


            # Get CAM from the pre-trained stage 1 network
            stage1_features = []
            def hook_stage1_feature(module, input, output):
                #stage1_features.append(output.data.cpu().numpy())
                stage1_features.append(output)
            stage1_net.model._modules['resnet'].layer4.register_forward_hook(hook_stage1_feature)
            stage1_params = list(stage1_net.model.parameters())
            #stage1_softmax_weight = np.squeeze(stage1_params[-2].data.numpy())
            stage1_softmax_weight = np.squeeze(stage1_params[-2])
            _ = stage1_net.model(images_coc)

            print('stage1_features', len(stage1_features))
            print('stage1_features[0]', stage1_features[0].shape)

            CAMs_pretrained = []
            for k in range(len(cooccur)):
                #CAMs_pretrained.append(returnCAM(stage1_features[0][k][np.newaxis, :], stage1_softmax_weight, cooccur_classes[k]))
                CAMs_pretrained.append(returnCAM(stage1_features[0][k].unsqueeze(0), stage1_softmax_weight, cooccur_classes[k]))
            print('CAMs_pretrained', len(CAMs_pretrained))
            print('CAMs_pretrained[0]', len(CAMs_pretrained[0]), CAMs_pretrained[0])

            

            # l_o = np.mean(CAMs[0]/255. * CAMs[1]/255.)
            # l_r = np.mean(np.abs(np.array(CAMs)/255. - np.array(CAMs_pretrained)/255.))

            loss_coc = l_coc_bce
            # loss_coc = 0.1*l_o + 0.01*l_r + l_coc_bce
            loss_coc.backward()
            Classifier.optimizer.step()
            l_coc = loss_coc.item()

        ### All other cases


        #if i%100 == 0:
            #print('Training epoch {} [{}|{}] non-exclusive({}/{}) {}, exclusive({}/{}) {}'.format(Classifier.epoch, i+1, len(trainset), (~exclusive).sum(), len(exclusive), l_non, (exclusive).sum(), len(exclusive),  l_exc), flush=True)

    # Classifier.save_model('{}/stage2_{}.pth'.format(outdir, Classifier.epoch))
    Classifier.epoch += 1
    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
