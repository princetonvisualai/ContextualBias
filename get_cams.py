import numpy as np
import PIL
import torch
import sys
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from classifier import multilabel_classifier
from load_data import *

###
# Referenced from:
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
###

def get_heatmap(CAM_map, img):
    CAM_map = cv2.resize(CAM_map, (img.shape[0], img.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * CAM_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap + np.float32(img)
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    return heatmap

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape
    output_cam = torch.Tensor(0, 7, 7)#.cuda()
    for idx in class_idx:
        cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - cam.min()
        cam_img = cam / cam.max()
        output_cam = torch.cat((output_cam, cam_img.unsqueeze(0)), 0)

    return output_cam

def main():
    dataset = sys.argv[1]

    # Open image
    img_path = '/n/fs/visualai-scr/Data/Coco/2014data/train2014/COCO_train2014_000000002560.jpg'
    img_name = img_path.split('/')[-1][:-4]
    original_img = Image.open(img_path).convert('RGB')

    outdir = '{}/CAMs/'.format(dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print('Saving CAMs in directory {}'.format(outdir))

    # Get image class labels
    img_labels = pickle.load(open('/n/fs/context-scr/{}/labels_train.pkl'.format(dataset), 'rb'))
    if img_path in img_labels:
        class_labels = img_labels[img_path].type('torch.ByteTensor') # torch.cuda.ByteTensor
    else:
        img_labels = pickle.load(open('/n/fs/context-scr/{}/labels_val.pkl'.format(dataset), 'rb'))
        if img_path in img_labels:
            class_labels = img_labels[img_path].type('torch.ByteTensor') # torch.cuda.ByteTensor
        else:
            print('No labels found for image {}'.format(img_path))
            class_labels = torch.zeros(1)
    class_labels = torch.flatten(torch.nonzero(class_labels))

    # Get image CAMs
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    classifier_features = []
    def hook_classifier_features(module, input, output):
        classifier_features.append(output)

    modelpath = '/n/fs/context-scr/{}/save/stage1/stage1_4.pth'.format(dataset)
    classifier = multilabel_classifier(torch.device('cpu'), torch.float32, modelpath=modelpath) # cuda
    classifier.model = classifier.model.to(device=classifier.device, dtype=classifier.dtype)

    classifier.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
    classifier_params = list(classifier.model.parameters())
    classifier_softmax_weight = classifier_params[-2].squeeze(0)

    img = transform(original_img)
    norm_img = normalize(img)
    norm_img = norm_img.to(device=classifier.device, dtype=classifier.dtype)
    norm_img = torch.unsqueeze(img, 0)
    _, x = classifier.forward(norm_img)

    CAMs = returnCAM(classifier_features[0][0].unsqueeze(0), classifier_softmax_weight, class_labels)
    CAMs = CAMs.cpu().detach().numpy()

    # Save CAM heatmap
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(dataset), 'rb'))
    onehot_to_humanlabels = {v: k for k,v in humanlabels_to_onehot.items()}

    img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
    class_labels = class_labels.cpu().detach().numpy()
    for i in range(len(class_labels)): 
        heatmap = get_heatmap(CAMs[i], img)
        plt.figure()
        plt.imshow(heatmap)
        plt.axis('off')
        plt.title(onehot_to_humanlabels[class_labels[i]])
        plt.savefig('{}/{}_{}.png'.format(outdir, img_name, class_labels[i]))
        plt.show()

if __name__ == '__main__':
    main()
