import numpy as np
import PIL
import torch
import sys
import os
import cv2
import argparse
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

def returnCAM(feature_conv, weight_softmax, class_labels, device):
    bz, nc, h, w = feature_conv.shape # (1, hidden_size, height, width)
    output_cam = torch.Tensor(0, 7, 7).to(device=device)
    for idx in class_labels:
        cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - cam.min()
        cam_img = cam / cam.max()
        output_cam = torch.cat([output_cam, cam_img.unsqueeze(0)], dim=0)
    return output_cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--img_ids', type=int, nargs='+', default=0)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    arg = vars(parser.parse_args())
    print(arg, '\n', flush=True)
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    classifier_features = []
    def hook_classifier_features(module, input, output):
        classifier_features.append(output)

    classifier = multilabel_classifier(device=arg['device'], dtype=arg['dtype'], modelpath=arg['modelpath'])
    classifier.model = classifier.model.to(device=classifier.device, dtype=classifier.dtype)

    classifier.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
    classifier_params = list(classifier.model.parameters())
    classifier_softmax_weight = classifier_params[-2].squeeze(0)

    for img_id in arg['img_ids']:
        # Open image
        img_path = '/n/fs/visualai-scr/Data/Coco/2014data/train2014/COCO_train2014_{:012d}.jpg'.format(img_id)
        img_name = img_path.split('/')[-1][:-4]
        if not os.path.exists(img_path):
            # Try searching in val set
            img_path = '/n/fs/visualai-scr/Data/Coco/2014data/train2014/COCO_train2014_{:012d}.jpg'.format(img_id)
            img_name = img_path.split('/')[-1][:-4]
            if not os.path.exists(img_path):
                print('WARNING: Could not find img {}'.format(img_id), flush=True)
                continue
        original_img = Image.open(img_path).convert('RGB')

        outdir = 'COCOStuff/CAMs/{}'.format(img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('Processing img {}'.format(img_id), flush=True)

        # Get image class labels
        img_labels = pickle.load(open('/n/fs/context-scr/COCOStuff/labels_train.pkl', 'rb'))
        if img_path in img_labels:
            if torch.cuda.device_count() > 0:
                class_labels = img_labels[img_path].type('torch.cuda.ByteTensor')
            else:
                class_labels = img_labels[img_path].type('torch.ByteTensor')
        else:
            img_labels = pickle.load(open('/n/fs/context-scr/COCOStuff/labels_val.pkl', 'rb'))
            if img_path in img_labels:
                if torch.cuda.device_count() > 0:
                    class_labels = img_labels[img_path].type('torch.cuda.ByteTensor')
                else:
                    class_labels = img_labels[img_path].type('torch.ByteTensor')
            else:
                print('No labels found for image {}'.format(img_path), flush=True)
                class_labels = torch.zeros(1)
        class_labels = torch.flatten(torch.nonzero(class_labels))

        img = transform(original_img)
        norm_img = normalize(img)
        norm_img = norm_img.to(device=classifier.device, dtype=classifier.dtype)
        norm_img = norm_img.unsqueeze(0)
        x = classifier.forward(norm_img)

        CAMs = returnCAM(classifier_features[0], classifier_softmax_weight, class_labels, arg['device'])
        CAMs = CAMs.detach().cpu().numpy()

        # Save CAM heatmap
        humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
        onehot_to_humanlabels = {v: k for k,v in humanlabels_to_onehot.items()}

        img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
        class_labels = class_labels.cpu().detach().numpy()
        for i in range(len(class_labels)): 
            heatmap = get_heatmap(CAMs[i], img)
            plt.figure()
            plt.imshow(heatmap)
            plt.axis('off')
            plt.title(onehot_to_humanlabels[class_labels[i]])
            humanlabel = onehot_to_humanlabels[class_labels[i]].replace(' ', '+')
            plt.savefig('{}/{}_{}.png'.format(outdir, img_name, humanlabel))
            plt.show()
            plt.close()

if __name__ == '__main__':
    main()
