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
#
# Example usage:
# python get_cams.py --modelpath $MODELPATH --img_ids 535811 430054 554674
#
# --modelpath: path to the model to visualize
# --img_ids: COCOStuff image IDs (use the Explore tool on the COCO dataset website)
###

def get_heatmap(CAM_map, img):
    CAM_map = cv2.resize(CAM_map, (img.shape[0], img.shape[1]))
    CAM_map = CAM_map - np.min(CAM_map)
    CAM_map = CAM_map / np.max(CAM_map)
    CAM_map = 1.0 - CAM_map # make sure colormap is not reversed
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

def returnCAM_featuresplit(feature_conv, weight_softmax, class_labels, device, split=1024):
    feature_conv_o = feature_conv[:,:split,:,:]
    feature_conv_s = feature_conv[:,split:,:,:]
    bz, nc, h, w = feature_conv.shape
    output_cam = torch.Tensor(0, 7, 7).to(device=device)
    for idx in class_labels:
        cam_o = torch.mm(weight_softmax[idx][:split].unsqueeze(0), feature_conv_o.reshape((split, h*w)))
        cam_s = torch.mm(weight_softmax[idx][split:].unsqueeze(0), feature_conv_s.reshape((nc-split, h*w)))
        cam_o = cam_o.reshape(h, w)
        cam_s = cam_s.reshape(h, w)
        cam_o = cam_o - cam_o.min()
        cam_s = cam_s - cam_s.min()
        cam_o_img = cam_o / cam_o.max()
        cam_s_img = cam_s / cam_s.max()
        output_cam = torch.cat([output_cam, cam_o_img.unsqueeze(0), cam_s_img.unsqueeze(0)], dim=0)
    return output_cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--img_ids', type=int, nargs='+', default=0)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--featuresplit', default=False, action="store_true")
    parser.add_argument('--split', type=int, default=1024)
    parser.add_argument('--coco2014_images', type=str, default=None)
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
        img_path = '{}/train2014/COCO_train2014_{:012d}.jpg'.format(arg['coco2014_images'], img_id)
        img_name = img_path.split('/')[-1][:-4]
        if not os.path.exists(img_path):
            # Try searching in val set
            img_path = '{}/val2014/COCO_val2014_{:012d}.jpg'.format(arg['coco2014_images'], img_id)
            img_name = img_path.split('/')[-1][:-4]
            if not os.path.exists(img_path):
                print('WARNING: Could not find img {}'.format(img_id), flush=True)
                continue
        original_img = Image.open(img_path).convert('RGB')

        if arg['outdir'] != None:
            outdir = '{}/{}'.format(arg['outdir'], img_id)
        else:
            outdir = str(img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('Processing img {}'.format(img_id), flush=True)

        # Get image class labels
        img_labels = pickle.load(open('COCOStuff/labels_train.pkl', 'rb'))
        if img_path in img_labels:
            if torch.cuda.device_count() > 0:
                class_labels = img_labels[img_path].type('torch.cuda.ByteTensor')
            else:
                class_labels = img_labels[img_path].type('torch.ByteTensor')
        else:
            img_labels = pickle.load(open('COCOStuff/labels_test.pkl', 'rb'))
            if img_path in img_labels:
                if torch.cuda.device_count() > 0:
                    class_labels = img_labels[img_path].type('torch.cuda.ByteTensor')
                else:
                    class_labels = img_labels[img_path].type('torch.ByteTensor')
            else:
                print('No labels found for image {}'.format(img_path), flush=True)
                class_labels = torch.zeros(1)
        class_labels = torch.flatten(torch.nonzero(class_labels))

        classifier_features.clear()
        img = transform(original_img)
        norm_img = normalize(img)
        norm_img = norm_img.to(device=classifier.device, dtype=classifier.dtype)
        norm_img = norm_img.unsqueeze(0)
        x = classifier.forward(norm_img)

        if arg['featuresplit']:
            CAMs = returnCAM_featuresplit(classifier_features[0], classifier_softmax_weight, class_labels, arg['device'], split=arg['split'])
        else:
            CAMs = returnCAM(classifier_features[0], classifier_softmax_weight, class_labels, arg['device'])
        CAMs = CAMs.detach().cpu().numpy()

        # Save CAM heatmap
        humanlabels_to_onehot = pickle.load(open('COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
        onehot_to_humanlabels = {v: k for k,v in humanlabels_to_onehot.items()}

        img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
        class_labels = class_labels.cpu().detach().numpy()
        if arg['featuresplit']:
            for i in range(len(class_labels)):
                heatmap_o = get_heatmap(CAMs[2*i], img)
                heatmap_s = get_heatmap(CAMs[2*i+1], img)

                fig = plt.figure()
                fig_o = fig.add_subplot(121)
                fig_o.imshow(heatmap_o)
                fig_o.axis('off')
                fig_o.set_title('{} ({})'.format(onehot_to_humanlabels[class_labels[i]], 'Wo'))

                fig_s = fig.add_subplot(122)
                fig_s.imshow(heatmap_s)
                fig_s.axis('off')
                fig_s.set_title('{} ({})'.format(onehot_to_humanlabels[class_labels[i]], 'Ws'))
                humanlabel = onehot_to_humanlabels[class_labels[i]].replace(' ', '+')
                plt.savefig('{}/{}_{}.png'.format(outdir, img_name, humanlabel))
                plt.show()
                plt.close()
        else:
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
