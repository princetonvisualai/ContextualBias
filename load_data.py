import pickle
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform

class Dataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=T.ToTensor()):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert('RGB')
        X = self.transform(img)
        y = self.img_labels[ID]

        return X, y, ID

def create_dataset(dataset, labels_path, biased_classes_mapped, B=100, train=True, removeclabels=False, removecimages=False, removeximages=False, splitbiased=False):

    img_labels = pickle.load(open(labels_path, 'rb'))
    img_paths = sorted(list(img_labels.keys()))

    # Strong baseline - remove co-occuring labels
    if removeclabels:
        for i, img_path in enumerate(img_labels):
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                if img_labels[img_path][b] == 1:
                    img_labels[img_path][c] = 0

    # Strong baseline - remove co-occuring images
    if removecimages:
        remove_img_paths = []
        for i, img_path in enumerate(img_labels):
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                if (img_labels[img_path][b] == 1) and (img_labels[img_path][c] == 1):
                    remove_img_paths.append(img_path)
                    break

        print('Removing {} co-occur images from {} total training images'.format(len(remove_img_paths), len(img_labels)), flush=True)
        for remove_img_path in remove_img_paths:
            del img_labels[remove_img_path]
            img_paths.remove(remove_img_path)
        print('{}/{} training images remaining'.format(len(img_paths), len(img_labels)), flush=True)

    # Strong baseline - split biased category into exclusive and co-occuring
    if splitbiased:
        nclasses = len(list(img_labels.values())[0])
        biased_classes_list = sorted(list(biased_classes_mapped.keys()))
        for i, img_path in enumerate(img_labels):
            label = img_labels[img_path]
            newlabel = torch.cat((label, torch.zeros(20)))
            for k in range(len(biased_classes_list)):
                b = biased_classes_list[k]
                c = biased_classes_mapped[b]

                # If b and c co-occur, make b label 0 and N+k label 1
                # so as to separate exclusive and co-occur labels
                if (label[b]==1) and (label[c]==1):
                    newlabel[b] = 0
                    newlabel[nclasses + k] = 1

            # Replace the N-D label with new (N+20)-D label
            img_labels[img_path] = newlabel

    # Common from here
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        random_resize = True
        if random_resize:
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        shuffle = True
    else:
        center_crop = True
        if center_crop:
            transform = T.Compose([
               T.Resize(256),
               T.CenterCrop(224),
               T.ToTensor(),
               normalize
            ])
        else: # To use ten-crop, also change the test function in classifier.py
            transform = T.Compose([
                T.Resize(256),
                T.TenCrop(224),
                T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                normalize
            ])
        shuffle = False

    dset = Dataset(img_paths, img_labels, transform)

    loader = DataLoader(dset, batch_size=B, shuffle=shuffle, num_workers=1)

    return loader

# Calculate weights used in the feature-splitting method
def calculate_featuresplit_weight(labels_path, nclasses, biased_classes_mapped, alpha_min=3):

    labels = pickle.load(open(labels_path, 'rb'))

    w = torch.ones(nclasses)
    greater_than_alpha_min = 0
    less_than_alpha_min = 0
    for b in biased_classes_mapped.keys():
        c = biased_classes_mapped[b]
        exclusive = 0; cooccur = 0
        for key in labels.keys():
            if labels[key][b]==1 and labels[key][c]==1:
                cooccur += 1
            elif labels[key][b]==1 and labels[key][c]==0:
                exclusive += 1
        alpha = np.sqrt(cooccur/exclusive)
        if alpha > alpha_min:
            greater_than_alpha_min += 1
            w[b] = alpha
        else:
            less_than_alpha_min += 1
            w[b] = alpha_min
            print('b {:2d}: alpha {:.4f} replaced with {}'.format(b, alpha, alpha_min))

    print('Greater than alpha_min: {}'.format(greater_than_alpha_min), flush=True)
    print('Less than alpha_min: {}'.format(less_than_alpha_min), flush=True) 

    return w

# Calculate weights used in the class-balancing method
def calculate_classbalancing_weight(labels_path, nclasses, biased_classes_mapped, beta=0.99):

    labels = pickle.load(open(labels_path, 'rb'))

    w = torch.ones(nclasses, 3) # columns: exclusive, cooccur, other
    for b in biased_classes_mapped.keys():
        c = biased_classes_mapped[b]
        exclusive = 0; cooccur = 0; other = 0
        for key in labels.keys():
            if labels[key][b]==1 and labels[key][c]==1:
                cooccur += 1
            elif labels[key][b]==1 and labels[key][c]==0:
                exclusive += 1
            else:
                other += 1

        # Calculate weight (1-beta)/(1-beta^n) for each group
        cooccur_weight = (1-beta)/(1-np.power(beta, cooccur))
        exclusive_weight = (1-beta)/(1-np.power(beta, exclusive))
        other_weight = (1-beta)/(1-np.power(beta, other))

        # Have the smallest weight be 1 and the rest proportional to it
        # so as to balance it with other categories that have weight 1
        # instead of having the three weights sum to 1
        sum_weight = np.min([exclusive_weight, cooccur_weight, other_weight])

        # Appropriately normalize the weights
        exclusive_weight /= sum_weight
        cooccur_weight /= sum_weight
        other_weight /= sum_weight

        # Save the weights
        w[b, 0] = exclusive_weight
        w[b, 1] = cooccur_weight
        w[b, 2] = other_weight

    return w
