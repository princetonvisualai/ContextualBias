import pickle
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform
import collections

class COCOStuff(Dataset):
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

        return X, y

class COCOStuff_ID(Dataset):
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

def create_dataset(dataset, labels='labels_train.pkl', B=32, baseline1=False, baseline2=False, biased_classes_mapped=None):

    img_labels = pickle.load(open(labels, 'rb'))
    img_paths = list(img_labels.keys())

    if baseline1:
        list_of_b = list(biased_classes_mapped.keys()) # list of one hot for biased categ b

        # remove co-occuring labels 
        for i, img_path in enumerate(img_labels):

            img_label = img_labels[img_path] # tensor

            idx = np.nonzero(img_label[list_of_b]).numpy() # return list of indexes for the biased_classes_mapped keys that are contained within the img
            
            for i in idx: # for each label that exists in this image from the top K biased apirs
                b = list_of_b[i[0]] # get label b
                img_label[biased_classes_mapped[b]] = 0 # set label b's corresponding label c = 0
    
    if baseline2:
        list_of_b = list(biased_classes_mapped.keys())
        label_to_img = collections.defaultdict(list)
        for img_name in img_paths:

            idx_list = list(np.nonzero(img_labels[img_name]))
            for idx in idx_list:
                label = int(idx[0])
                label_to_img[label].append(img_name)

        # for each biased category, get the img names of co-occuring instances
        imgs_to_delete = set()
        for b in list_of_b:

            imgs_b = set(label_to_img[b]) # List of images containing b
            c = biased_classes_mapped[b]
            imgs_c = set(label_to_img[c])

            co_occur = imgs_b.intersection(imgs_c)
            for img in co_occur:
                imgs_to_delete.add(img)

        # delete the co-occuring image names 
        [img_labels.pop(str(key)) for key in imgs_to_delete] 
        # reset image paths 
        img_paths = list(img_labels.keys())
        

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if labels == 'labels_train.pkl':
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        shuffle = True
    else:
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
        shuffle = False

    dset = dataset(img_paths, img_labels, transform)

    loader = DataLoader(dset, batch_size=B, shuffle=shuffle, num_workers=1)

    return loader
