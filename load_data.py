import pickle
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform

class Dataset(Dataset): # rename something different from Dataset?
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

def create_dataset(dataset, labels_path, biased_classes_mapped, B=100, train=True, removeclabels=False, removecimages=False, splitbiased=False):

    img_labels = pickle.load(open(labels_path, 'rb'))
    img_paths = sorted(list(img_labels.keys()))

    # Strong baseline - remove co-occuring labels
    if removeclabels:
        for i, img_path in enumerate(img_labels):
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                if img_labels[img_path][b] == 1:
                    img_labels[img_path][c] == 0

    # Strong baseline - remove co-occuring images
    if removecimages:
        remove_img_paths = []
        for i, img_path in enumerate(img_labels):
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                if (img_labels[img_path][b] == 1) and (img_labels[img_path][c] == 1):
                    remove_img_paths.append(img_path)
                    break

        for remove_img_path in remove_img_paths:
            del img_labels[remove_img_path]
            img_paths.remove(remove_img_path)

    # Strong baseline - split biased category into exclusive and co-occuring
    if splitbiased:
        biased_classes_list = sorted(list(biased_classes_mapped.keys()))
        for i, img_path in enumerate(img_labels):
            addlabel = torch.zeros((20))
            for k in range(len(biased_classes_list)):
                b = biased_classes_list[k]
                c = biased_classes_mapped[b]
                label = img_labels[img_path]

                # If b and c co-occur, make b label 0 and N+b label 1
                # so as to separate exclusive and co-occur labels
                if (label[b]==1) and (label[c]==1):
                    label[b] = 0
                    addlabel[k] = 1

            # Replace the N-D label with new (N+20)-D label
            newlabel = torch.cat((label, addlabel))
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
        #transform = T.Compose([
        #    T.Resize(256),
        #    T.CenterCrop(224),
        #    T.ToTensor(),
        #    normalize
        #])
        transform = T.Compose([
            T.Resize(256),
            T.TenCrop(224),
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops]))
        ])
        shuffle = False

    dset = Dataset(img_paths, img_labels, transform)

    loader = DataLoader(dset, batch_size=B, shuffle=shuffle, num_workers=1)

    return loader

# Calculate weights used in the feature-splitting method
def calculate_weight(labels_path, nclasses, biased_classes_mapped, humanlabels_to_onehot):

    labels = pickle.load(open(labels_path, 'rb'))

    w = torch.ones(nclasses)
    for b in biased_classes_mapped.keys():
        exclusive = 0; cooccur = 0
        for key in labels.keys():
            if labels[key][b]==1 and labels[key][biased_classes_mapped[b]]==1:
                cooccur += 1
            elif labels[key][b]==1 and labels[key][biased_classes_mapped[b]]==0:
                exclusive += 1
        alpha = np.sqrt(cooccur/exclusive)
        if alpha > 1:
            w[b] = alpha

    return w
