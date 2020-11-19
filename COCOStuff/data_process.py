import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image

# Load labels
labels_txt = open('labels.txt', "r")
labels_txt = labels_txt.read().split('\n')
humanlabels = {}
for i in range(1, 183):
    humanlabels[i] = labels_txt[i].split(' ', 1)[1]

# Remove 'bad' classes based on https://github.com/nightrome/cocostuff/blob/master/labels.md
bad_classes = ['street sign', 'hat', 'shoe', 'eye glasses', 'plate', 'mirror',
    'window', 'desk', 'door', 'blender', 'hair brush']
humanlabels_171 = [humanlabels[i] for i in range(1, 183) if humanlabels[i] not in bad_classes]

# Create a dictionary that maps human-readable labels to [0-170]
humanlabels_to_onehot = {}
for i in range(171):
    humanlabels_to_onehot[humanlabels_171[i]] = i
with open('humanlabels_to_onehot.pkl', 'wb+') as handle:
    pickle.dump(humanlabels_to_onehot, handle)
print('Saved humanlabels_to_onehot.pkl')

# Create a list of annotation file names
anno_val = sorted(glob.glob('/n/fs/visualai-scr/Data/cocostuff/dataset/annotations/val2017/*.png'))
anno_train = sorted(glob.glob('/n/fs/visualai-scr/Data/cocostuff/dataset/annotations/train2017/*.png'))
anno = anno_val + anno_train

# Create a list of image file names
val = sorted(glob.glob('/n/fs/visualai-scr/Data/Coco/2014data/val2014/*.jpg'))
train = sorted(glob.glob('/n/fs/visualai-scr/Data/Coco/2014data/train2014/*.jpg'))

print('anno_train {}, anno_val {}, train {}, val {}\n'.format(len(anno_train), len(anno_val), len(train), len(val)))

start_time = time.time()

# Process COCO-2014 validation set labels:
# 1. Remove class 'unlabeled'
# 2. Replace COCO-2014 labels that only have things annotations with
#    COCO-2017 things and stuff annotations
# 3. One-hot encode to [0-170]
if True:
    count = 0
    labels = {}
    for file in val:

        # COCO-2014 validation images can be in COCO-2017 train or validation
        anno_train_file = file.replace('/n/fs/visualai-scr/Data/Coco/2014data/val2014/COCO_val2014_',
            '/n/fs/visualai-scr/Data/cocostuff/dataset/annotations/train2017/')
        anno_train_file = anno_train_file.replace('jpg', 'png')
        anno_val_file = anno_train_file.replace('train2017', 'val2017')

        # Open the correct things+stuff annotation file
        if anno_train_file in anno:
            anno_image = Image.open(anno_train_file)
        else:
            anno_image = Image.open(anno_val_file)

        # Process the COCO-2017 things+stuff annotations
        label = list(np.unique(np.array(anno_image)).astype(np.int16))
        if 255 in label:
            label.remove(255) # Remove class 'unlabeled'
        label = [humanlabels[k+1] for k in label] # Convert to human-readable labels
        label = [s for s in label if s not in bad_classes] # Remove bad labels
        label = [humanlabels_to_onehot[s] for s in label] # Map labels to [0-170]
        label_onehot = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=171)
        label_onehot = label_onehot.sum(dim=0).float()
        labels[file] = label_onehot # Save the one-hot encoded label

        count += 1
        if count%1000 == 0:
            print(count, time.time()-start_time)

    print('Finished processing {} val labels'.format(len(labels)))
    with open('labels_val.pkl', 'wb+') as handle:
       pickle.dump(labels, handle)

# Process COCO-2014 train set labels:
# 1. Remove 'unlabeled' classes
# 2. Replace COCO-2014 labels that only have things annotations with
#    COCO-2017 things and stuff annotations
# 3. One-hot encode to [0-170]
if False:
    count = 0
    labels = {}
    for file in train:

        # COCO-2014 train images can be in COCO-2017 train or validation
        anno_train_file = file.replace('/n/fs/visualai-scr/Data/Coco/2014data/train2014/COCO_train2014_',
            '/n/fs/visualai-scr/Data/cocostuff/dataset/annotations/train2017/')
        anno_train_file = anno_train_file.replace('jpg', 'png')
        anno_val_file = anno_train_file.replace('train2017', 'val2017')

        # Open the correct things+stuff annotation file
        if anno_train_file in anno:
            anno_image = Image.open(anno_train_file)
        else:
            anno_image = Image.open(anno_val_file)

        # Process the COCO-2017 things+stuff annotations
        label = list(np.unique(np.array(anno_image)).astype(np.int16))
        if 255 in label:
       	    label.remove(255) # Remove class 'unlabeled'
        label = [humanlabels[k+1] for k in label] # Convert to human-readable labels
        label = [s for s in label if s not in bad_classes] # Remove bad labels
        label = [humanlabels_to_onehot[s] for s in label] # Map labels to [0-170]
        label_onehot = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=171)
        label_onehot = label_onehot.sum(dim=0).float()
        labels[file] = label_onehot # Save the one-hot encoded label

        count += 1
        if count%1000 == 0:
            print(count, time.time()-start_time)

    print('Finished processing {} train labels'.format(len(labels)))
    with open('labels_train.pkl', 'wb+') as handle:
        pickle.dump(labels, handle)
