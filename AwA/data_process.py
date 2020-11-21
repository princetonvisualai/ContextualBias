import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image

# Data directory
datadir = '/n/fs/visualai-scr/Data/AwA2'

# Load class labels
classes_txt = open('{}/classes.txt'.format(datadir), "r")
classes_txt = classes_txt.read().split('\n')
classlabels = {}
for i in range(1, 51):
    classlabels[i] = classes_txt[i-1][7:]

# Load predicate labels
predicates_txt = open('{}/predicates.txt'.format(datadir), "r")
predicates_txt = predicates_txt.read().split('\n')
humanlabels = {}
for i in range(1, 86):
    humanlabels[i] = predicates_txt[i-1][7:]

# Create a dictionary that maps human-readable class labels to [0-49]
classlabels_to_onehot = {}
for i in range(50):
    classlabels_to_onehot[classlabels[i+1]] = i
with open('classlabels_to_onehot.pkl', 'wb+') as handle:
    pickle.dump(classlabels_to_onehot, handle)
print('Saved classlabels_to_onehot.pkl')

# Create a dictionary that maps human-readable predicate labels to [0-84]
humanlabels_to_onehot = {}
for i in range(85):
    humanlabels_to_onehot[humanlabels[i+1]] = i
with open('humanlabels_to_onehot.pkl', 'wb+') as handle:
    pickle.dump(humanlabels_to_onehot, handle)
print('Saved humanlabels_to_onehot.pkl')

# Create a dictionary mapping onehot class label to predicates
predicate_matrix_txt = open('{}/predicate-matrix-binary.txt'.format(datadir), "r")
predicate_matrix_txt = predicate_matrix_txt.read().split('\n')
predicate_matrix = np.zeros((50, 85), dtype='uint8')
for i in range(50):
    predicates = predicate_matrix_txt[i].split(' ')
    predicates = [int(j) for j in predicates]
    predicate_matrix[i] = predicates
with open('predicate_matrix.pkl', 'wb+') as handle:
    pickle.dump(predicate_matrix, handle)

# Create a list of image file names (train)
train_classes_txt = open('{}/trainclasses.txt'.format(datadir), "r")
train_classes_txt = train_classes_txt.read().split('\n')
train = None
for classname in train_classes_txt:
    print(classname)
    if not train:
        train = sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(datadir, classname)))
    else:
        train.extend(sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(datadir, classname))))
print('Compiled train image file names')

# Create a list of image file names (test)
test_classes_txt = open('{}/testclasses.txt'.format(datadir), "r")
test_classes_txt = test_classes_txt.read().split('\n')
test = None
for classname in test_classes_txt:
    print(classname)
    if not test:
        test = sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(datadir, classname)))
    else:
        test.extend(sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(datadir, classname))))
print('Compiled test image file names')
print('train {}, test {}\n'.format(len(train), len(test)))

start_time = time.time()

# Process AwA2 test set labels:
# 1. Assign the corresponding predicates to each image under a class
# 2. One-hot encode to [0-84]
if True:
    count = 0
    labels = {}
    for file in test:

        # Get the image class and assign the predicate labels
        classlabel = classlabels_to_onehot[file.split('/')[-2]]
        label = predicate_matrix[classlabel]
        label_onehot = torch.LongTensor(label).float()
        labels[file] = label_onehot # Save the one-hot encoded label

        count += 1

    print('Finished processing {} test labels'.format(len(labels)))
    with open('labels_val.pkl', 'wb+') as handle:
       pickle.dump(labels, handle)

# Process AwA2 train set labels:
# 1. Assign the corresponding predicates to each image under a class
# 2. One-hot encode to [0-84]
if True:
    count = 0
    labels = {}
    for file in train:

        # Get the image class and assign the predicate labels
        classlabel = classlabels_to_onehot[file.split('/')[-2]]
        label = predicate_matrix[classlabel]
        label_onehot = torch.LongTensor(label).float()
        labels[file] = label_onehot # Save the one-hot encoded label

        count += 1

    print('Finished processing {} train labels'.format(len(labels)))
    with open('labels_train.pkl', 'wb+') as handle:
       pickle.dump(labels, handle)

