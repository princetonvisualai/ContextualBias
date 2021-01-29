import pickle, time, glob, argparse
import torch
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='Data/AwA2')
parser.add_argument('--classlabels_to_onehot', type=str, default='AwA/classlabels_to_onehot.pkl')
parser.add_argument('--humanlabels_to_onehot', type=str, default='AwA/humanlabels_to_onehot.pkl')
parser.add_argument('--predicate_matrix', type=str, default='AwA/predicate_matrix.pkl')
parser.add_argument('--labels_train', type=str, default='AwA/labels_train.pkl')
parser.add_argument('--labels_test', type=str, default='AwA/labels_test.pkl')
parser.add_argument('--biased_classes', type=str, default='AwA/biased_classes.pkl')
parser.add_argument('--biased_classes_mapped', type=str, default='AwA/biased_classes_mapped.pkl')
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load class labels
classes_txt = open('{}/classes.txt'.format(arg['datadir']), "r")
classes_txt = classes_txt.read().split('\n')
classlabels = {}
for i in range(1, 51):
    classlabels[i] = classes_txt[i-1][7:]

# Load predicate labels
predicates_txt = open('{}/predicates.txt'.format(arg['datadir']), "r")
predicates_txt = predicates_txt.read().split('\n')
humanlabels = {}
for i in range(1, 86):
    humanlabels[i] = predicates_txt[i-1][7:]

# Create a dictionary that maps human-readable class labels to [0-49]
classlabels_to_onehot = {}
for i in range(50):
    classlabels_to_onehot[classlabels[i+1]] = i
with open(arg['classlabels_to_onehot'], 'wb+') as handle:
    pickle.dump(classlabels_to_onehot, handle)
print('Saved classlabels_to_onehot.pkl')

# Create a dictionary that maps human-readable predicate labels to [0-84]
humanlabels_to_onehot = {}
for i in range(85):
    humanlabels_to_onehot[humanlabels[i+1]] = i
with open(arg['humanlabels_to_onehot'], 'wb+') as handle:
    pickle.dump(humanlabels_to_onehot, handle)
print('Saved humanlabels_to_onehot.pkl')

# Create a dictionary mapping onehot class label to predicates
predicate_matrix_txt = open('{}/predicate-matrix-binary.txt'.format(arg['datadir']), "r")
predicate_matrix_txt = predicate_matrix_txt.read().split('\n')
predicate_matrix = np.zeros((50, 85), dtype='uint8')
for i in range(50):
    predicates = predicate_matrix_txt[i].split(' ')
    predicates = [int(j) for j in predicates]
    predicate_matrix[i] = predicates
with open(arg['predicate_matrix'], 'wb+') as handle:
    pickle.dump(predicate_matrix, handle)

# Create a list of image file names (train)
train_classes_txt = open('{}/trainclasses.txt'.format(arg['datadir']), "r")
train_classes_txt = train_classes_txt.read().split('\n')
train = None
for classname in train_classes_txt:
    print(classname)
    if not train:
        train = sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(arg['datadir'], classname)))
    else:
        train.extend(sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(arg['datadir'], classname))))
print('Compiled train image file names')

# Create a list of image file names (test)
test_classes_txt = open('{}/testclasses.txt'.format(arg['datadir']), "r")
test_classes_txt = test_classes_txt.read().split('\n')
test = None
for classname in test_classes_txt:
    print(classname)
    if not test:
        test = sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(arg['datadir'], classname)))
    else:
        test.extend(sorted(glob.glob('{}/JPEGImages/{}/*.jpg'.format(arg['datadir'], classname))))
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
    with open(arg['labels_test'], 'wb+') as handle:
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
    with open(arg['labels_train'], 'wb+') as handle:
       pickle.dump(labels, handle)

# 20 most biased classes identified in the original paper
biased_classes = {}
biased_classes['white'] = 'ground'
biased_classes['longleg'] = 'domestic'
biased_classes['forager'] = 'nestspot'
biased_classes['lean'] = 'stalker'
biased_classes['fish'] = 'timid'
biased_classes['hunter'] = 'big'
biased_classes['plains'] = 'stalker'
biased_classes['nocturnal'] = 'white'
biased_classes['nestspot'] = 'meatteeth'
biased_classes['jungle'] = 'muscle'
biased_classes['muscle'] = 'black'
biased_classes['meat'] = 'fish'
biased_classes['mountains'] = 'paws'
biased_classes['tree'] = 'tail'
biased_classes['domestic'] = 'inactive'
biased_classes['spots'] = 'longleg'
biased_classes['bush'] = 'meat'
biased_classes['buckteeth'] = 'smelly'
biased_classes['slow'] = 'strong'
biased_classes['blue'] = 'coastal'
with open(arg['biased_classes'], 'wb+') as handle:
    pickle.dump(biased_classes, handle)

biased_classes_mapped = dict((humanlabels_to_onehot[key], humanlabels_to_onehot[value]) for (key, value) in biased_classes.items())
with open(arg['biased_classes_mapped'], 'wb+') as handle:
    pickle.dump(biased_classes_mapped, handle)
