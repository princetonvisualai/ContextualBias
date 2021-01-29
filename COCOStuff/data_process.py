import pickle, time, glob, argparse
import torch
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, default='labels_txt')
parser.add_argument('--humanlabels_to_onehot', type=str, default='COCOStuff/humanlabels_to_onehot.pkl')
parser.add_argument('--cocostuff_annotations', type=str, default='Data/cocostuff/dataset/annotations')
parser.add_argument('--coco2014_images', type=str, default='Data/Coco/2014data')
parser.add_argument('--labels_val', type=str, default='COCOStuff/labels_test.pkl')
parser.add_argument('--labels_train', type=str, default='COCOStuff/labels_train.pkl')
parser.add_argument('--biased_classes', type=str, default='COCOStuff/biased_classes.pkl')
parser.add_argument('--biased_classes_mapped', type=str, default='COCOStuff/biased_classes_mapped.pkl')
parser.add_argument('--unbiased_classes_mapped', type=str, default='COCOStuff/unbiased_classes_mapped.pkl')
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load labels
labels_txt = open('COCOStuff/labels.txt', "r")
labels_txt = labels_txt.read().split('\n')
humanlabels = {}
for i in range(1, 183):
    humanlabels[i] = labels_txt[i].split(' ', 1)[1]

# Remove 'bad' classes based on https://github.com/nightrome/cocostuff/blob/master/labels.md
bad_classes = ['street sign', 'hat', 'shoe', 'eye glasses', 'plate', 'mirror',
    'window', 'desk', 'door', 'blender', 'hair brush']
humanlabels_171 = [x for x in humanlabels.values() if x not in bad_classes]

# Create a dictionary that maps human-readable labels to [0-170]
humanlabels_to_onehot = {}
for i in range(171):
    humanlabels_to_onehot[humanlabels_171[i]] = i
with open(arg['humanlabels_to_onehot'], 'wb+') as handle:
    pickle.dump(humanlabels_to_onehot, handle)
print('Saved humanlabels_to_onehot.pkl')

# Create a list of annotation file names
anno_val = sorted(glob.glob('{}/val2017/*.png'.format(arg['cocostuff_annotations'])))
anno_train = sorted(glob.glob('{}/train2017/*.png'.format(arg['cocostuff_annotations'])))
anno = anno_val + anno_train

# Create a list of image file names
val = sorted(glob.glob('{}/val2014/*.jpg'.format(arg['coco2014_images'])))
train = sorted(glob.glob('{}/train2014/*.jpg'.format(arg['coco2014_images'])))

print('anno_train {}, anno_val {}, train {}, val {}\n'.format(len(anno_train), len(anno_val), len(train), len(val)))

start_time = time.time()

# Process COCO-2014 validation set labels:
# 1. Remove class 'unlabeled'
# 2. Replace COCO-2014 labels that only have things annotations with
#    COCO-2017 things and stuff annotations
# 3. One-hot encode to [0-170]
if False:
    count = 0
    labels = {}
    for file in val:

        # COCO-2014 validation images can be in COCO-2017 train or validation
        anno_train_file = file.replace('{}/val2014/COCO_val2014_'.format(arg['coco2014_images']),
            '{}/train2017/'.format(arg['cocostuff_annotations']))
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
    with open(arg['labels_val'], 'wb+') as handle:
       pickle.dump(labels, handle)

# Process COCO-2014 train set labels:
# 1. Remove 'unlabeled' classes
# 2. Replace COCO-2014 labels that only have things annotations with
#    COCO-2017 things and stuff annotations
# 3. One-hot encode to [0-170]
if True:
    count = 0
    labels = {}
    for file in train:

        # COCO-2014 train images can be in COCO-2017 train or validation
        anno_train_file = file.replace('{}/train2014/COCO_train2014_'.format(arg['coco2014_images']),
            '{}/train2017/'.format(arg['cocostuff_annotations']))
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
    with open(arg['labels_train'], 'wb+') as handle:
        pickle.dump(labels, handle)

# 20 most biased classes identified in the original paper
biased_classes = {}
biased_classes['cup'] = 'dining table'
biased_classes['wine glass'] = 'person'
biased_classes['handbag'] = 'person'
biased_classes['apple'] = 'fruit'
biased_classes['car'] = 'road'
biased_classes['bus'] = 'road'
biased_classes['potted plant'] = 'vase'
biased_classes['spoon'] = 'bowl'
biased_classes['microwave'] = 'oven'
biased_classes['keyboard'] = 'mouse'
biased_classes['skis'] = 'person'
biased_classes['clock'] = 'building-other'
biased_classes['sports ball'] = 'person'
biased_classes['remote'] = 'person'
biased_classes['snowboard'] = 'person'
biased_classes['toaster'] = 'ceiling-other'
biased_classes['hair drier'] = 'towel'
biased_classes['tennis racket'] = 'person'
biased_classes['skateboard'] = 'person'
biased_classes['baseball glove'] = 'person'
with open(arg['biased_classes'], 'wb+') as handle:
    pickle.dump(biased_classes, handle)

biased_classes_mapped = dict((humanlabels_to_onehot[key], humanlabels_to_onehot[value]) for (key, value) in biased_classes.items())
with open(arg['biased_classes_mapped'], 'wb+') as handle:
    pickle.dump(biased_classes_mapped, handle)

unbiased_classes_mapped = [i for i in list(np.arange(80)) if i not in biased_classes_mapped.keys()]
with open(arg['unbiased_classes_mapped'], 'wb+') as handle:
    pickle.dump(unbiased_classes_mapped, handle)
