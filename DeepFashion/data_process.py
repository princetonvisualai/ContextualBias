import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image

datadir = '/n/fs/visualai-scr/Data/DeepFashion/Category and Attribute Prediction Benchmark'

# Load labels
attributes_txt = open('{}/Anno_coarse/list_attr_cloth.txt'.format(datadir))
attributes_txt = attributes_txt.read().split('\n')
attributes_txt = attributes_txt[2:-1] # first two lines contain headers and number of attrs, last line is empty

# Create a dictionary that maps human-readable labels to [0-1000], but don't save because we only need the top 250
humanlabels = []
humanlabels_to_onehot = {}
for i in range(1000):
    humanlabel = ''.join(i for i in attributes_txt[i] if not i.isdigit()).rstrip()
    humanlabels.append(humanlabel)
    humanlabels_to_onehot[humanlabel] = i

# Create a dictionary mapping filenames to labels
list_attr_img = open('{}/Anno_coarse/list_attr_img.txt'.format(datadir))
list_attr_img = list_attr_img.read().split('\n')
list_attr_img = list_attr_img[2:-1] # first two lines contain headers and number of images, last line is empty

print('Mapping filenames to labels')
start_time = time.time()
img_to_label = {}
for count,anno in enumerate(list_attr_img):
    anno = anno.split()
    img = anno[0]
    img = '{}/Img/img/{}'.format(datadir, img)
    label = [int(i) for i in anno[1:]]
    label = [(i + 1) // 2 for i in label]
    label = torch.LongTensor(label).float()
    img_to_label[img] = label
    
    if count%1000 == 0:
        print(count, time.time()-start_time)
print('Done')

# Create a list of train, val, test, image file names
eval_split_txt = open('{}/Eval/list_eval_partition.txt'.format(datadir))
eval_split_txt = eval_split_txt.read().split('\n')
eval_split_txt = eval_split_txt[2:-1]
val = []
train = []
test = []
for line in eval_split_txt:
    filename, split = line.split()
    filename = '{}/Img/img/{}'.format(datadir, filename)
    if split == 'train':
        train.append(filename)
    elif split == 'val':
        val.append(filename)
    elif split == 'test':
        test.append(filename)
    else:
        print('Unknown split: {}'.format(filename))
print('train {}, val {}, test {}\n'.format(len(train), len(val), len(test)))

# Get top 250 categories in train set
attr_count = torch.zeros(1000)
for file in train:
    attr_count += img_to_label[file]
sorted_attr = torch.argsort(attr_count)

top_250 = [] # store onehot indices of top 250 labels for easy access
for i in range(250):
    onehot = sorted_attr[-(i+1)]
    top_250.append(int(onehot))
top_250.sort() # make sure indices are in increasing order

humanlabels_to_onehot_250 = {}
for i in range(250):    
    onehot = top_250[i]
    humanlabel = humanlabels[onehot]
    humanlabels_to_onehot_250[humanlabel] = i 

# Save top 250 most common labels in train set
with open('humanlabels_to_onehot.pkl', 'wb+') as handle:
    pickle.dump(humanlabels_to_onehot_250, handle)
print('Saved top 250 train labels in humanlabels_to_onehot.pkl')

# Process DeepFashion validation set labels
if True:
    count = 0
    labels = {}
    for file in val:
        label_onehot_1000 = img_to_label[file]
        label_onehot_250 = label_onehot_1000[top_250]
        labels[file] = label_onehot_250 # Save the one-hot encoded label
        count += 1

    print('Finished processing {} val labels'.format(len(labels)))
    with open('labels_val.pkl', 'wb+') as handle:
       pickle.dump(labels, handle)

# Process DeepFashion train set labels
if True: 
    count = 0
    labels = {} 
    for file in train:
        label_onehot_1000 = img_to_label[file]
        label_onehot_250 = label_onehot_1000[top_250]
        labels[file] = label_onehot_250 # Save the one-hot encoded label        
        count += 1

    print('Finished processing {} train labels'.format(len(labels)))
    with open('labels_train.pkl', 'wb+') as handle:
        pickle.dump(labels, handle)

# Process DeepFashion test set labels
if True:
    count = 0
    labels = {}
    for file in test:
        label_onehot_1000 = img_to_label[file]
        label_onehot_250 = label_onehot_1000[top_250]
        labels[file] = label_onehot_250 # Save the one-hot encoded label                                                                                       
        count += 1

    print('Finished processing {} test labels'.format(len(labels)))
    with open('labels_test.pkl', 'wb+') as handle:                                                                                                            
       pickle.dump(labels, handle)    
        
