import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image
import collections

# FUNCTIONS
# get prediction probability of image i for category b from the multi-label classifier
def p(i, b):
  img_name = i[-31:-4]
  score_filename = path+'scores/'+img_name+'.pkl'
  score = pickle.load(open(score_filename, 'rb'))
  prob = score[b]
  return prob

# get bias score given label b, co-occuring label z, imgs_b (list of imgs with label b), imgs_z (list of images with label z), and co_occur (list of images with b and z). Co-occur is passed in just to reduce computation of finding the intersection of sets again. Returns bias value
def bias(b, z, imgs_b, imgs_z, co_occur):
    b_with_z_imgs = co_occur # Ib AND Iz
    b_without_z_imgs = imgs_b.difference(imgs_z) # Ib \ Iz
    num_b_with_z_imgs = len(b_with_z_imgs)
    num_b_without_z_imgs = len(b_with_z_imgs)
    p_with = 0
    p_without = 0
    for i in b_with_z_imgs:
      p_with += p(i, b)
    for i in b_without_z_imgs:
      p_without += p(i, b)
    
    bias_val = (num_b_with_z_imgs * p_with)/(num_b_without_z_imgs * p_without)
    return bias_val
    


path = '/n/fs/context-scr/nmeister/'

# return a mapping from label (word) to number (ie: 'bench' --> 13))
humanlabels_to_onehot = pickle.load(open(path+'humanlabels_to_onehot.pkl', 'rb'))
# return a dictionary mapping image names to a 171 Dim tensor where if value==1, then the label excists in that image
# '/n/fs/visualai-scr/Data/Coco/2014data/train2014/COCO_train2014_000000000009.jpg' --> 171 Dim Tensor
labels_train = pickle.load(open(path+'labels_train.pkl', 'rb'))

# Number that labels must co-occur by (10%)
CO_OCCUR_PERCENT = 0.1

# Load labels
labels_txt = open(path+'labels.txt', "r")
labels_txt = labels_txt.read().split('\n')
humanlabels = {}
for i in range(1, 183):
    humanlabels[i] = labels_txt[i].split(' ', 1)[1]
    # human labels is dictionary containing 182 keys that map the number to the label "1 --> person"

# construct dictionary mapping category to the image name. 
# label_to_img[1] contains all the img file names that contain label 1 (from 171 Dim Tensor from labels_train)
label_to_img = collections.defaultdict(list)

# for each image update the label_to_image dictionary
for img_name in labels_train.keys():
  idx_list = list(np.nonzero(labels_train[img_name]))
  for idx in idx_list:
    label = int(idx[0])
    label_to_img[label].append(img_name)


num_categs = len(label_to_img.keys()) # 171
biased_pairs = np.zeros((num_categs, 3)) # 2d array with row as category label. Each col is b, c, bias_val

for b in range(num_categs):
    print('categ: ', b)
    # list of images containing b
    imgs_b = set(label_to_img[b])
    # number of images containing label b
    num_imgs_b = len(imgs_b)
    # array containing bias value of label b and label z (indexed by row)
    biases_b = np.zeros(num_categs)

    # find categories that co-occur > 10% of the times: overlap of imgs containing b and imgs containing other category
    for z in range(num_categs):
      print('z: ', z)
      if z==b:
        biases_b[z] = 0
        continue
      
      imgs_z = set(label_to_img[z])
      co_occur = imgs_b.intersection(imgs_z)
      if len(co_occur)/len(imgs_b) > CO_OCCUR_PERCENT:
        # calc biases
        biases_b[z] = bias(b, z, imgs_b, imgs_z, co_occur)
    # if something co occurs with this class, update the biased pairs array
    if np.sum(biases_b) != 0:
      c = np.argmax(biases_b)
      biased_pairs[b] = [b, c, biases_b[c]]
    data = {'biased_pairs': biased_pairs}
    with open(path+'biased_pairs/biased_categs_'+str(b)+'.pkl', 'wb') as handle:
      pickle.dump(data, handle)

# map the 171 integers to their word label
onehot_to_labels = dict((y,x) for x,y in humanlabels_to_onehot.items())


# get top 20 image and image pairs
top_20_idx = np.argsort(biased_pairs[:, 2])[-20:]
top_20 = []
for i in top_20_idx:
  top_20.append([onehot_to_labels[int(biased_pairs[i, 0])], onehot_to_labels[int(biased_pairs[i, 1])], biased_pairs[i, 2]])


data = {'top_20': top_20, 'biased_pairs': biased_pairs}
with open(path+'biased_categs.pkl', 'wb') as handle:
    pickle.dump(data, handle)
