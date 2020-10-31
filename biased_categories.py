import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image
import collections

# Return bias value, given categories b, z, lists of images with these categories
# (imgs_b, imgs_z), a list of images where b and z co-occur (co-ooccur), and a
# dictionary of prediction probabilities of the images (scores_val)
# co-occur is passed in just to reduce computation of finding the intersection of sets again.
def bias(b, z, imgs_b, imgs_z, co_occur, scores_val):
    b_with_z_imgs = co_occur # Ib AND Iz
    b_without_z_imgs = imgs_b.difference(imgs_z) # Ib \ Iz
    num_b_with_z_imgs = len(b_with_z_imgs)
    num_b_without_z_imgs = len(b_without_z_imgs)

    p_with = 0
    p_without = 0
    for i in b_with_z_imgs:
        p_with += scores_val[i][b]
    for i in b_without_z_imgs:
        p_without += scores_val[i][b]

    bias_val = (p_with/num_b_with_z_imgs)/(p_without/num_b_without_z_imgs)

    return bias_val

# Load files
labels_val = pickle.load(open('labels_val.pkl', 'rb'))
scores_val = pickle.load(open('scores_val.pkl', 'rb'))
humanlabels_to_onehot = pickle.load(open('../humanlabels_to_onehot.pkl', 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# c should co-occur with b at least 10% of the times b appears
CO_OCCUR_PERCENT = 0.1

# Construct a dictionary whre label_to_img[k] contains filenames of images that
# contain label k. k is in [0-170].
label_to_img = collections.defaultdict(list)
for img_name in labels_val.keys():
    idx_list = list(np.nonzero(labels_val[img_name]))
    for idx in idx_list:
        label = int(idx[0])
        label_to_img[label].append(img_name)

# Calculate bias and get the most biased category for b
num_categs = 171
biased_pairs = np.zeros((num_categs, 3)) # 2d array with columns b, c, bias

for b in range(num_categs):

    imgs_b = set(label_to_img[b]) # List of images containing b
    num_imgs_b = len(imgs_b) # Number of images containing b

    b_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(b)]
    print('\n{}({}) appears in {}/{} images'.format(b_human, b, num_imgs_b, len(labels_val)))

    # Calculate bias values for categories that co-occur with b more than 10% of the times
    biases_b = np.zeros(num_categs) # Array containing bias value of (b, z)
    for z in range(num_categs):
        if z == b:
            continue

        imgs_z = set(label_to_img[z])
        co_occur = imgs_b.intersection(imgs_z)
        if len(co_occur)/len(imgs_b) > CO_OCCUR_PERCENT:
            biases_b[z] = bias(b, z, imgs_b, imgs_z, co_occur, scores_val)

    # Identify c that has the highest bias for b
    if np.sum(biases_b) != 0:
        c = np.argmax(biases_b)
        biased_pairs[b] = [b, c, biases_b[c]]

    c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(c)]
    print('b {}({}), c {}({}), bias {}'.format(b_human, b, c_human, c, biases_b[c]))


# Get top 20 biased categories
top_20_idx = np.argsort(biased_pairs[:,2])[-20:]
top_20 = []
for i in top_20_idx:
    top_20.append([onehot_to_labels[int(biased_pairs[i,0])], onehot_to_labels[int(biased_pairs[i,1])], biased_pairs[i,2]])

data = {'top_20': top_20, 'biased_pairs': biased_pairs}
with open('biased_categories.pkl', 'wb') as handle:
    pickle.dump(data, handle)
