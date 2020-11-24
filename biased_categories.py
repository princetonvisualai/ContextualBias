import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image
import collections
import sys

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

def get_pair_bias(b, z, scores_val, label_to_img, cooccur_thresh):
    if b == z:
        print('Same category, exiting')
        return 0.0

    imgs_b = set(label_to_img[b]) # List of images containing b
    num_imgs_b = len(imgs_b)

    imgs_z = set(label_to_img[z])
    co_occur = imgs_b.intersection(imgs_z)
    if len(co_occur)/len(imgs_b) < cooccur_thresh:
        print('WARNING: Categories {} and {} co-occur infrequently'.format(b, z))
    return bias(b, z, imgs_b, imgs_z, co_occur, scores_val)

def main():
    dataset = sys.argv[1]
	
    # Load class information
    if dataset == 'COCOStuff':
        labels_bias_split = 'labels_train_20.pkl'
        CO_OCCUR_PERCENT = 0.2
        num_categs = 171
    elif dataset == 'AwA':
        labels_bias_split = 'labels_val.pkl'
        CO_OCCUR_PERCENT = 0.2
        num_categs = 85
    elif dataset == 'DeepFashion':
        labels_bias_split = 'labels_val.pkl'
        CO_OCCUR_PERCENT = 0.1
        num_categs = 250
    else:
        labels_bias_split = None
        CO_OCCUR_PERCENT = 0.1
        num_categs = 0
        print('Invalid dataset: {}'.format(dataset))

    # Load files
    labels_val = pickle.load(open('/n/fs/context-scr/{}/{}'.format(dataset, labels_bias_split), 'rb'))
    scores_val = pickle.load(open('/n/fs/context-scr/{}/scores_bias_split.pkl'.format(dataset), 'rb'))
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(dataset), 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

    # Construct a dictionary where label_to_img[k] contains filenames of images that
    # contain label k. k is in [0-N].
    label_to_img = collections.defaultdict(list)
    for img_name in labels_val.keys():
        idx_list = list(np.nonzero(labels_val[img_name]))
        for idx in idx_list:
            label = int(idx[0])
            label_to_img[label].append(img_name)

    # Compute biases for 20 categories in paper
    original_biased_pairs = pickle.load(open('{}/biased_classes.pkl'.format(dataset), 'rb'))

    if True:
        for pair in original_biased_pairs.items():
            b = humanlabels_to_onehot[pair[0]]
            z = humanlabels_to_onehot[pair[1]]
            pair_bias = get_pair_bias(b, z, scores_val, label_to_img, CO_OCCUR_PERCENT)
            print('({}, {}): {}'.format(pair[0], pair[1], pair_bias))

    # Compute top biased pair for each category and record top 20 most biased category pairs
    if False:
        # Calculate bias and get the most biased category for b 
        biased_pairs = np.zeros((num_categs, 3)) # 2d array with columns b, c, bias
 
        for b in range(num_categs):
            imgs_b = set(label_to_img[b]) # List of images containing b
            num_imgs_b = len(imgs_b) # Number of images containing b

            b_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(b)]
            print('\n{}({}) appears in {}/{} images'.format(b_human, b, num_imgs_b, len(labels_val)))

            # Calculate bias values for categories that co-occur with b more than 10-20% of the times
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
            top_20.append([onehot_to_humanlabels[int(biased_pairs[i,0])], onehot_to_humanlabels[int(biased_pairs[i,1])], biased_pairs[i,2]])

        data = {'top_20': top_20, 'biased_pairs': biased_pairs}
        with open('{}/biased_categories.pkl'.format(dataset), 'wb') as handle:
            pickle.dump(data, handle)

if __name__ == '__main__':
    main()
