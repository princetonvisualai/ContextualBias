import pickle, glob, collections, time, argparse
import torch
import numpy as np

from classifier import multilabel_classifier
from load_data import *

# Return bias value, given categories b, z, lists of images with these categories
# (imgs_b, imgs_z), a list of images where b and z co-occur (co-ooccur), and a
# dictionary of prediction probabilities of the images (scores_val)
# co-occur is passed in just to reduce computation of finding the intersection of sets again.
def bias(b, z, imgs_b, imgs_z, co_occur, scores_dict):
    b_with_z_imgs = co_occur # Ib AND Iz
    b_without_z_imgs = imgs_b.difference(imgs_z) # Ib \ Iz
    num_b_with_z_imgs = len(b_with_z_imgs)
    num_b_without_z_imgs = len(b_without_z_imgs)

    p_with = 0
    p_without = 0
    for i in b_with_z_imgs:
        p_with += scores_dict[i][b]
    for i in b_without_z_imgs:
        p_without += scores_dict[i][b]
    
    if num_b_with_z_imgs > 0 and p_without > 0 and num_b_without_z_imgs > 0:
        bias_val = (p_with/num_b_with_z_imgs) / (p_without/num_b_without_z_imgs)
    else:
        bias_val = 0

    return bias_val

def get_pair_bias(b, z, scores_dict, label_to_img, cooccur_thresh):
    if b == z:
        print('Same category, exiting')
        return 0.0

    imgs_b = set(label_to_img[b]) # List of images containing b
    num_imgs_b = len(imgs_b)

    imgs_z = set(label_to_img[z])
    co_occur = imgs_b.intersection(imgs_z)
    if len(co_occur)/len(imgs_b) < cooccur_thresh:
        print('WARNING: Categories {} and {} co-occur infrequently ({:4.2f})'.format(b, z, len(co_occur)/len(imgs_b)))

    return bias(b, z, imgs_b, imgs_z, co_occur, scores_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels', type=str, default='/n/fs/context-scr/COCOStuff/labels_train_20.pkl')
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--nclasses', type=int, default=171)
    parser.add_argument('--cooccur', type=float, default=0.1)
    parser.add_argument('--precomputed', type=bool, default=False)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    arg = vars(parser.parse_args())
    print('\n', arg, '\n')

    # Load files
    labels_dict = pickle.load(open(arg['labels'], 'rb'))
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

    if arg['precomputed']:
        scores_dict = pickle.load(open('{}/scores_dict.pkl'.format(arg['dataset']), 'rb'))
    else:
        # Get scores for the bias split data
        valset = create_dataset(arg['dataset'], arg['labels'], None, B=arg['batchsize'], train=False)
        classifier = multilabel_classifier(arg['device'], arg['dtype'], nclasses=arg['nclasses'], modelpath=arg['modelpath'])
        classifier.model = classifier.model.to(device=classifier.device, dtype=classifier.dtype)
        classifier.model.eval()
        scores_dict = {}
        with torch.no_grad():
            for i, (images, labels, ids) in enumerate(valset):
                images = images.to(device=classifier.device, dtype=classifier.dtype)
                labels = labels.to(device=classifier.device, dtype=classifier.dtype)

                # Ten crop
                # bs, ncrops, c, h, w = images.size()
                # outputs, _ = classifier.forward(images.view(-1, c, h, w))
                # outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                outputs = classifier.forward(images)
                scores = torch.sigmoid(outputs).squeeze().data.cpu().numpy()
                for j in range(images.shape[0]):
                    id = ids[j]
                    scores_dict[id] = scores[j]
        
        with open('{}/scores_dict.pkl'.format(arg['dataset']), 'wb+') as handle:
            pickle.dump(scores_dict, handle)

    # Construct a dictionary where label_to_img[k] contains filenames of images that
    # contain label k. k is in [0-N].
    label_to_img = collections.defaultdict(list)
    for img_name in labels_dict:
        idx_list = np.where(labels_dict[img_name]>0)[0]
        for label in idx_list:
            label_to_img[label].append(img_name)
    
    # Compute biases for 20 categories in paper
    original_biased_pairs = pickle.load(open('/n/fs/context-scr/{}/biased_classes.pkl'.format(arg['dataset']), 'rb'))
    if True:
        print('Original biased pairs')
        print('\n{:>11} {:>11} {:>8}'.format('b', 'c', 'bias'), flush=True)
        for pair in original_biased_pairs.items():

            b = humanlabels_to_onehot[pair[0]]
            z = humanlabels_to_onehot[pair[1]]
            pair_bias = get_pair_bias(b, z, scores_dict, label_to_img, arg['cooccur'])
            print('{:>11} {:>11} {:8.2f}'.format(pair[0], pair[1], pair_bias))


    # Compute top biased pair for each category and record top 20 most biased category pairs
    if True:
        print('\nBiased category analysis')
        print('\n{:>16} {:>16} {:>8} {:>10} {:>10} {:>10}'.format('b', 'c', 'bias', 'co-occur', 'exclusive', 'co-freq'), flush=True)
        # Calculate bias and get the most biased category for b
        biased_pairs = np.zeros((arg['nclasses'], 3)) # 2d array with columns b, c, bias
        for b in range(arg['nclasses']):

            imgs_b = set(label_to_img[b]) # List of images containing b
            if len(imgs_b) == 0:
                print('{:>11}({:>3}) {:>11}({:>3}) {:>8} {:>10} {:>10} {:>10}'.format(onehot_to_humanlabels[b], b, 'N/A', ' ', '---', 0, len(imgs_b), 0), flush=True)
                continue

            # Calculate bias values for categories that co-occur with b more than 10% of the times
            biases_b = np.zeros(arg['nclasses']) # Array containing bias value of (b, z)
            for z in range(arg['nclasses']):
                if z == b:
                    continue

                imgs_z = set(label_to_img[z])
                co_occur = imgs_b.intersection(imgs_z)
                if len(co_occur)/len(imgs_b) >= arg['cooccur']:
                    bias_b_z = bias(b, z, imgs_b, imgs_z, co_occur, scores_dict)
                    biases_b[z] = bias_b_z

            # Identify c that has the highest bias for b
            if np.sum(biases_b) != 0:
                c = np.argmax(biases_b)
                biased_pairs[b] = [b, c, biases_b[c]]
                imgs_c = set(label_to_img[c])
                co_occur = imgs_b.intersection(imgs_c)

                c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(c)]
                print('{:>11}({:>3}) {:>11}({:>3}) {:>8.2f} {:>10} {:>10} {:>10.2f}'.format(onehot_to_humanlabels[b], b, onehot_to_humanlabels[c], c, biases_b[c], len(co_occur), len(imgs_b)-len(co_occur), len(co_occur)/len(imgs_b)), flush=True)

        # Get top 20 biased categories
        top_20_idx = np.argsort(biased_pairs[:,2])[-20:]
        top_20 = []
        for i in top_20_idx:
            top_20.append([onehot_to_humanlabels[int(biased_pairs[i,0])], onehot_to_humanlabels[int(biased_pairs[i,1])], biased_pairs[i,2]])

        result = {'top_20': top_20, 'biased_pairs': biased_pairs}
        print('\nTop 20 biased categories', flush=True)
        print('\n{:>11} {:>11} {:>8}'.format('b', 'c', 'bias'), flush=True)
        for pair in result['top_20']:
            print('{:>11} {:>11} {:8.2f}'.format(pair[0], pair[1], pair[2]), flush=True)
        
if __name__ == '__main__':
    main()
