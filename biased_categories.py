import pickle, glob, collections, time, argparse
import torch
import numpy as np

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels', type=str, default='/n/fs/context-scr/COCOStuff/labels_train_20.pkl')
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--nclasses', type=int, default=171)
    parser.add_argument('--cooccur', type=float, default=0.1)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    arg = vars(parser.parse_args())
    print('\n', arg, '\n')

    # Load files
    labels = pickle.load(open(arg['labels'], 'rb'))
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

    # Get scores for the bias split data
    valset = create_dataset(arg['dataset']f, arg['labels'], None, B=arg['batchsize'], train=False)
    Classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'])
    Classifier.model.eval()
    scores_dict = {}
    with torch.no_grad():
        for i, (images, labels, ids) in enumerate(valset):
            images = images.to(device=Classifier.device, dtype=Classifier.dtype),
            labels = labels.to(device=Classifier.device, dtype=Classifier.dtype)
            scores, _ = Classifier.forward(images)
            scores = torch.sigmoid(scores).squeeze().data.cpu().numpy()
            for j in range(images.shape[0]):
                id = ids[j]
                scores_dict[id] = scores[j]

    # Construct a dictionary where label_to_img[k] contains filenames of images that
    # contain label k. k is in [0-N].
    label_to_img = collections.defaultdict(list)
    for img_name in labels.keys():
        idx_list = list(np.nonzero(labels[img_name]))
        for idx in idx_list:
            label = int(idx[0])
            label_to_img[label].append(img_name)
    
    # Compute biases for 20 categories in paper
    original_biased_pairs = pickle.load(open('{}/biased_classes.pkl'.format(arg['dataset']), 'rb'))

    if True:
        for pair in original_biased_pairs.items():
            b = humanlabels_to_onehot[pair[0]]
            z = humanlabels_to_onehot[pair[1]]
            pair_bias = get_pair_bias(b, z, scores_val, label_to_img, CO_OCCUR_PERCENT)
            print('({}, {}): {}'.format(pair[0], pair[1], pair_bias))

    # Compute top biased pair for each category and record top 20 most biased category pairs
    if False:
        # Calculate bias and get the most biased category for b
        biased_pairs = np.zeros((arg['nclasses'], 3)) # 2d array with columns b, c, bias
        for b in range(arg['nclasses']):

            imgs_b = set(label_to_img[b]) # List of images containing b

            # Calculate bias values for categories that co-occur with b more than 10% of the times
            biases_b = np.zeros(arg['nclasses']) # Array containing bias value of (b, z)
            for z in range(arg['nclasses']):
                if z == b:
                    continue

                imgs_z = set(label_to_img[z])
                co_occur = imgs_b.intersection(imgs_z)
                if len(co_occur)/len(imgs_b) > arg['cooccur']:
                    biases_b[z] = bias(b, z, imgs_b, imgs_z, co_occur, scores_dict)

            # Identify c that has the highest bias for b
            if np.sum(biases_b) != 0:
                c = np.argmax(biases_b)
                biased_pairs[b] = [b, c, biases_b[c]]

            c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(c)]
            print('\nb {}({}), c {}({}), bias {}, co-occur {}, exclusive {}, total {}'.format(onehot_to_humanlabels[b], b, onehot_to_humanlabels[c], c, biases_b[c], len(co_occur), len(imgs_b)-len(co_occur), len(labels)))

        # Get top 20 biased categories
        top_20_idx = np.argsort(biased_pairs[:,2])[-20:]
        top_20 = []
        for i in top_20_idx:
            top_20.append([onehot_to_humanlabels[int(biased_pairs[i,0])], onehot_to_humanlabels[int(biased_pairs[i,1])], biased_pairs[i,2]])

        result = {'top_20': top_20, 'biased_pairs': biased_pairs}
        print('\nTop 20 biased categories')
        print(result)
        
if __name__ == '__main__':
    main()