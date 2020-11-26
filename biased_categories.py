import pickle, glob, collections, time, argparse
import torch
import numpy as np

from classifier import *
from load_data import *

parser = argparse.ArgumentParser()
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
labels_val = pickle.load(open(arg['labels'], 'rb'))
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# Get scores for the 20 split data
loader = create_dataset(COCOStuff, arg['labels'], None, B=arg['batchsize'], train=False)
Classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'])
Classifier.model.cuda()
Classifier.model.eval()

scores_dict = pickle.load(open('scores_dict.pkl', 'rb'))

#scores_dict = {}
#with torch.no_grad():
#    for i, (images, labels, ids) in enumerate(loader):
#        images = images.to(device=Classifier.device, dtype=Classifier.dtype)
#        labels = labels.to(device=Classifier.device, dtype=Classifier.dtype)
#        scores, _ = Classifier.forward(images)
#        scores = torch.sigmoid(scores).squeeze().data.cpu().numpy()
#        for j in range(images.shape[0]):
#            id = ids[j]
#            scores_dict[id] = scores[j]
#with open('scores_dict.pkl', 'wb+') as handle:
#        pickle.dump(scores_dict, handle, protocol=4)

# Construct a dictionary where label_to_img[k] contains filenames of images that
# contain label k. k is in [0-170].
label_to_img = collections.defaultdict(list)
for img_name in labels_val.keys():
    idx_list = list(np.nonzero(labels_val[img_name]))
    for idx in idx_list:
        label = int(idx[0])
        label_to_img[label].append(img_name)

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
    print('\nb {}({}), c {}({})'.format(b, onehot_to_humanlabels[b], c, onehot_to_humanlabels[c]))
    print('bias {:.4f}, co-occur {}, exclusive {}, total {}'.format(biases_b[c], len(co_occur), 
        len(imgs_b)-len(co_occur), len(labels_val)))

# Save all bias values
with open('bias.pkl', 'wb+') as handle:
    pickle.dump(biased_pairs, handle)

# Get top 20 biased categories
top_20 = np.argsort(biased_pairs[:,2])[-20:]
biased_classes = {}
for i in top_20:
    biased_classes[onehot_to_humanlabels[int(biased_pairs[i,0])]] = onehot_to_humanlabels[int(biased_pairs[i,1])]

print('\nTop 20 biased categories:', biased_classes)
with open('biased_classes.pkl', 'wb+') as handle:
    pickle.dump(biased_classes, handle)

# Alternatively, output the K most biased categories identified in the original paper
if True:
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
    biased_classes['toaster'] = 'ceiling-other' # unclear from the paper
    biased_classes['hair drier'] = 'towel'
    biased_classes['tennis racket'] = 'person'
    biased_classes['skateboard'] = 'person'
    biased_classes['baseball glove'] = 'person'

    print('\nTop 20 biased categories from the original paper:', biased_classes)
    with open('biased_classes.pkl', 'wb+') as handle:
        pickle.dump(biased_classes, handle)

# Save other useful files
biased_classes_mapped = dict((humanlabels_to_onehot[key], humanlabels_to_onehot[value]) for (key, value) in biased_classes.items())
with open('biased_classes_mapped.pkl', 'wb+') as handle:
    pickle.dump(biased_classes_mapped, handle)

# Save unbiased object classes (80 - 20 things) used in the appendiix
unbiased_classes_mapped = [i for i in list(np.arange(80)) if i not in biased_classes_mapped.keys()]
with open('unbiased_classes_mapped.pkl', 'wb+') as handle:
    pickle.dump(unbiased_classes_mapped, handle)
