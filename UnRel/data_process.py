import pickle, time, glob, argparse
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='Data/UnRel')
parser.add_argument('--humanlabels_to_onehot', type=str, default='COCOStuff/humanlabels_to_onehot.pkl')
parser.add_argument('--labels_unrel', type=str, default='UnRel/labels_unrel.pkl')
arg = vars(parser.parse_args())
print('\n', arg, '\n')


# Load COCOStuff labels
humanlabels_to_onehot = pickle.load(open(arg['humanlabels_to_onehot'], 'rb'))

# Shared classes between COCOStuff biased and UnRel
shared_classes = ['car', 'bus', 'skateboard']
shared_classes.append('road') # context category for 'car' and 'bus'
shared_classes.append('person') # context category for 'skateboard'

# Create a list of image file names (test)
start_time = time.time()
annotations = loadmat('{}/annotations.mat'.format(arg['datadir']))['annotations']
object_list = []
if True:
    count = 0
    labels = {}

    # Process images
    for i in range(annotations.shape[0]):
        filename = '{}/images/{}'.format(arg['datadir'], annotations[i][0][0][0][0][0])
        annotation = annotations[i][0][0][0][3] # loadmat creates a deeply nested array
        label = []
        for obj in annotation:
            s = obj[0][0][0][0][0] # loadmat creates a deeply nested array
            object_list.append(s)

            if s in shared_classes and humanlabels_to_onehot[s] not in label:
                label.append(humanlabels_to_onehot[s])

        label_onehot = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=171)
        label_onehot = label_onehot.sum(dim=0).float()
        labels[filename] = label_onehot

        count += 1

    print('Finished processing {} UnRel labels'.format(len(labels)))
    with open(arg['labels_unrel'], 'wb+') as handle:
       pickle.dump(labels, handle)

    print('Objects in UnRel:', set(object_list))
