import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat

# Data directory
datadir = '/n/fs/visualai-scr/Data/UnRel'

# Load COCOStuff labels
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))

# Shared classes between COCOStuff biased and UnRel
shared_classes = ['car', 'bus', 'skateboard']
shared_classes.append('road') # context category for 'car' and 'bus'
shared_classes.append('person') # context category for 'skateboard'

# Create a list of image file names (test)
start_time = time.time()
annotations = loadmat('/n/fs/visualai-scr/Data/UnRel/annotations.mat')['annotations']
object_list = []
if True:
    count = 0
    labels = {}

    # Process images
    for i in range(annotations.shape[0]):
        filename = '/n/fs/visualai-scr/Data/UnRel/images/{}'.format(annotations[i][0][0][0][0][0])
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
    with open('labels_unrel.pkl', 'wb+') as handle:
       pickle.dump(labels, handle)

    print('Objects in UnRel:', set(object_list))
