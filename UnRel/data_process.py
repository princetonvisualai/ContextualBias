import pickle
import time
import glob
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat

# Data directory
datadir = '/n/fs/visualai-scr/Data/UnRel'

# Create human labels (only 3 shared with biased categories in COCOStuff)
humanlabels = {
    1: 'car',
    2: 'bus',
    3: 'skateboard'
}

# Create a dictionary that maps human-readable labels to [0-2]
humanlabels_to_onehot = {
    'car': 0,
    'bus': 1,
    'skateboard': 2
}
with open('humanlabels_to_onehot.pkl', 'wb+') as handle:
    pickle.dump(humanlabels_to_onehot, handle)
print('Saved humanlabels_to_onehot.pkl')

# Create a list of image file names (test)
start_time = time.time()
annotations = loadmat('/n/fs/visualai-scr/Data/UnRel/annotations.mat')['annotations']
if True:
    count = 0
    labels = {}
    
    # Process image if it contains a label in COCOStuff
    for i in range(annotations.shape[0]):
        filename = annotations[i][0][0][0][0][0]
        annotation = annotations[i][0][0][0][3] # loadmat creates a deeply nested array
        keep = False
        label = []
        for obj in annotation:
            s = obj[0][0][0][0][0] # loadmat creates a deeply nested array
            
            # Keep the image and create a label if it shares classes with COCOStuff
            if s in humanlabels_to_onehot:
                keep = True
                label.append(humanlabels_to_onehot[s])
        if keep:
            label_onehot = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=3)
            label_onehot = label_onehot.sum(dim=0).float()
            labels[filename] = label_onehot

        count += 1

    print('Finished processing {} test labels'.format(len(labels)))
    with open('labels_test.pkl', 'wb+') as handle:
       pickle.dump(labels, handle)

