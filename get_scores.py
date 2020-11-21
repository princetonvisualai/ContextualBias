import pickle
import glob
import torch
import numpy as np
from PIL import Image
import sys

from classifier import multilabel_classifier
from load_data import *

# Example usage:
# python get_scores.py COCOStuff

dataset = sys.argv[1]

# Load bias split
BATCH_SIZE = 500
if dataset == 'COCOStuff':
    labels_bias_split = 'labels_train_20.pkl'
    num_categs = 171
    learning_rate = 0.1
elif dataset == 'AwA':
    labels_bias_split = 'labels_val.pkl'
    num_categs = 85
    learning_rate = 0.01
elif dataset == 'DeepFashion':
    labels_bias_split = 'labels_val.pkl'
    num_categs = 250
    learning_rate = 0.1 # Double-check learning rate
else:
    labels_bias_split = None
    num_categs = 0
    learning_rate = 0.01
    print('Invalid dataset: {}'.format(dataset))

valset = create_dataset(dataset, labels=labels_bias_split, B=BATCH_SIZE)
print('Batch size {}, Total number of batches {}'.format(BATCH_SIZE, len(valset)))

# Set path to the trained model
modelpath = '/n/fs/context-scr/{}/save/stage1/stage1_4.pth'.format(dataset)

# Load model and set it in evaluation mode
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, 
                                   num_categs=num_categs, learning_rate=learning_rate, 
                                   modelpath=modelpath)
Classifier.model.cuda()
Classifier.model.eval()

# Go through the dataset and save scores
scores_dict = {}
with torch.no_grad():
    for i, (images, labels, ids) in enumerate(valset):

        # Get scores
        images, labels = images.to(device=Classifier.device, dtype=Classifier.dtype), labels.to(device=Classifier.device, dtype=Classifier.dtype)
        scores, _ = Classifier.forward(images)
        scores = torch.sigmoid(scores).squeeze().data.cpu().numpy()
        
        # Add scores to the dictionary
        for j in range(images.shape[0]):
            id = ids[j]
            scores_dict[id] = scores[j]

print('scores_dict', len(scores_dict))
with open('{}/scores_bias_split.pkl'.format(dataset), 'wb') as handle:
    pickle.dump(scores_dict, handle, protocol=4)
