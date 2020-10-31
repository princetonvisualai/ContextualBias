import pickle
import glob
import torch
import numpy as np
from PIL import Image

from classifier import multilabel_classifier
from load_data import *

# Set path to the trained model
modelpath = '/n/fs/context-scr/save/stage1/stage1_99.pth'

# Load util files
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/humanlabels_to_onehot.pkl', 'rb'))
labels_train = pickle.load(open('/n/fs/context-scr/labels_train.pkl', 'rb'))

# Load model and set it in evaluation mode
Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, modelpath=modelpath)
Classifier.model.cuda()
Classifier.model.eval()

BATCH_SIZE = 500
valset = create_dataset(COCOStuff_ID, labels='/n/fs/context-scr/labels_val.pkl', B=BATCH_SIZE)
print('Batch size {}, Total number of batches {}'.format(BATCH_SIZE, len(valset)))

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
with open('scores_val.pkl', 'wb') as handle:
    pickle.dump(scores_dict, handle, protocol=4)
