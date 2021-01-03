import pickle, argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default=None)
parser.add_argument('--labels', type=str, default='/n/fs/context-scr/labels_unrel.pkl')
parser.add_argument('--batchsize', type=int, default=170)
parser.add_argument('--nclasses', type=int, default=171)
parser.add_argument('--hs', type=int, default=2048)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--dtype', default=torch.float32)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load utility files
humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))

# Create dataloader
testset = create_dataset(None, arg['labels'], None, B=arg['batchsize'], train=False)

# Load model
classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'], hidden_size=arg['hs'])

# Do inference with the model
labels_list, scores_list, val_loss_list = classifier.test(testset)

# Calculate AP for car, bus, skateboard
APs = []
for category in ['car', 'bus', 'skateboard']:
    k = humanlabels_to_onehot[category]
    AP = average_precision_score(labels_list[:,k], scores_list[:,k])
    APs.append(AP)
    print('AP ({}): {:.2f}'.format(category, AP*100.))

# Calculate mAP
mAP = np.nanmean(APs)
print('\nmAP (3 classes): {:.2f}'.format(mAP*100.))
