import pickle, argparse
import numpy as np
import torch
from scipy.spatial.distance import cosine

from classifier import multilabel_classifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCOStuff')
parser.add_argument('--modelpath', type=str, default=None)
parser.add_argument('--hs', type=int, default=2048)
parser.add_argument('--nclasses', type=int, default=171)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--dtype', default=torch.float32)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load utility files
biased_classes_mapped = pickle.load(open('{}/biased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
humanlabels_to_onehot = pickle.load(open('{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# Load model
classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'], hidden_size=arg['hs'])

# Define Wo and Ws
if True:
    Wo = classifier.model.resnet.fc.weight[:, :1024].data.cpu().numpy()
    Ws = classifier.model.resnet.fc.weight[:, 1024:].data.cpu().numpy()
else:
    np.random.seed(1)
    indices = np.random.choice(2048, size=1024, replace=False)
    Wo = classifier.model.resnet.fc.weight[:, indices].data.cpu().numpy()
    Ws = classifier.model.resnet.fc.weight[:, ~indices].data.cpu().numpy()

# For each b, calculate the cosine similarity between Wo and Ws
similarity_list = []
for b in biased_classes_mapped.keys():
    c = biased_classes_mapped[b]
    similarity = 1 - cosine(Wo[b], Ws[b])
    # Manual calculation yields same results
    # similarity = np.dot(Wo[b],Ws[b])/(np.linalg.norm(Wo[b])*np.linalg.norm(Ws[b]))
    similarity_list.append(similarity)

    print(b, onehot_to_humanlabels[b], similarity)

print('\nMean', np.mean(similarity_list))
