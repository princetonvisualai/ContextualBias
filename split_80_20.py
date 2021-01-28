import pickle, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--labels_train', type=str, default=None)
parser.add_argument('--labels_train_20', type=str, default=None)
parser.add_argument('--labels_train_80', type=str, default=None)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load the processed train labels
labels_train = pickle.load(open(arg['labels_train'], 'rb'))

# Do a 80-20 split of train
N = len(list(labels_train.keys()))
N_80 = int(N*0.8)

# Select images to be added to the 80 split
np.random.seed(1234)
keys = list(labels_train.keys())
inds_80 = np.random.choice(N, N_80, replace=False)
keys_80 = np.array(keys)[inds_80]
keys_20 = np.delete(keys, inds_80)

# Create smaller label dictionaries
labels_train_80 = {k: labels_train[k] for k in keys_80}
labels_train_20 = {k: labels_train[k] for k in keys_20}

with open(arg['labels_train_80'], 'wb') as handle:
    pickle.dump(labels_train_80, handle, protocol=4)
with open(arg['labels_train_20'], 'wb') as handle:
    pickle.dump(labels_train_20, handle, protocol=4)
