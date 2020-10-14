import pickle
import torch
import numpy as np

datatype = 'train'
labels = pickle.load(open('labels_{}.pkl'.format(datatype), 'rb'))
biased_classes = pickle.load(open('biased_classes.pkl', 'rb'))
biased_classes_mapped = pickle.load(open('biased_classes_mapped.pkl', 'rb'))
humanlabels_to_onehot = pickle.load(open('humanlabels_to_onehot.pkl', 'rb'))

# Calculate alphas
alphas = {}
for b in biased_classes_mapped.keys():
    exclusive_positive = 0
    cooccur_positive = 0

    for key in labels.keys():
        label = labels[key]

        if label[b]==1 and label[biased_classes_mapped[b]]==1:
            cooccur_positive += 1
        elif label[b]==1 and label[biased_classes_mapped[b]]==0:
            exclusive_positive += 1

    alpha = np.sqrt(cooccur_positive/exclusive_positive)

    # Print how many images are in each set
    b_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(b)]
    c_human = list(humanlabels_to_onehot.keys())[list(humanlabels_to_onehot.values()).index(biased_classes_mapped[b])]
    print()
    print('{} - {}'.format(b_human, c_human))
    print('  co-occur: {} images'.format(cooccur_positive))
    print('  exclusive: {} images'.format(exclusive_positive))
    print('  alpha: {}'.format(alpha))
    alphas[b_human] = alpha

# Calculate and save weights
w = torch.ones(171)
for key in alphas.keys():
    index = humanlabels_to_onehot[key]
    if alphas[key] > 0:
        w[index] = alphas[key]

with open('weight_{}.pkl'.format(datatype), 'wb+') as handle:
    pickle.dump(w, handle)
