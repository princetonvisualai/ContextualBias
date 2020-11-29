import pickle
import glob
import torch
import numpy as np
from PIL import Image

from classifier import multilabel_classifier
from nicole_load_data import *

path = '/n/fs/context-scr/nmeister/'

prevlabels = glob.glob('/n/fs/visualai-scr/Data/Coco/2014data/train2014/*.pkl')
pkl_labels = [label[:-4] for label in prevlabels]

humanlabels_to_onehot = pickle.load(open(path+'humanlabels_to_onehot.pkl', 'rb'))
labels_train = pickle.load(open(path+'labels_train.pkl', 'rb'))
modelpath = path + 'stage1_23.pth' 
Classifier = multilabel_classifier(torch.device('cpu'), torch.float32, modelpath=modelpath)
Classifier.model.eval()

BATCH_SIZE = 64
trainset = create_dataset(COCOStuff, labels=path+'labels_train.pkl', B=BATCH_SIZE)

for i, (images, labels, ids) in enumerate(trainset):

  print(i)
  
  next_iter = False
  
  for i, ID in enumerate(ids):
    if ID[:-4] in pkl_labels:
      next_iter = True
      continue
  if next_iter:
    continue 
  
  images, labels = images.to(device=Classifier.device, dtype=Classifier.dtype), labels.to(device=Classifier.device, dtype=Classifier.dtype)
  
  scores, _ = Classifier.forward(images)
  scores = torch.sigmoid(scores).squeeze()
  
  for i, ID in enumerate(ids):
    with open(ID[:-4]+".pkl", 'wb') as handle:
      pickle.dump(scores[i], handle)
  
