import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from sklearn.metrics import average_precision_score

from basenet import ResNet50

class multilabel_classifier():

    def __init__(self, device, dtype, num_categs=171, learning_rate=0.1, modelpath=None):
        self.model = ResNet50(n_classes=num_categs, pretrained=True)
        self.num_categs = num_categs
        self.model.require_all_grads()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.device = device
        self.dtype = dtype
        self.epoch = 0
        self.print_freq = 10
        if modelpath != None:
            A = torch.load(modelpath, map_location=device)
            self.model.load_state_dict(A['model'])
            self.epoch = A['epoch']

    def forward(self, x):
        out, feature = self.model(x)
        return out, feature

    def save_model(self, path):
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer, 'epoch':self.epoch}, path)

    def train(self, loader):
        """Train the model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        for i, (images, labels, IDs) in enumerate(loader):

            images, labels = images.to(device=self.device, dtype=self.dtype), labels.to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss_bce = torch.nn.BCEWithLogitsLoss()
            loss = loss_bce(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1

    def test(self, loader):
        """Evaluate with the model"""

        if (self.device==torch.device('cuda')):
            self.model.cuda()
        self.model.eval()

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.num_categs)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.num_categs)

            for i, (images, labels, IDs) in enumerate(loader):

                images, labels = images.to(device=self.device, dtype=self.dtype), labels.to(device=self.device, dtype=self.dtype)

                scores, _ = self.forward(images)
                scores = torch.sigmoid(scores).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

            APs = []
            for k in range(self.num_categs):
                APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))

            mAP = np.nanmean(APs)

        return APs, mAP
