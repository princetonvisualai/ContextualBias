import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class ResNet50(nn.Module):
    def __init__(self, n_classes=1000, pretrained=True, hidden_size=2048):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(hidden_size, n_classes)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs

class multilabel_classifier():

    def __init__(self, device, dtype, nclasses=171, modelpath=None, hidden_size=2048, learning_rate=0.1, weight_decay=1e-4, attribdecorr=False, compshare_lambda=0.1):
        self.nclasses = nclasses
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
            
        # For attribute decorrelation baseline, we train a linear classifier on top of pretrained deep features
        if attribdecorr:
            params = OrderedDict([
                ('W', torch.nn.Linear(hidden_size, nclasses, bias=False))
            ])
            self.model = torch.nn.Sequential(params)
            for param in self.model.parameters():
                param.requires_grad = True
            self.compshare_lambda = compshare_lambda
        else:
            self.model = ResNet50(n_classes=nclasses, hidden_size=hidden_size, pretrained=True)
            self.model.require_all_grads()
        
        # Multi-GPU training
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        self.epoch = 0
        self.print_freq = 100

        if modelpath != None:
            A = torch.load(modelpath, map_location=device)
            load_state_dict = A['model']
            load_prefix = list(load_state_dict.keys())[0][:6]
            new_state_dict = {}
            for key in load_state_dict:
                value = load_state_dict[key]
                # Multi-GPU state dict has the prefix 'module.' appended in front of each key
                if torch.cuda.device_count() > 1:
                    if load_prefix != 'module': 
                        new_key = 'module.' + key
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                else:
                    if load_prefix == 'module':
                        new_key = key[7:]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
            self.model.load_state_dict(new_state_dict)
            self.epoch = A['epoch']
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def save_model(self, path):
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer, 'epoch':self.epoch}, path)

    def train(self, loader):
        """Train the 'standard baseline' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()
            outputs = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test(self, loader):
        """Evaluate the 'standard baseline' model"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                # Center crop
                outputs = self.forward(images)

                # Ten crop
                # bs, ncrops, c, h, w = images.size()
                # outputs = self.forward(images.view(-1, c, h, w)) # fuse batch size and ncrops
                # outputs = outputs.view(bs, ncrops, -1).mean(1) # avg over crops

                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(outputs.squeeze(), labels)
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def train_negativepenalty(self, loader, biased_classes_mapped, penalty=10):
        """Train the 'strong baseline - negative penalty' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images  = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()
            outputs = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_tensor = criterion(outputs.squeeze(), labels)

            # Create a loss weight tensor with a large negative penalty for c
            weight_tensor = torch.ones_like(outputs)
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                weight_tensor[exclusive, c] = penalty

            # Calculate and make updates with the weighted loss
            loss = (weight_tensor * loss_tensor).mean()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test_negativepenalty(self, loader, biased_classes_mapped, penalty=10):
        """Evaluate the 'strong baseline - negative penalty' model"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                outputs = self.forward(images)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_tensor = criterion(outputs.squeeze(), labels)

                # Create a loss weight tensor with a large negative penalty for c
                weight_tensor = torch.ones_like(outputs)
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                    weight_tensor[exclusive, c] = penalty

                # Calculate and make updates with the weighted loss
                loss = (weight_tensor * loss_tensor).mean()
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def train_classbalancing(self, loader, biased_classes_mapped, weight):
        """Train the 'strong baseline - class-balancing' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):

            images  = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()
            outputs = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_tensor = criterion(outputs.squeeze(), labels)

            # Create a loss weight tensor with class balancing weights
            weight_tensor = torch.ones_like(outputs)
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                cooccur = (labels[:,b]==1) & (labels[:,c]==1)
                exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                other = (~exclusive) & (~cooccur)

                # Assign the weights
                weight_tensor[exclusive, b] = weight[b, 0]
                weight_tensor[cooccur, b] = weight[b, 1]
                weight_tensor[other, b] = weight[b, 2]

            # Calculate and make updates with the weighted loss
            loss = (weight_tensor * loss_tensor).mean()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test_classbalancing(self, loader, biased_classes_mapped, weight):
        """Evaluate the 'strong baseline - class-balancing' model"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                outputs = self.forward(images)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_tensor = criterion(outputs.squeeze(), labels)

                # Create a loss weight tensor with class balancing weights
                weight_tensor = torch.ones_like(outputs)
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    cooccur = (labels[:,b]==1) & (labels[:,c]==1)
                    exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                    other = (~exclusive) & (~cooccur)

                    # Assign the weights
                    weight_tensor[exclusive, b] = weight[b, 0]
                    weight_tensor[cooccur, b] = weight[b, 1]
                    weight_tensor[other, b] = weight[b, 2]

                # Calculate and make updates with the weighted loss
                loss = (weight_tensor * loss_tensor).mean()
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def train_weighted(self, loader, biased_classes_mapped, weight=10):
        """Train the 'strong baseline - weighted loss' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images  = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()
            outputs = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_tensor = criterion(outputs.squeeze(), labels)

            # Create a loss weight tensor with a large negative penalty for b
            weight_tensor = torch.ones_like(outputs)
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                weight_tensor[exclusive, b] = weight

            # Calculate and make updates with the weighted loss
            loss = (weight_tensor * loss_tensor).mean()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test_weighted(self, loader, biased_classes_mapped, weight=10):
        """Evaluate the 'strong baseline - weighted loss' model"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                outputs = self.forward(images)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_tensor = criterion(outputs.squeeze(), labels)

                # Create a loss weight tensor with a large negative penalty for b
                weight_tensor = torch.ones_like(outputs)
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                    weight_tensor[exclusive, b] = weight

                # Calculate and make updates with the weighted loss
                loss = (weight_tensor * loss_tensor).mean()
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def train_attribdecorr(self, loader, pretrained_net, biased_classes_mapped, humanlabels_to_onehot, pretrained_features, compshare_lambda=0.1):
        # Define semantic groups according to http://vision.cs.utexas.edu/projects/resistshare/
        semantic_attributes = [
            ['patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin'],
            ['fierce', 'timid', 'smart', 'group', 'solitary', 'nestspot', 'domestic'],
            ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow'],
            ['flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail',
             'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws',
             'tusks', 'bipedal', 'quadrapedal'],
            ['flys', 'hops', 'swims', 'tunnels', 'walks', 'fast', 'slow', 'strong',
             'weak', 'muscle'],
            ['fish', 'meat', 'plankton', 'vegetation', 'insects', 'forager', 'grazer',
             'hunter', 'scavenger', 'skimmer', 'stalker'],
            ['coastal', 'desert', 'bush', 'plains', 'forest', 'fields', 'jungle', 'mountains',
             'ocean', 'ground', 'water', 'tree', 'cave'],
            ['active', 'inactive', 'nocturnal', 'hibernate', 'agility'],
            ['big', 'small', 'bulbous', 'lean']
        ]
        semantic_attributes_idxs = []
        for group in semantic_attributes:
            idxs = [humanlabels_to_onehot[attribute] for attribute in group]
            semantic_attributes_idxs.append(idxs)

        pretrained_net.model.to(device=self.device, dtype=self.dtype)
        pretrained_net.model.eval()
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Get output scores by substituting the last fully connected layer with W
            self.optimizer.zero_grad()
            W_key = list(self.model.state_dict().keys())[0]
            W = self.model.state_dict()[W_key]
            pretrained_features.clear()
            pretrained_net.model.forward(images)

            conv_outputs = [x.to(device=self.device, dtype=self.dtype) for x in pretrained_features]
            conv_outputs = torch.cat(conv_outputs, dim=0).to(device=self.device, dtype=self.dtype)
            outputs = self.model.forward(conv_outputs)
            criterion = torch.nn.BCEWithLogitsLoss()
            regression_loss = criterion(outputs, labels)

            # Compute competition-sharing loss
            delta = 0
            for d in range(self.hidden_size):
                for g in range(len(semantic_attributes_idxs)):
                    Sg = torch.LongTensor(semantic_attributes_idxs[g])
                    w_dSg = W[Sg, d]
                    delta += torch.linalg.norm(w_dSg)

            compshare_loss = 0
            for d in range(self.hidden_size):
                for g in range(len(semantic_attributes_idxs)):
                    Sg = torch.LongTensor(semantic_attributes_idxs[g])
                    w_dSg = W[Sg, d]
                    delta_dg = torch.linalg.norm(w_dSg) / delta
                    compshare_loss += torch.linalg.norm(w_dSg)**2.0 / delta_dg

            loss = regression_loss + compshare_lambda * compshare_loss
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training step {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test_attribdecorr(self, loader, pretrained_net, biased_classes_mapped, pretrained_features, compshare_lambda=0.01):

        pretrained_net.model.to(device=self.device, dtype=self.dtype)
        pretrained_net.model.eval()
        self.model.eval()

        with torch.no_grad():
            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                # Get output scores by substituting the last fully connected layer with W
                W_key = list(self.model.state_dict().keys())[0]
                W = self.model.state_dict()[W_key]
                pretrained_features.clear()
                pretrained_net.model.forward(images)

                # Center crop
                conv_outputs = [x.to(device=self.device, dtype=self.dtype) for x in pretrained_features]
                conv_outputs = torch.cat(conv_outputs, dim=0).to(device=self.device, dtype=self.dtype)
                outputs = self.model.forward(conv_outputs)

                # Ten crop
                # bs, ncrops, c, h, w = images.size()
                # outputs = self.forward(images.view(-1, c, h, w)) # fuse batch size and ncrops
                # outputs = outputs.view(bs, ncrops, -1).mean(1) # avg over crops

                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(outputs.squeeze(), labels)
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)
        return labels_list, scores_list, loss_list

    def train_cam(self, loader, pretrained_net, biased_classes_mapped, pretrained_features, classifier_features):
        """Train the 'CAM-based' model for one epoch"""

        def returnCAM(feature_conv, weight_softmax, class_idx, device):
            nc, h, w = feature_conv.shape # (hidden_size, height, width)
            output_cam = torch.Tensor(0, 7, 7).to(device=device)
            for idx in class_idx:
                cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - cam.min()
                cam_img = cam / cam.max() # (h, w)
                output_cam = torch.cat((output_cam, cam_img.unsqueeze(0)), dim=0)
            return output_cam # (2, height, width) dim0 is 2 because idx = [b, c]

        # Define GPU for computing CAMs
        if torch.cuda.device_count() > 1:
            cam_device = torch.device('cuda:1')
        else:
            cam_device = torch.device('cuda:0')

        # Load models and save final layer (softmax) weights
        pretrained_net.model = pretrained_net.model.to(device=self.device, dtype=self.dtype)
        pretrained_net.model.eval()
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        classifier_params = list(self.model.parameters())
        classifier_softmax_weight = classifier_params[-2].squeeze(0).to(cam_device)
        pretrained_params = list(pretrained_net.model.parameters())
        pretrained_softmax_weight = np.squeeze(pretrained_params[-2]).to(cam_device)

        # Loop over batches
        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Identify co-occurring instances
            cooccur = [] # Image indices with co-occurrences
            cooccur_classes = [] # (b, c) pair for the above images
            for m in range(labels.shape[0]):
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    if (labels[m,b]==1) & (labels[m,c]==1):
                        cooccur.append(m)
                        cooccur_classes.append([b, c])

            # Get CAM from the current network
            classifier_features.clear()
            outputs = self.forward(images) # where the length of classifier_features increases
            outputs.cpu()

            # In multi-GPU training, the hook outputs a list of outputs from each GPU, 
            # so we need to recombine them
            classifier_features_all = [x.to(device=torch.device('cpu'), dtype=torch.float32) for x in classifier_features]
            classifier_features_all = torch.cat(classifier_features_all, dim=0)
            biased_classifier_features = classifier_features_all[cooccur].to(device=cam_device, dtype=self.dtype)

            CAMs = torch.Tensor(0, 2, 7, 7).to(device=cam_device)
            for k in range(len(cooccur)):
                CAM = returnCAM(biased_classifier_features[k], classifier_softmax_weight, cooccur_classes[k], cam_device)
                CAMs = torch.cat((CAMs, CAM.unsqueeze(0)), 0)
            del biased_classifier_features
            CAMs = CAMs.to(device=torch.device('cpu'))

            # Get CAM from the pre-trained network
            pretrained_features.clear()
            pretrained_net.model(images)

            # In multi-GPU training, the hook outputs a list of outputs from each GPU,
            # so we need to recombine them
            pretrained_features_all = [x.to(device=torch.device('cpu'), dtype=torch.float32) for x in pretrained_features]
            pretrained_features_all = torch.cat(pretrained_features_all, dim=0)
            biased_pretrained_features = pretrained_features_all[cooccur].to(device=cam_device, dtype=self.dtype)

            CAMs_pretrained = torch.Tensor(0, 2, 7, 7).to(cam_device)
            for k in range(len(cooccur)):
                CAM_pretrained = returnCAM(biased_pretrained_features[k], pretrained_softmax_weight, cooccur_classes[k], cam_device)
                CAMs_pretrained = torch.cat((CAMs_pretrained, CAM_pretrained.unsqueeze(0)), 0)
            del biased_pretrained_features
            CAMs_pretrained = CAMs_pretrained.to(device=torch.device('cpu'))

            # Compute and update with the loss
            outputs = outputs.to(device=self.device, dtype=self.dtype)
            self.optimizer.zero_grad()
            l_o = (CAMs[:,0] * CAMs[:,1]).mean()
            l_r = torch.abs(CAMs - CAMs_pretrained).mean()
            criterion = torch.nn.BCEWithLogitsLoss()
            l_bce = criterion(outputs.squeeze(), labels)
            loss = 0.1*l_o + 0.01*l_r + l_bce
            loss.backward()
            self.optimizer.step()
            
            # Clean up
            del outputs
            del CAMs
            del CAMs_pretrained

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1

        return loss_list

    def test_cam(self, loader, pretrained_net, biased_classes_mapped):
        """Evaluate the 'CAM-based' model"""

        def returnCAM(feature_conv, weight_softmax, class_idx, device):
            bz, nc, h, w = feature_conv.shape
            output_cam = torch.Tensor(0, 7, 7).to(device=device)
            for idx in class_idx:
                cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - cam.min()
                cam_img = cam / cam.max()
                output_cam = torch.cat((output_cam, cam_img.unsqueeze(0)), 0)
            return output_cam

        pretrained_net.model = pretrained_net.model.to(device=self.device, dtype=self.dtype)
        pretrained_net.model.eval()

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Hook the feature extractor
        classifier_features = []
        def hook_classifier_features(module, input, output):
            classifier_features.append(output)
        self.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
        classifier_params = list(self.model.parameters())
        classifier_softmax_weight = classifier_params[-2].squeeze(0)

        pretrained_features = []
        def hook_pretrained_feature(module, input, output):
            pretrained_features.append(output)
        pretrained_net.model._modules['resnet'].layer4.register_forward_hook(hook_pretrained_feature)
        pretrained_params = list(pretrained_net.model.parameters())
        pretrained_softmax_weight = np.squeeze(pretrained_params[-2])

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                # Identify co-occurring instances
                cooccur = [] # Image indices with co-occurrences
                cooccur_classes = [] # (b, c) pair for the above images
                for m in range(labels.shape[0]):
                    for b in biased_classes_mapped.keys():
                        c = biased_classes_mapped[b]
                        if (labels[m,b]==1) & (labels[m,c]==1):
                            cooccur.append(m)
                            cooccur_classes.append([b, c])

                # Get CAM from the current network
                classifier_features = []
                outputs = self.forward(images)
                CAMs = torch.Tensor(0, 2, 7, 7).to(device=self.device)
                for k in range(len(cooccur)):
                    CAM = returnCAM(classifier_features[0][cooccur[k]].unsqueeze(0), classifier_softmax_weight, cooccur_classes[k], self.device)
                    CAMs = torch.cat((CAMs, CAM.unsqueeze(0)), 0)

                # Get CAM from the pre-trained network
                pretrained_features = []
                _ = pretrained_net.model(images)
                CAMs_pretrained = torch.Tensor(0, 2, 7, 7).to(self.device)
                for k in range(len(cooccur)):
                    CAM_pretrained = returnCAM(pretrained_features[0][cooccur[k]].unsqueeze(0), pretrained_softmax_weight, cooccur_classes[k], self.device)
                    CAMs_pretrained = torch.cat((CAMs_pretrained, CAM_pretrained.unsqueeze(0)), 0)

                # Compute the loss
                l_o = (CAMs[:,0] * CAMs[:,1]).mean()
                l_r = torch.abs(CAMs - CAMs_pretrained).mean()
                criterion = torch.nn.BCEWithLogitsLoss()
                l_bce = criterion(outputs.squeeze(), labels)
                loss = 0.1*l_o + 0.01*l_r + l_bce

                # Keep track of the values
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()
                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def train_featuresplit(self, loader, biased_classes_mapped, weight, xs_prev_ten, classifier_features, split=1024):
        """Train the 'feature-splitting' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Identify exclusive instances
            exclusive = torch.zeros((labels.shape[0]), dtype=bool)
            exclusive_list = [] # Image indices with exclusives
            exclusive_classes = [] # (b, c) pair for the above images
            for m in range(labels.shape[0]):
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    if (labels[m,b]==1) & (labels[m,c]==0):
                        exclusive[m] = True
                        exclusive_list.append(m)
                        exclusive_classes.append(b)

            # Update parameters with non-exclusive samples (co-occur or neither b nor c appears)
            if (~exclusive).sum() > 0:
                self.optimizer.zero_grad()
                classifier_features.clear()
                out_non = self.forward(images[~exclusive])
                x_non = classifier_features[0]
                if len(x_non.shape) < 2:
                    x_non = x_non.unsqueeze(0)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss_non = criterion(out_non, labels[~exclusive])
                loss_non.backward()
                self.optimizer.step()
                del out_non
                
                # Keep track of xs
                xs_prev_ten.append(x_non[:, split:].detach())
                if len(xs_prev_ten) > 10:
                    xs_prev_ten.pop(0)

                l_non = loss_non.item()
            else:
                l_non = 0.

            # Update parameters with exclusive samples
            if exclusive.sum() > 0:
                self.optimizer.zero_grad()
                classifier_features.clear()
                self.forward(images[exclusive])
                x_exc = classifier_features[0]
                if len(x_exc.shape) < 2:
                    x_exc = x_exc.unsqueeze(0)

                # Replace the second half of the features with xs_mean
                if len(xs_prev_ten) > 0:
                    xs_mean = torch.cat(xs_prev_ten).mean(0)
                    x_exc[:, split:] = xs_mean.detach()
                
                # Get the loss
                out_exc = self.model.resnet.fc(x_exc)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_exc_tensor = criterion(out_exc, labels[exclusive])

                # Create a loss weight tensor
                weight_tensor = torch.ones_like(out_exc)
                exclusive_unique_list = sorted(list(set(exclusive_list)))
                for k in range(len(exclusive_list)):
                    m = exclusive_unique_list.index(exclusive_list[k])
                    b = exclusive_classes[k]
                    weight_tensor[m, b] = weight[b]
                
                # Compute the final loss and the gradients
                loss_exc = (weight_tensor * loss_exc_tensor).mean()
                loss_exc.backward()

                # Zero out Ws gradients and make an update
                b_list = [i in exclusive_classes for i in range(self.nclasses)]
                #if torch.cuda.device_count() > 1:
                #    self.model._modules['module'].resnet.fc.weight.grad[b_list, split:] = None
                #    assert not (self.model._modules['module'].resnet.fc.weight.grad[b_list, split:] != 0.).sum() > 0
                #else:
                #    self.model.resnet.fc.weight.grad[b_list, split:] = None
                #    assert not (self.model.resnet.fc.weight.grad[b_list, split:] != 0.).sum() > 0
                Ws_old = self.model.resnet.fc.weight.clone().detach()
                self.optimizer.step()
                with torch.no_grad():
                    self.model.resnet.fc.weight[b_list, split:] = Ws_old[b_list, split:]

                l_exc = loss_exc.item()
            else:
                l_exc = 0.

            loss = (l_non*(~exclusive).sum() + l_exc*exclusive.sum())/exclusive.shape[0]
            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1

        return loss_list, xs_prev_ten

    def train_fs_weighted(self, loader, biased_classes_mapped, weight):
        """Train the 'non-feature-split weighted loss' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images  = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Identify exclusive instances
            exclusive = torch.zeros((labels.shape[0]), dtype=bool)
            exclusive_list = [] # Image indices with exclusives
            exclusive_classes = [] # (b, c) pair for the above images
            for m in range(labels.shape[0]):
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    if (labels[m,b]==1) & (labels[m,c]==0):
                        exclusive[m] = True
                        exclusive_list.append(m)
                        exclusive_classes.append(b)

            self.optimizer.zero_grad()
            outputs = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_tensor = criterion(outputs.squeeze(), labels)

            # Create a loss weight tensor
            weight_tensor = torch.ones_like(outputs)
            exclusive_unique_list = sorted(list(set(exclusive_list)))
            for k in range(len(exclusive_list)):
                m = exclusive_unique_list.index(exclusive_list[k])
                b = exclusive_classes[k]
                weight_tensor[m, b] = weight[b]

            # Calculate and make updates with the weighted loss
            loss = (weight_tensor * loss_tensor).mean()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1

        return loss_list

    def test_fs_weighted(self, loader, biased_classes_mapped, weight):
        """Test the 'non-feature-split weighted loss' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        with torch.no_grad():

            labels_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)
            loss_list = []

            for i, (images, labels, ids) in enumerate(loader):
                images  = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                # Identify exclusive instances
                exclusive = torch.zeros((labels.shape[0]), dtype=bool)
                exclusive_list = [] # Image indices with exclusives
                exclusive_classes = [] # (b, c) pair for the above images
                for m in range(labels.shape[0]):
                    for b in biased_classes_mapped.keys():
                        c = biased_classes_mapped[b]
                        if (labels[m,b]==1) & (labels[m,c]==0):
                            exclusive[m] = True
                            exclusive_list.append(m)
                            exclusive_classes.append(b)

                self.optimizer.zero_grad()
                outputs = self.forward(images)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_tensor = criterion(outputs.squeeze(), labels)

                # Create a loss weight tensor
                weight_tensor = torch.ones_like(outputs)
                exclusive_unique_list = sorted(list(set(exclusive_list)))
                for k in range(len(exclusive_list)):
                    m = exclusive_unique_list.index(exclusive_list[k])
                    b = exclusive_classes[k]
                    weight_tensor[m, b] = weight[b]

                # Calculate and make updates with the weighted loss
                loss = (weight_tensor * loss_tensor).mean()
                loss_list.append(loss.item())
                scores = torch.sigmoid(outputs).squeeze()

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def train_fs_noweighted(self, loader, biased_classes_mapped, weight, xs_prev_ten, split=1024):
        """Train the 'feature-splitting without weighted loss' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            # Identify exclusive instances
            exclusive = torch.zeros((labels.shape[0]), dtype=bool)
            exclusive_list = [] # Image indices with exclusives
            exclusive_classes = [] # (b, c) pair for the above images
            for m in range(labels.shape[0]):
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    if (labels[m,b]==1) & (labels[m,c]==0):
                        exclusive[m] = True
                        exclusive_list.append(m)
                        exclusive_classes.append(b)

            # Update parameters with non-exclusive samples (co-occur or neither b nor c appears)
            if (~exclusive).sum() > 0:
                self.optimizer.zero_grad()
                x_non = self.forward(images[~exclusive])
                criterion = torch.nn.BCEWithLogitsLoss()
                loss_non = criterion(x_non, labels[~exclusive])
                loss_non.backward()
                self.optimizer.step()

                # Keep track of xs
                xs_prev_ten.append(x_non[:, split:].detach())
                if len(xs_prev_ten) > 10:
                    xs_prev_ten.pop(0)

                l_non = loss_non.item()
            else:
                l_non = 0.

            # Update parameters with exclusive samples
            if exclusive.sum() > 0:
                self.optimizer.zero_grad()
                x_exc = self.forward(images[exclusive])

                # Replace the second half of the features with xs_mean
                if len(xs_prev_ten) > 0:
                    xs_mean = torch.cat(xs_prev_ten).mean(0)
                    x_exc[:, split:] = xs_mean.detach()

                # Get the loss
                out_exc = x_exc
                criterion = torch.nn.BCEWithLogitsLoss()
                loss_exc = criterion(out_exc, labels[exclusive])
                loss_exc.backward()

                # Zero out Ws gradients and make an update
                b_list = [i in exclusive_classes for i in range(self.nclasses)]
                self.model.resnet.fc.weight.grad[b_list, split:] = 0.
                assert not (self.model.resnet.fc.weight.grad[b_list, split:] != 0.).sum() > 0
                self.optimizer.step()

                l_exc = loss_exc.item()
            else:
                l_exc = 0.

            loss = (l_non*(~exclusive).sum() + l_exc*exclusive.sum())/exclusive.shape[0]
            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1

        return loss_list, xs_prev_ten
