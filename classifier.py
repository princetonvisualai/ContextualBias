import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

    def __init__(self, device, dtype, nclasses=171, modelpath=None, hidden_size=2048, learning_rate=0.1, weight_decay=1e-4):
        self.nclasses = nclasses
        self.device = device
        self.dtype = dtype
        self.model = ResNet50(n_classes=nclasses, hidden_size=hidden_size, pretrained=True)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.require_all_grads()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        self.epoch = 0
        self.print_freq = 10
        if modelpath != None:
            A = torch.load(modelpath, map_location=device)
            self.model.load_state_dict(A['model'])
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
            ids_list = []
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
            ids_list = []
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
            ids_list = []
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
            ids_list = []
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

    def train_cam(self, loader, pretrained_net, biased_classes_mapped):
        """Train the 'CAM-based' model for one epoch"""

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
        self.model.train()

        # Hook the feature extractor
        Classifier_features = []
        def hook_classifier_features(module, input, output):
            Classifier_features.append(output)
        self.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
        Classifier_params = list(self.model.parameters())
        Classifier_softmax_weight = Classifier_params[-2].squeeze(0)

        pretrained_features = []
        def hook_pretrained_feature(module, input, output):
            pretrained_features.append(output)
        pretrained_net.model._modules['resnet'].layer4.register_forward_hook(hook_pretrained_feature)
        pretrained_params = list(pretrained_net.model.parameters())
        pretrained_softmax_weight = np.squeeze(pretrained_params[-2])

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
            Classifier_features = []
            outputs = self.forward(images) # where the length of Classifier_features increases
            CAMs = torch.Tensor(0, 2, 7, 7).to(device=self.device)
            for k in range(len(cooccur)):
                CAM = returnCAM(Classifier_features[0][cooccur[k]].unsqueeze(0), Classifier_softmax_weight, cooccur_classes[k], self.device)
                CAMs = torch.cat((CAMs, CAM.unsqueeze(0)), 0)

            # Get CAM from the pre-trained network
            pretrained_features = []
            _ = pretrained_net.model(images)
            CAMs_pretrained = torch.Tensor(0, 2, 7, 7).to(self.device)
            for k in range(len(cooccur)):
                CAM_pretrained = returnCAM(pretrained_features[0][cooccur[k]].unsqueeze(0), pretrained_softmax_weight, cooccur_classes[k], self.device)
                CAMs_pretrained = torch.cat((CAMs_pretrained, CAM_pretrained.unsqueeze(0)), 0)

            # Compute and update with the loss
            self.optimizer.zero_grad()
            l_o = (CAMs[:,0] * CAMs[:,1]).mean()
            l_r = torch.abs(CAMs - CAMs_pretrained).mean()
            criterion = torch.nn.BCEWithLogitsLoss()
            l_bce = criterion(outputs.squeeze(), labels)
            loss = 0.1*l_o + 0.01*l_r + l_bce
            loss.backward()
            self.optimizer.step()

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
        Classifier_features = []
        def hook_classifier_features(module, input, output):
            Classifier_features.append(output)
        self.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
        Classifier_params = list(self.model.parameters())
        Classifier_softmax_weight = Classifier_params[-2].squeeze(0)

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
                Classifier_features = []
                outputs = self.forward(images)
                CAMs = torch.Tensor(0, 2, 7, 7).to(device=self.device)
                for k in range(len(cooccur)):
                    CAM = returnCAM(Classifier_features[0][cooccur[k]].unsqueeze(0), Classifier_softmax_weight, cooccur_classes[k], self.device)
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


    def train_featuresplit(self, loader, biased_classes_mapped, weight, xs_prev_ten):
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
                x_non = self.forward(images[~exclusive])
                criterion = torch.nn.BCEWithLogitsLoss()
                loss_non = criterion(x_non, labels[~exclusive])
                loss_non.backward()
                self.optimizer.step()

                # Keep track of xs
                xs_prev_ten.append(x_non[:, 1024:].detach())
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
                    x_exc[:, 1024:] = xs_mean.detach()

                # Get the loss
                out_exc = x_exc
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
                self.model.resnet.fc.weight.grad[b_list, 1024:] = 0.
                assert not (self.model.resnet.fc.weight.grad[b_list, 1024:] != 0.).sum() > 0
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
