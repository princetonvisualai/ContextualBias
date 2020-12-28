import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T

from basenet import ResNet50

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
        out, feature = self.model(x)
        return out, feature

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
            outputs, _ = self.forward(images)
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
                outputs, _ = self.forward(images)

                # Ten crop
                # bs, ncrops, c, h, w = images.size()
                # outputs, _ = self.forward(images.view(-1, c, h, w)) # fuse batch size and ncrops
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
            outputs, _ = self.forward(images)
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

                outputs, _ = self.forward(images)
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

    def train_classbalancing(self, loader, biased_classes_mapped, beta=0.99):
        """Train the 'strong baseline - class-balancing' model for one epoch"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        for i, (images, labels, ids) in enumerate(loader):

            images  = images.to(device=self.device, dtype=self.dtype)
            labels = labels.to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_tensor = criterion(outputs.squeeze(), labels)

            # Create a loss weight tensor with class balancing weights
            weight_tensor = torch.ones_like(outputs)
            for b in biased_classes_mapped.keys():
                c = biased_classes_mapped[b]
                cooccur = (labels[:,b]==1) & (labels[:,c]==1)
                exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                other = (~exclusive) & (~cooccur)

                # Calculate weight (1-beta)/(1-beta^n) for each group
                cooccur_weight = (1-beta)/(1-torch.pow(beta, cooccur.sum()))
                exclusive_weight = (1-beta)/(1-torch.pow(beta, exclusive.sum()))
                other_weight = (1-beta)/(1-torch.pow(beta, other.sum()))

                # Have the smallest weight be 1 and the rest proportional to it
                # so as to balance it with other categories that have weight 1
                # instead of having the three weights sum to 1
                sum_weight = torch.min(torch.stack([exclusive_weight, cooccur_weight, other_weight]))

                # Appropriately normalize the weights
                exclusive_weight /= sum_weight
                cooccur_weight /= sum_weight
                other_weight /= sum_weight

                # Replace inf with 1
                if cooccur.sum() == 0: cooccur_weight = cooccur.sum() + 1.
                if exclusive.sum() == 0: exclusive_weight = exclusive.sum() + 1.
                if other.sum() == 0: other_weight = other.sum() + 1.

                # Assign the weights
                weight_tensor[exclusive, b] = exclusive_weight
                weight_tensor[cooccur, b] = cooccur_weight
                weight_tensor[other, b] = other_weight

            # Calculate and make updates with the weighted loss
            loss = (weight_tensor * loss_tensor).mean()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test_classbalancing(self, loader, biased_classes_mapped, beta=0.99):
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

                outputs, _ = self.forward(images)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss_tensor = criterion(outputs.squeeze(), labels)

                # Create a loss weight tensor with class balancing weights
                weight_tensor = torch.ones_like(outputs)
                for b in biased_classes_mapped.keys():
                    c = biased_classes_mapped[b]
                    cooccur = (labels[:,b]==1) & (labels[:,c]==1)
                    exclusive = (labels[:,b]==1) & (labels[:,c]==0)
                    other = (~exclusive) & (~cooccur)

                    # Calculate weight (1-beta)/(1-beta^n) for each group
                    cooccur_weight = (1-beta)/(1-torch.pow(beta, cooccur.sum()))
                    exclusive_weight = (1-beta)/(1-torch.pow(beta, exclusive.sum()))
                    other_weight = (1-beta)/(1-torch.pow(beta, other.sum()))

                    # Have the smallest weight be 1 and the rest proportional to it
                    # so as to balance it with other categories that have weight 1
                    # instead of having the three weights sum to 1
                    sum_weight = torch.min(torch.stack([exclusive_weight, cooccur_weight, other_weight]))

                    # Appropriately normalize the weights
                    exclusive_weight /= sum_weight
                    cooccur_weight /= sum_weight
                    other_weight /= sum_weight

                    # Replace inf with 1
                    if cooccur.sum() == 0: cooccur_weight = cooccur.sum() + 1.
                    if exclusive.sum() == 0: exclusive_weight = exclusive.sum() + 1.
                    if other.sum() == 0: other_weight = other.sum() + 1.

                    # Assign the weights
                    weight_tensor[exclusive, b] = exclusive_weight
                    weight_tensor[cooccur, b] = cooccur_weight
                    weight_tensor[other, b] = other_weight

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
            outputs, _ = self.forward(images)
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

                outputs, _ = self.forward(images)
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

    def train_CAM(self, loader, pretrained_net, biased_classes_mapped):
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

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        pretrained_net.model = pretrained_net.model.to(device=self.device, dtype=self.dtype)
        pretrained_net.model.eval()

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

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

            # Hook the feature extractor
            Classifier_features = []
            def hook_classifier_features(module, input, output):
                Classifier_features.append(output)
            self.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
            Classifier_params = list(self.model.parameters())
            Classifier_softmax_weight = Classifier_params[-2].squeeze(0)

            # Get image features
            self.optimizer.zero_grad()
            _, features = self.forward(images)
            outputs = features

            # Get CAM from the current network
            CAMs = torch.Tensor(0, 2, 7, 7).to(device=self.device)
            for k in range(len(cooccur)):
                CAM = returnCAM(Classifier_features[0][cooccur[k]].unsqueeze(0), Classifier_softmax_weight, cooccur_classes[k], self.device)
                CAMs = torch.cat((CAMs, CAM.unsqueeze(0)), 0)

            # Get CAM from the pre-trained network
            pretrained_features = []
            def hook_pretrained_feature(module, input, output):
                pretrained_features.append(output)
            pretrained_net.model._modules['resnet'].layer4.register_forward_hook(hook_pretrained_feature)
            pretrained_params = list(pretrained_net.model.parameters())
            pretrained_softmax_weight = np.squeeze(pretrained_params[-2])
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
                _, x_non = self.forward(images[~exclusive])
                #out_non = self.model.fc(self.model.dropout(self.model.relu(x_non)))
                out_non = x_non
                criterion = torch.nn.BCEWithLogitsLoss()
                loss_non = criterion(out_non, labels[~exclusive])
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
                _, x_exc = self.forward(images[exclusive])

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
                #self.model.fc.weight.grad[b_list, 1024:] = 0.
                #assert not (self.model.fc.weight.grad[b_list, 1024:] != 0.).sum() > 0
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
