import pickle, time, argparse
from os import path, makedirs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str, default='baseline',
    choices=['baseline', 'cam', 'featuresplit', 'splitbiased', 'weighted',
    'removeclabels', 'removecimages', 'negativepenalty', 'classbalancing'])
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--train_batchsize', type=int, default=200)
parser.add_argument('--test_batchsize', type=int, default=170)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--hs', type=int, default=2048)
parser.add_argument('--nclasses', type=int, default=171)
parser.add_argument('--modelpath', type=str, default=None)
parser.add_argument('--pretrainedpath', type=str)
parser.add_argument('--outdir', type=str, default='/n/fs/context-scr/COCOStuff/save')
parser.add_argument('--labels_train', type=str, default='/n/fs/context-scr/COCOStuff/labels_train.pkl')
parser.add_argument('--labels_val', type=str, default='/n/fs/context-scr/COCOStuff/labels_val.pkl')
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--dtype', default=torch.float32)

arg = vars(parser.parse_args())
arg['outdir'] = '{}/{}'.format(arg['outdir'], arg['model'])
if arg['model'] == 'splitbiased':
    arg['nclasses'] = arg['nclasses'] + 20
print('\n', arg, '\n')

# Create output directory
if not path.isdir(arg['outdir']):
    makedirs(arg['outdir'])

# Load utility files
#biased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/biased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
biased_classes_mapped = pickle.load(open('biased_classes_mapped.pkl', 'rb'))
if arg['dataset'] == 'COCOStuff':
    unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/COCOStuff/unbiased_classes_mapped.pkl', 'rb'))
#humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
humanlabels_to_onehot = pickle.load(open('humanlabels_to_onehot.pkl', 'rb'))
onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

# Create data loaders
removeclabels = True if (arg['model'] == 'removeclabels') else False
removecimages = True if (arg['model'] == 'removecimages') else False
splitbiased = True if (arg['model'] == 'splitbiased') else False
trainset = create_dataset(arg['dataset'], arg['labels_train'], biased_classes_mapped, B=arg['train_batchsize'], train=True, removeclabels=removeclabels, removecimages=removecimages, splitbiased=splitbiased)
testset = create_dataset(arg['dataset'], arg['labels_val'], biased_classes_mapped, B=arg['test_batchsize'], train=False, splitbiased=splitbiased)

# Initialize classifier
classifier = multilabel_classifier(arg['device'], arg['dtype'], nclasses=arg['nclasses'], modelpath=arg['modelpath'], hidden_size=arg['hs'], learning_rate=arg['lr'])
if arg['model'] == 'cam':
    pretrained_net = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['pretrainedpath'])
classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=arg['lr'], momentum=0.9, weight_decay=arg['wd'])

# Calculate loss weights for the feature-splitting method
if arg['model'] == 'featuresplit':
    weight = calculate_weight(arg['labels_train'], arg['nclasses'], biased_classes_mapped, humanlabels_to_onehot)
    weight = weight.to(arg['device'])

# Start training
tb = SummaryWriter(log_dir='{}/runs'.format(arg['outdir']))
start_time = time.time()
print('\nStarted training at {}\n'.format(start_time))
for i in range(classifier.epoch, classifier.epoch+arg['nepoch']+1):

    # Reduce learning rate from 0.1 to 0.01
    if i == 20 and arg['dataset'] == 'COCOStuff': # Paper: decay at epoch 60
        classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.01, momentum=0.9, weight_decay=arg['wd'])
    if i == 10 and arg['dataset'] == 'AwA':
        classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.001, momentum=0.9, weight_decay=arg['wd'])
    if i == 20 and arg['dataset'] == 'DeepFashion': # Paper: decay at epoch 30
        classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.01, momentum=0.9, weight_decay=arg['wd'])

    if arg['model'] in ['baseline', 'removeclabels', 'removecimages', 'splitbiased']:
        train_loss_list = classifier.train(trainset)
    if arg['model'] == 'negativepenalty':
        train_loss_list = classifier.train_negativepenalty(trainset, biased_classes_mapped, penalty=10)
    if arg['model'] == 'classbalancing':
        train_loss_list = classifier.train_classbalancing(trainset, biased_classes_mapped, beta=0.99)
    if arg['model'] == 'weighted':
        train_loss_list = classifier.train_weighted(trainset, biased_classes_mapped, weight=10)
    if arg['model'] == 'cam':
        train_loss_list = classifier.train_CAM(trainset, pretrained_net, biased_classes_mapped)
    if arg['model'] == 'featuresplit':
        if i == 0:
            xs_prev_ten = []
        train_loss_list, xs_prev_ten = classifier.train_featuresplit(trainset, biased_classes_mapped, weight, xs_prev_ten)

    # Save the model
    classifier.save_model('{}/model_{}.pth'.format(arg['outdir'], i))

    # Do inference with the model
    if arg['model'] in ['baseline', 'removeclabels', 'removecimages', 'splitbiased']:
        labels_list, scores_list, test_loss_list = classifier.test(testset)
    if arg['model'] == 'negativepenalty':
        labels_list, scores_list, test_loss_list = classifier.test_negativepenalty(testset, biased_classes_mapped, penalty=10)
    if arg['model'] == 'classbalancing':
        labels_list, scores_list, test_loss_list = classifier.test_classbalancing(testset, biased_classes_mapped, beta=0.99)
    if arg['model'] == 'weighted':
        labels_list, scores_list, test_loss_list = classifier.test_weighted(testset, biased_classes_mapped, weight=10)

    # Record train/val loss
    tb.add_scalar('Loss/Train', np.mean(train_loss_list), i)
    tb.add_scalar('Loss/Test', np.mean(test_loss_list), i)

    # Calculate and record mAP
    APs = []
    for k in range(arg['nclasses']):
        APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
    mAP = np.nanmean(APs)
    tb.add_scalar('mAP/all', mAP*100, i)
    if arg['dataset'] == 'COCOStuff':
        mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
        tb.add_scalar('mAP/unbiased', mAP_unbiased*100, i)

    # Calculate exclusive/co-occur AP for each biased category
    exclusive_AP_dict = {}; cooccur_AP_dict = {}
    biased_classes_list = sorted(list(biased_classes_mapped.keys()))

    for k in range(len(biased_classes_list)):
        b = biased_classes_list[k]
        c = biased_classes_mapped[b]

        # Categorize the images into co-occur/exclusive/other
        if splitbiased:
            cooccur = (labels_list[:,arg['nclasses']+k-20]==1)
            exclusive = (labels_list[:,b]==1)
        else:
            cooccur = (labels_list[:,b]==1) & (labels_list[:,c]==1)
            exclusive = (labels_list[:,b]==1) & (labels_list[:,c]==0)
        other = (~exclusive) & (~cooccur)

        # Calculate AP for co-occur/exclusive sets
        if splitbiased:
            cooccur_AP = average_precision_score(labels_list[cooccur+other, arg['nclasses']+k-20],
                scores_list[cooccur+other, arg['nclasses']+k-20])
        else:
            cooccur_AP = average_precision_score(labels_list[cooccur+other, b],
                scores_list[cooccur+other, b])
        exclusive_AP = average_precision_score(labels_list[exclusive+other ,b],
            scores_list[exclusive+other, b])
        cooccur_AP_dict[b] = cooccur_AP
        exclusive_AP_dict[b] = exclusive_AP

        # Record co-occur/exclusive AP
        tb.add_scalar('{}/co-occur'.format(onehot_to_humanlabels[b]), cooccur_AP_dict[b]*100, i)
        tb.add_scalar('{}/exclusive'.format(onehot_to_humanlabels[b]), exclusive_AP_dict[b]*100, i)

    # Record mean co-occur/exclusive AP
    tb.add_scalar('mAP/co-occur', np.mean(list(cooccur_AP_dict.values()))*100, i)
    tb.add_scalar('mAP/exclusive', np.mean(list(exclusive_AP_dict.values()))*100, i)

    # Print out information
    print('\nEpoch: {}'.format(i))
    print('Loss: train {:.5f}, val {:.5f}'.format(np.mean(train_loss_list), np.mean(test_loss_list)))
    if arg['dataset'] == 'COCOStuff':
        print('Test mAP: all {} {:.5f}, unbiased 60 {:.5f}'.format(arg['nclasses'], mAP*100, mAP_unbiased*100))
    else:
        print('Test mAP: all {} {:.5f}'.format(arg['nclasses'], mAP*100))
    print('Test mAP: co-occur {:.5f}, exclusive {:.5f}'.format(np.mean(list(cooccur_AP_dict.values()))*100, np.mean(list(exclusive_AP_dict.values()))*100))
    print('Time passed so far: {:.2f} minutes\n'.format((time.time()-start_time)/60.))

# Close tensorboard logger
tb.close()
