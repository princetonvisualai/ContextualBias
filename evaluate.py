import pickle, time, argparse
from os import path, mkdir
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

def recall3(labels_list, scores_list, k):
    '''
    Top-3 recall evaluation (used for DeepFashion)
    '''
    num_imgs = np.sum(labels_list)
    image_scores = scores_list[labels_list.astype(bool)]
    
    # Get top 3 labels (index) from images that contain label k       
    if len(image_scores.shape)==1:
        top3_labels = image_scores.argsort()[-3:]
    else:
        top3_labels = image_scores.argsort()[:, -3:]
    num_intop3 = np.sum(top3_labels==k)

    # Recall@3 is num_intop3/num_imgs
    recall = num_intop3 / num_imgs if num_imgs != 0 else 0
    return recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels', type=str, default='/n/fs/context-scr/COCOStuff/labels_val.pkl')
    parser.add_argument('--batchsize', type=int, default=170)
    parser.add_argument('--nclasses', type=int, default=171)
    parser.add_argument('--hs', type=int, default=2048)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    arg = vars(parser.parse_args())
    if arg['model'] == 'splitbiased':
        arg['nclasses'] = arg['nclasses'] + 20
    print('\n', arg, '\n')

    # Load utility files
    biased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/biased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
    if arg['dataset'] == 'COCOStuff':
        unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/{}/unbiased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())

    # Create dataloader
    testset = create_dataset(arg['dataset'], arg['labels'], biased_classes_mapped, B=arg['batchsize'], train=False, splitbiased=(arg['model']=='splitbiased'))

    # Load model
    classifier = multilabel_classifier(arg['device'], arg['dtype'], arg['nclasses'], arg['modelpath'], hidden_size=arg['hs'])

    # Do inference with the model
    if arg['model'] == 'attribdecorr':
        labels_list, scores_list, test_loss_list = classifier.test_attribdecorr(testset, pretrained_net, biased_classes_mapped, pretrained_features)
    else:
        labels_list, scores_list, val_loss_list = classifier.test(testset)

    # Calculate and record mAP
    APs = []

    if arg['dataset'] == 'DeepFashion':
        for k in range(arg['nclasses']):
            recall = recall3(labels_list[:,k], scores_list, k)
            APs.append(recall)

    else:
        for k in range(arg['nclasses']):
            APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
    mAP = np.nanmean(APs)
    print('mAP (all): {:.2f}'.format(mAP*100.))
    if arg['dataset'] == 'COCOStuff':
        mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
        print('mAP (unbiased): {:.2f}\n'.format(mAP_unbiased*100.))

    # Calculate exclusive/co-occur AP for each biased category
    exclusive_AP_list = []
    cooccur_AP_list = []
    biased_classes_list = sorted(list(biased_classes_mapped.keys()))

    for k in range(len(biased_classes_list)):
        b = biased_classes_list[k]
        c = biased_classes_mapped[b]

        # Categorize the images into co-occur/exclusive/other
        if arg['model'] == 'splitbiased':
            cooccur = (labels_list[:,arg['nclasses']+k-20]==1)
            print(len(np.where(cooccur==True)[0]))
            exclusive = (labels_list[:,b]==1)
        else:
            cooccur = (labels_list[:,b]==1) & (labels_list[:,c]==1)
            exclusive = (labels_list[:,b]==1) & (labels_list[:,c]==0)
        other = (~exclusive) & (~cooccur)

        # Calculate AP for co-occur/exclusive sets
        if arg['model'] == 'splitbiased':
            cooccur_AP = average_precision_score(labels_list[cooccur+other, arg['nclasses']+k-20],
                scores_list[cooccur+other, arg['nclasses']+k-20])
        else:
            if arg['dataset'] == 'DeepFashion':
                cooccur_AP = recall3(labels_list[cooccur+other, b], scores_list[cooccur+other], b)
            else:
                cooccur_AP = average_precision_score(labels_list[cooccur+other, b],scores_list[cooccur+other, b])
        if arg['dataset'] == 'DeepFashion':
            exclusive_AP = recall3(labels_list[exclusive+other, b], scores_list[exclusive+other], b)
        else:
            exclusive_AP = average_precision_score(labels_list[exclusive+other, b],scores_list[exclusive+other, b])

        # Record and print
        cooccur_AP_list.append(cooccur_AP)
        exclusive_AP_list.append(exclusive_AP)
        print('{:>10} - {:>10}: exclusive {:5.2f}, co-occur {:5.2f}'.format(onehot_to_humanlabels[b], onehot_to_humanlabels[c], exclusive_AP*100., cooccur_AP*100.))

    print('\nMean: exclusive {:.2f}, co-occur {:.2f}'.format(np.mean(exclusive_AP_list)*100., np.mean(cooccur_AP_list)*100.))

if __name__ == "__main__":
    main()
