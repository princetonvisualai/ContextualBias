# [Re] Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias

This is a non-official implementation of [Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias (CVPR 2020)](https://arxiv.org/abs/2001.03152). We developed this codebase to reproduce the experiments in the paper, as part of our participation in the [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020).

## Data pre-processing
```data_process.py```
- **Input**: labels.txt (downloaded from the official COCO-Stuff repository: https://github.com/nightrome/cocostuff)
- **Output**: humanlabels_to_onehot.pkl, labels_val.pkl, labels_train.pkl
- **Description**: Processes COCO-2014 train and validation sets by grabbing the corresponding things+stuff annotations from COCO-2017-Stuff. labels_val.pkl and label_train.pkl contain image paths as keys (e.g. '/n/fs/visualai-scr/Data/Coco/2014data/val2014/COCO_val2014_000000581913.jpg) and 171-D one-hot encoded labels as values (e.g. [0, 1, ..., 0]).

```split_80_20.py```
- **Input**: labels_train.pkl
- **Output**: labels_train_80.pkl, labels_train_20.pkl
- **Description**: Do a 80-20 split of the COCO training set to train a model for biased categories identification.

## Training
```train.py```
- **Input**: labels_train.pkl, labels_val.pkl, unbiased_classes_mapped.pkl, biased_classes_mapped.pkl, humanlabels_to_onehot.pkl
- **Output**: Optimized model parameters
- **Description**: Trains various models (baseline, cam, featuresplit, removeclabels, removecimages, negativepenalty, classbalancing)

## Evaluation
```evaluate.py```
- **Input**: biased_classes.pkl, humanlabels_to_onehot.pkl, path to the trained model you want to evaluate
- **Output**: Scores saved in evalresults
- **Description**: Evaluates a trained model on the exclusive and co-occur test distributions.

## Biased categories identification
```biased_categories.py```
- **Input**: labels_train_20.pkl, humanlabels_to_onehot.pkl, path to the model trained on labels_train_80.pkl
- **Output**: K=20 biased categories
- **Description**: Calculates bias and identifies the K=20 most biased categories

## Utils
```load_data.py```
- **Description**: Creates dataset loaders

```classifier.py```
- **Description**: Defines the multi-label classifier with various training methods. 

```basenet.py```
- **Description**: Defines ResNet-50 backbone architecture. 
