# [Re] Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias

This is a non-official implementation of [Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias](https://arxiv.org/abs/2001.03152) (CVPR 2020). As part of our participation in the [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020), we aimed to replicate the experiments described in the paper.


## Run scripts in this order
1. ```data_process.py```: Process COCO-Stuff
2. ```split_80_20.py```: Do a 80-20 split of the COCO-2014 training set
3. ```stage1_80_20.py```: Train a standard classifier (on the 80 split of the COCO-2014 training set) for biased categories identification
4. ```get_scores.py```: Get classifier scores for bias calculation
5. ```biased_categories.py```: Identify top K=20 biased categories
6. ```create_evaldata.py```: Create "co-occur" and "exclusive" test distributions
7. ```stage1.py```: Train a standard classifier
8. ```stage2_cam.py```: Train a classifier with the CAM-based method
9. ```calculate_alpha.py```: Calculate values needed for the feature-splitting method
10. ```stage2_featuresplit.py```: Train a classifier with the feature-splitting method
11. ```evaluate.py```: Evaluate the trained models on "co-occur" and "exclusive" test distributions


## Pre-processing
```data_process.py```
- **Input**: labels.txt (downloaded from the official COCO-Stuff repository: https://github.com/nightrome/cocostuff)
- **Output**: humanlabels_to_onehot.pkl, labels_val.pkl, labels_train.pkl
- **Description**: Processes COCO-2014 train and validation sets by grabbing the corresponding things+stuff annotations from COCO-2017-Stuff. labels_val.pkl and label_train.pkl contain image paths as keys (e.g. '/n/fs/visualai-scr/Data/Coco/2014data/val2014/COCO_val2014_000000581913.jpg) and 171-D one-hot encoded labels as values (e.g. [0, 1, ..., 0]).

```create_evaldata.py```
- **Input**: K biased categories list, humanlabels_to_onehot.pkl, labels_val.pkl
- **Output**: biased_classes.pkl, biased_classes_mapped.pkl, unbiased_classes_mapped.pkl, 2 (exclusive, co-occur) x K image path-label dictionaries in directory evaldata (e.g. evaldata/exclusive_snowboard_person.pkl)
- **Description**: Creates biased categories-related dictionaries and construct 'exclusive' and 'co-occur' test distributions from the COCO-2014 validation set.

```calculate_alpha.py```
- **Input**: labels_train.pkl, biased_classes.pkl, biased_classes_mapped.pkl, humanlabels_to_onehot.pkl
- **Output**: weights_train.pkl
- **Description**: Calculates alphas and weights needed for the feature splitting method's weighted loss.


## Biased categories identification
```get_scores.py```
- **Input**: labels_train_20.pkl, path to the model trained on labels_train_80.pkl
- **Output**: scores_train_20.pkl
- **Description**: Passes images in labels_train_20.pkl (20% split of COCO-2014 train) through the standard classsifier trained on labels_train_80.pkl (80% of COCO-2014 train) and outputs scores

```biased_categories.py```
- **Input**: labels_train_20.pkl, scores_train_20.pkl, humanlabels_to_onehot.pkl
- **Output**: biased_categories.pkl
- **Description**: Calculates bias and identifies the K=20 most biased categories


## Training
```stage1_80_20.py```
- **Input**: labels_train_80.pkl, labels_train_20.pkl, unbiased_classes_mapped.pkl
- **Output**: Optimized model parameters
- **Description**: Trains a "standard" baseline classifier on the 80% split of the COCO-2014 training set. This classifier is only used for biased categories identification.

```stage1.py```
- **Input**: labels_train.pkl, labels_val.pkl, unbiased_classes_mapped.pkl
- **Output**: Optimized model parameters
- **Description**: Trains a "standard" baseline classifier. The trained classifier serves as the starting point for stage2 training.

```stage2_cam.py```
- **Input**: labels_train.pkl, labels_val.pkl, biased_classes_mapped.pkl, path to the trained baseline model
- **Output**: Optimized model parameters
- **Description**: Does stage2 training with the CAM-method.

```stage2_featuresplit.py```
- **Input**: labels_train.pkl, labels_val.pkl, biased_classes_mapped.pkl, weight_train.pkl, path to the trained baseline model
- **Output**: Optimized model parameters
- **Description**: Does stage2 training with the feature splitting method.


## Evaluation
```evaluate.py```
- **Input**: biased_classes.pkl, humanlabels_to_onehot.pkl, path to the trained model you want to evaluate
- **Output**: Scores saved in evalresults
- **Description**: Evaluates a trained model on the exclusive and co-occur test distributions.


## Utils
```load_data.py```
- **Description**: Creates dataset loader. 

```classifier.py```
- **Description**: Defines the multi-label classifier class. 

```basenet.py```
- **Description**: Defines ResNet-50 backbone architecture. 
