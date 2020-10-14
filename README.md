# ML Reproducibility Challenge 2020

Our implementation of "Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias." Krishna Kumar Singh, Dhruv Mahajan, Kristen Grauman, Yong Jae Lee, Matt Feiszli, Deepti Ghadiyaram. CVPR 2020. https://arxiv.org/abs/2001.03152

## Pre-processing (run once)
```data_process.py```
- **Input**: labels.txt
- **Output**: humanlabels_to_onehot.pkl, labels_val.pkl, labels_train.pkl
- **Description**: Processes COCO-2014 train and validation sets by grabbing the corresponding things+stuff annotations from COCO-2017-Stuff. labels_val.pkl and label_train.pkl contain image paths as keys (e.g. '/n/fs/visualai-scr/Data/Coco/2014data/val2014/COCO_val2014_000000581913.jpg) and one-hot encoded labels as values (e.g. [0, 1, ..., 0]).

```create_evaldata.py```
- **Input**: K biased categories list, humanlabels_to_onehot.pkl, labels_val.pkl
- **Output**: biased_classes.pkl, biased_classes_mapped.pkl, unbiased_classes_mapped.pkl, 2 (exclusive, co-occur) x K image path-label dictionaries in directory evaldata (e.g. evaldata/exclusive_snowboard_person.pkl)
- **Description**: Creates biased categories-related dictionaries and construct 'exclusive' and 'co-occur' test distributions from the COCO-2014 validation set.

```calculate_alpha.py```
- **Input**: labels_train/val.pkl, biased_classes.pkl, biased_classes_mapped.pkl, humanlabels_to_onehot.pkl
- **Output**: weights_train/val.pkl
- **Description**: Calculates alphas and weights needed for feature splitting method's weighted loss.


## Training
```stage1.py```
- **Input**: labels_train.pkl, labels_val.pkl, unbiased_classes_mapped.pkl
- **Output**: Optimized model parameters
- **Description**: Trains a "standard" baseline classifier.

```stage2_cam.py``` (In development)
- **Description**: Does stage2 training with the CAM-method.

```stage2_featuresplit.py```
- **Input**: labels_train.pkl, labels_val.pkl, biased_classes_mapped.pkl, weight_train.pkl
- **Output**: Optimized model parameters
- **Description**: Does stage2 training with the feature splitting method.


## Evaluation
```evaluate.py```
- **Input**: biased_classes.pkl, humanlabels_to_onehot.pkl, path to saved model parameters
- **Output**: Scores saved in evalresults
- **Description**: Evaluates a trained model on the exclusive and co-occur test distributions.


## Utils
```load_data.py```
- **Description**: Creates dataset loader. Needed in stage1.py.

```classifier.py```
- **Description**: Defines the multi-label classifier class. Needed in stage1.py

```basenet.py```
- **Description**: Defines ResNet-50 backbone architecture. Needed in classifier.py

