# [Re] Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias

This is a non-official implementation of [Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias (CVPR 2020)](https://arxiv.org/abs/2001.03152). We developed this codebase to reproduce the experiments in the paper, as part of our participation in the [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020).

## Dependencies

- Python 3 (e.g. conda create -n contextualbias python=3.7.3)
- pytorch, torch, torchvision, numpy, scipy, sklearn, PIL, cv2, matplotlib, pickle, collections, time, argparse

## Usage

We provide an example job script ```job.sh``` that contains the script execution order with example commands.

## Code overview

We provide a brief description of the individual scripts.

### Data processing
```{Dataset}/data_process.py```: Processes the COCO-Stuff, DeepFashion, AwA, UnRel datasets

```split_80_20.py```: Does a 80-20 split of the COCO-Stuff/AwA training set to create a validation set

### Biased categories identification
```biased_categories.py```: Calculates bias and identifies the K=20 most biased categories

### Training
```train.py```: Trains various models (standard, cam, featuresplit, removeclabels, removecimages, splitbiased, weighted, negativepenalty, classbalancing, attribdecorr)

### Evaluation
```evaluate.py```: Evaluates a trained model on the COCO-Stuff, DeepFashion, AwA datasets, on their exclusive and co-occur test distributions

```evaluate_unrel.py```: Evaluates a trained model on the UnRel dataset

```weight_similarity.py```: Calculates the cosine similarity between W_o and W_s to verify that they capture distinct information

```get_cams.py```: Saves class activation maps (CAMs) to understand what the model is looking at
- **Image IDs for Figure 2**: Skateboard (317040), Microwave (191632)
- **Image IDs for Figure 6**: Handbag (167235, 37124), Snowboard (423602, 581921), Car (574087, 119802), Spoon (227858, 42526), Remote (390829)

```get_prediction_examples.py```: Finds successful and unsuccessful image examples of a model's prediction for a category b
- **Image IDs for Figure 5**: Skateboard (175612, 198043, 292789, 300842), Microwave (47873, 68833, 332480, 568281), Snowboard (50482, 174103, 435894, 422328)

### Utils
```classifier.py```: Defines a multi-label classifier with various training methods

```load_data.py```: Creates dataset loaders and calculates loss weights for class-balancing and feature-split methods

```recall.py```: Defines the recall@3 function for DeepFashion evaluation
