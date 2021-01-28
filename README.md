# [Re] Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias

This is a non-official implementation of [Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias (CVPR 2020)](https://arxiv.org/abs/2001.03152). We developed this codebase to reproduce the experiments in the paper, as part of our participation in the [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020).

## Data processing
```data_process.py```
- **Description**: Processes datasets

```split_80_20.py```
- **Description**: Do a 80-20 split of the COCO/AwA training set to create a validation set.

## Biased categories identification
```biased_categories.py```
- **Description**: Calculates bias and identifies the K=20 most biased categories

## Training
```train.py```
- **Description**: Trains various models (baseline, cam, featuresplit, removeclabels, removecimages, negativepenalty, classbalancing)

## Evaluation
```evaluate.py```
- **Description**: Evaluates a trained model on the exclusive and co-occur test distributions.

```evaluate_unrel.py```
- **Description**: Evaluates UnRel dataset on the exclusive and co-occur test distributions, prints mAP

```get_cams.py```
- **Description**: Visualize the CAM heatmap to understand what the model is learning
- **Image IDs for Figure 2**: Skateboard (317040), Microwave (191632)
- **Image IDs for Figure 5**:
- **Image IDs for Figure 6**: Handbag (167235, 37124), Snowboard (423602, 581921), Car (574087, 119802), Spoon (227858, 42526), Remote (390829)

```weight_similarity.py```
- **Description**: Calculating cosine similarity between W_o and W_s to verify that they capture distinct information

## Utils
```load_data.py```
- **Description**: Creates dataset loaders and calculates loss weights for class-balancing and feature-split methods.

```classifier.py```
- **Description**: Defines the multi-label classifier with various training methods.

```recall.py```
- **Description**: Top 3 recall function for DeepFashion evaluation
