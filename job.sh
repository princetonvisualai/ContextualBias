#!/bin/sh

# 1. Process datasets
python COCOStuff/data_process.py \
  --coco2014_images Data/Coco/2014data \
  --cocostuff_annotations Data/cocostuff/dataset/annotations

python DeepFashion/data_process.py \
  --datadir Data/DeepFashion/Category\ and\ Attribute\ Prediction\ Benchmark

python AwA/data_process.py --datadir Data/AwA2

python UnRel/data_process.py --datadir Data/UnRel

python split_80_20.py --labels_train COCOStuff/labels_train.pkl \
  --labels_train_20 COCOStuff/labels_train_20.pkl --labels_train_80 COCOStuff/labels_train_80.pkl

python split_80_20.py --labels_train AwA/labels_train.pkl \
  --labels_train_20 AwA/labels_train_20.pkl --labels_train_80 AwA/labels_train_80.pkl

# 2. Train a standard model
python train.py --dataset COCOStuff --nclasses 171 --model standard --outdir models/COCOStuff \
  --labels_train COCOStuff/labels_train_80.pkl --labels_val COCOStuff/labels_train_20.pkl \
  --nepoch 100 --lr 0.1 --drop 60 --wd 0 --momentum 0.9

python train.py --dataset DeepFashion --nclasses 250 --model standard --outdir models/DeepFashion \
  --labels_train DeepFashion/labels_train.pkl --labels_val DeepFashion/labels_val.pkl \
  --nepoch 50 --lr 0.1 --drop 30 --wd 0 --momentum 0.9

python train.py --dataset AwA --nclasses 85 --model standard --outdir models/AwA \
  --labels_train AwA/labels_train.pkl --labels_val AwA/labels_test.pkl \
  --nepoch 20 --lr 0.1 --drop 10 --wd 0 --momentum 0.9

# 3. Identify biased categories
python biased_categories.py --dataset COCOStuff --nclasses 171 --cooccur 0.2 \
  --labels_20 COCOStuff/labels_train_20.pkl --labels_80 COCOStuff/labels_train_80.pkl \
  --modelpath models/COCOStuff/standard/model_100.pth

python biased_categories.py --dataset DeepFashion --nclasses 250 --cooccur 0.1 \
  --labels_20 DeepFashion/labels_val.pkl --labels_80 DeepFashion/labels_train.pkl \
  --modelpath models/DeepFashion/standard/model_50.pth

python biased_categories.py --dataset AwA --nclasses 85 --cooccur 0.2 \
  --labels_20 AwA/labels_test.pkl --labels_80 AwA/labels_train.pkl \
  --modelpath models/AwA/standard/model_20.pth

# 4. Train stage 2 models
python train.py --dataset COCOStuff --nclasses 171 --model cam --outdir models/COCOStuff \
  --labels_train COCOStuff/labels_train.pkl --labels_val COCOStuff/labels_test.pkl \
  --nepoch 20 --lr 0.01 --drop 999 --wd 0 --momentum 0.9

python train.py --dataset DeepFashion --nclasses 250 --model featuresplit --outdir models/DeepFashion \
  --labels_train DeepFashion/labels_train.pkl --labels_val DeepFashion/labels_val.pkl \
  --nepoch 20 --lr 0.01 --drop 999 --wd 0 --momentum 0

python train.py --dataset AwA --nclasses 85 --model negativepenalty --outdir models/AwA \
  --labels_train AwA/labels_train.pkl --labels_val AwA/labels_test.pkl \
  --nepoch 20 --lr 0.01 --drop 999 --wd 0 --momentum 0.9

# 5. Evaluate models and analyze results
python evaluate.py --dataset COCOStuff --nclasses 171 --model cam \
  --modelpath models/COCOStuff/cam/model_20.pth \
  --labels_test COCOStuff/labels_test.pkl

python evaluate_unrel.py --modelpath models/COCOStuff/cam/model_20.pth

python weight_similarity.py --dataset COCOStuff --nclasses 171 \
  --modelpath models/COCOStuff/featuresplit/model_20.pth

python get_prediction_examples.py --dataset COCOStuff --nclasses 171 \
  --labels_test COCOStuff/labels_test.pkl \
  --standard_modelpath models/COCOStuff/standard/model_100.pth \
  --fs_modelpath models/COCOStuff/featuresplit/model_20.pth \
  --b car --outdir prediction_examples

python get_cams.py --coco2014_images Data/Coco/2014data \
  --img_ids 535811 430054 554674 --outdir CAM_examples \
  --modelpath models/COCOStuff/cam/model_20.pth
