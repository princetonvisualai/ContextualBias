#!/bin/sh

# 1. Process datasets
python COCOStuff/data_process.py
python DeepFashion/data_process.py
python AwA/data_process.py
python UnRel/data_process.py

python split_80_20.py --labels_train COCOStuff/labels_train.pkl \
  --labels_train_20 COCOStuff/labels_train_20.pkl --labels_80 COCOStuff/labels_train_80.pkl

python split_80_20.py --labels_train AwA/labels_train.pkl \
  --labels_train_20 AwA/labels_train_20.pkl --labels_80 AwA/labels_train_80.pkl

# 2. Train a standard model
python train.py --dataset COCOStuff --nclasses 171 --model standard --outdir models/COCOStuff \
  --labels_train COCOStuff/labels_train_80.pkl --labels_val COCOStuff/labels_train_20.pkl \
  --biased_classes_mapped COCOStuff/biased_classes_mapped.pkl \
  --unbiased_classes_mapped COCOStuff/unbiased_classes_mapped.pkl \
  --humanlabels_to_onehot COCOStuff/humanlabels_to_onehot.pkl \
  --nepoch 100 --lr 0.1 --drop 60 --wd 0 --momentum 0.9

python train.py --dataset DeepFashion --nclasses 250 --model standard --outdir models/DeepFashion \
  --labels_train DeepFashion/labels_train.pkl --labels_val DeepFashion/labels_val.pkl \
  --biased_classes_mapped DeepFashion/biased_classes_mapped.pkl \
  --humanlabels_to_onehot DeepFashion/humanlabels_to_onehot.pkl \
  --nepoch 50 --lr 0.1 --drop 30 --wd 0 --momentum 0.9

python train.py --dataset AwA --nclasses 85 --model standard --outdir models/AwA \
  --labels_train AwA/labels_train.pkl --labels_val AwA/labels_test.pkl \
  --biased_classes_mapped AwA/biased_classes_mapped.pkl \
  --humanlabels_to_onehot AwA/humanlabels_to_onehot.pkl \
  --nepoch 20 --lr 0.1 --drop 10 --wd 0 --momentum 0.9

# 3. Identify biased categories
python biased_categories.py --dataset COCOStuff --nclasses 171 --cooccur 0.2 \
  --labels_20 COCOStuff/labels_train_20.pkl --labels_80 COCOStuff/labels_train_80.pkl \
  --modelpath models/COCOStuff/standard/model_99.pth

python biased_categories.py --dataset DeepFashion --nclasses 250 --cooccur 0.1 \
  --labels_20 DeepFashion/labels_val.pkl --labels_80 DeepFashion/labels_train.pkl \
  --modelpath models/DeepFashion/standard/model_49.pth

python biased_categories.py --dataset AwA --nclasses 85 --cooccur 0.2 \
  --labels_20 AwA/labels_test.pkl --labels_80 COCOStuff/labels_train.pkl \
  --modelpath models/AwA/standard/model_19.pth

# 4. Train models


# 5. Evaluate models and analyze results
python evaluate.py

python get_cams.py

python weight_similarity.py
