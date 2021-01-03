#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)
#SBATCH --gres=gpu:rtx_3090:2  # number of GPUs requested
#SBATCH -t 36:00:00            # time requested in hour:minute:second

source /n/fs/context-scr/context/bin/activate # for RTX3090
#source /n/fs/visualai-scr/sunnie/basic/bin/activate # for non-RTX3090

### COCO-Stuff
#python train.py --dataset COCOStuff --model cam --nepoch 100 --nclasses 170 \
#  --test_batchsize 150 --train_batchsize 200 \
#  --save COCOStuff/save

#python train.py --dataset COCOStuff --model baseline --batchsize 200 \
#    --outdir save/coco/lr0.1_wd0.00001_b200 --lr 0.1 --wd 0.00001

#python train.py --dataset COCOStuff --model baseline --batchsize 200 \
#  --outdir save/coco/lr0.05_wd0.00001_b200 --lr 0.05 --wd 0.00001

#python train.py --dataset COCOStuff --model baseline --batchsize 200 \
#  --outdir save/coco/lr0.01_wd0.00001_b200 --lr 0.01 --wd 0.00001

#python train.py --dataset COCOStuff --model baseline --batchsize 100 \
#  --outdir save/coco/lr0.1_wd0.00001_b100 --lr 0.1 --wd 0.00001

#python train.py --dataset COCOStuff --model baseline --batchsize 100 \
#  --outdir save/coco/lr0.05_wd0.00001_b100 --lr 0.05 --wd 0.00001

#python train.py --dataset COCOStuff --model baseline --batchsize 100 \
#    --outdir save/coco/lr0.01_wd0.00001_b100 --lr 0.01 --wd 0.00001


### AwA
#python train.py --dataset AwA --model splitbiased --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0001 --drop 10 \
#  --test_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_test /n/fs/context-scr/AwA/labels_val.pkl \
#  --outdir AwA/save/splitbiased2

python train.py --dataset AwA --model attribdecorr --nepoch 20 --nclasses 85 \
  --lr 0.001 --wd 0.0001 --drop 20 --compshare_lambda 10 \
  --test_batchsize 150 --train_batchsize 200 \
  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
  --labels_test /n/fs/context-scr/AwA/labels_val.pkl \
  --pretrainedpath AwA/save/baseline/model_19.pth \
  --outdir AwA/save

### DeepFashion
#python train.py --dataset DeepFashion --model baseline --nepoch 50 --nclasses 250 \
#  --lr 0.1 --test_batchsize 140 --drop 30 \
#  --labels_train /n/fs/context-scr/DeepFashion/labels_train.pkl \
#  --labels_test /n/fs/context-scr/DeepFashion/labels_val.pkl \
#  --outdir DeepFashion/save

#python train.py --dataset DeepFashion --model featuresplit --nepoch 50 --nclasses 250 \
#  --lr 0.1 --test_batchsize 140 --drop 30 \
#  --labels_train /n/fs/context-scr/DeepFashion/labels_train.pkl \
#  --labels_test /n/fs/context-scr/DeepFashion/labels_test.pkl \
#  --outdir DeepFashion/save/featuresplit_1536
