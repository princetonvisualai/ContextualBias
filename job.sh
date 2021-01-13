#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)
#SBATCH --gres=gpu:rtx_3090:1  # number of GPUs requested
#SBATCH -t 9:00:00             # time requested in hour:minute:second

#SBATCH --mail-type=end,fail
#SBATCH --mail-user=sharonz@princeton.edu

source /n/fs/context-scr/context/bin/activate # for RTX3090
#source /n/fs/visualai-scr/sunnie/basic/bin/activate # for non-RTX3090

### COCO-Stuff
python train.py --dataset COCOStuff --model cam --nepoch 2 --nclasses 171 \
  --modelpath /n/fs/context-scr/sunnie/COCOStuff/lr0.1_wd0_drop60/baseline/model_67.pth \
  --val_batchsize 150 --train_batchsize 100 \
  --outdir COCOStuff/save/cam_test

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
# DONE
#python train.py --dataset AwA --model baseline --nepoch 40 --nclasses 85 \
#  --lr 0.1 --wd 0.0 --drop 10 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train_80.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --outdir AwA/save3

#python train.py --dataset AwA --model featuresplit --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 99 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3

#python train.py --dataset AwA --model featuresplit --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 --split 256 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_19.pth \
#  --outdir AwA/save2/featuresplit_256

#python train.py --dataset AwA --model removecimages --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3

#python train.py --dataset AwA --model removeclabels --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3

#python train.py --dataset AwA --model negativepenalty --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3

#python train.py --dataset AwA --model splitbiased --nepoch 40 --nclasses 85 \
#  --lr 0.1 --wd 0.0 --drop 10 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --outdir AwA/save3

#python train.py --dataset AwA --model classbalancing --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3

#python train.py --dataset AwA --model weighted --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --modelpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3

#python train.py --dataset AwA --model attribdecorr --nepoch 20 --nclasses 85 \
#  --lr 0.01 --wd 0.0 --drop 20 --compshare_lambda 2.0 \
#  --val_batchsize 150 --train_batchsize 200 \
#  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
#  --labels_val /n/fs/context-scr/AwA/labels_train_20.pkl \
#  --pretrainedpath /n/fs/context-scr/sharonz/ContextualBias/AwA/save2/baseline/model_20.pth \
#  --outdir AwA/save3


### DeepFashion
#python train.py --dataset DeepFashion --model baseline --nepoch 50 --nclasses 250 \
#  --lr 0.1 --val_batchsize 140 --drop 30 \
#  --labels_train /n/fs/context-scr/DeepFashion/labels_train.pkl \
#  --labels_val /n/fs/context-scr/DeepFashion/labels_val.pkl \
#  --outdir DeepFashion/save

#python train.py --dataset DeepFashion --model featuresplit --nepoch 50 --nclasses 250 \
#  --lr 0.1 --val_batchsize 140 --drop 30 \
#  --labels_train /n/fs/context-scr/DeepFashion/labels_train.pkl \
#  --labels_test /n/fs/context-scr/DeepFashion/labels_test.pkl \
#  --outdir DeepFashion/save/featuresplit_1536
