#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)
#SBATCH --gres=gpu:rtx_3090:1  # number of GPUs requested
#SBATCH -t 60:00:00            # time requested in hour:minute:second

source /n/fs/context-scr/context/bin/activate # for RTX3090
#source /n/fs/visualai-scr/sunnie/basic/bin/activate # for non-RTX3090

### COCO-Stuff
# python train.py --dataset COCOStuff --model baseline --batchsize 200 \
#     --save weight_decay/COCOStuff/save/0



### AwA
python train.py --dataset AwA --model baseline --nepoch 50 --batchsize 200 --nclasses 85 \
  --lr 0.01 --wd 0.0001 \
  --labels_train /n/fs/context-scr/AwA/labels_train.pkl \
  --labels_val /n/fs/context-scr/AwA/labels_val.pkl \
  --outdir AwA/save/baseline_2

### DeepFashion
#python train.py --dataset DeepFashion --model baseline --nepoch 50 --batchsize 200 --nclasses 250 \
#  --lr 0.1 --wd 0.0001 \
#  --labels_train /n/fs/context-scr/DeepFashion/labels_train.pkl \
#  --labels_val /n/fs/context-scr/DeepFashion/labels_val.pkl \
#  --outdir DeepFashion/save/baseline_lr2
