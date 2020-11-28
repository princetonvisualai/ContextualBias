#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH -w node017
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)
#SBATCH --gres=gpu:rtx_3090:1  # number of GPUs requested
#SBATCH -t 60:00:00            # time requested in hour:minute:second

source /n/fs/context-scr/context/bin/activate
#source /n/fs/visualai-scr/sunnie/basic/bin/activate

#python train.py --model baseline --batchsize 200 --outdir save
#python train.py --model cam --batchsize 50 --outdir save
#python train.py --model featuresplit --batchsize 100 --outdir save
#python train.py --model removeclabels --batchsize 100 --outdir save
#python train.py --model removecimages --batchsize 100 --outdir save
#python train.py --model negativepenalty --batchsize 100 --outdir save
#python train.py --model featuresplit --batchsize 100 --outdir save

python train.py --dataset AwA --model baseline --batchsize 200 --nclasses 85 --labels_train /n/fs/context-scr/AwA/labels_train.pkl --labels_val /n/fs/context-scr/AwA/labels_val.pkl --learning_rate 0.1 --outdir AwA/save/baseline2
