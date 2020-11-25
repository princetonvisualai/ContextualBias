#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=4G      # memory per cpu-core (4G default)
#SBATCH --gres=gpu:1           # number of GPUs requested
#SBATCH -t 10:00:00            # time requested in hour:minute:second

#source /n/fs/context-scr/context/bin/activate
source /n/fs/visualai-scr/sunnie/basic/bin/activate

#python train_models.py --model baseline --batchsize 200 --outdir save
#python train_models.py --model cam --batchsize 50 --outdir save
#python train_models.py --model featuresplit --batchsize 100 --outdir save
#python train_models.py --model removeclabels --batchsize 100 --outdir save
#python train_models.py --model removecimages --batchsize 100 --outdir save
#python train_models.py --model negativepenalty --batchsize 100 --outdir save
python train_models.py --model featuresplit --batchsize 100 --outdir save
