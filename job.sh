#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)
#SBATCH --gres=gpu:rtx_3090:1  # number of GPUs requested
#SBATCH -t 100:00:00           # time requested in hour:minute:second

source /n/fs/context-scr/context/bin/activate # for RTX3090
# source /n/fs/visualai-scr/sunnie/basic/bin/activate # for non-RTX3090

python train.py --model baseline --batchsize 200 --outdir save

# python train.py --model removeclabels --batchsize 200 --outdir save
# python train.py --model removecimages --batchsize 200 --outdir save
# python train.py --model negativepenalty --batchsize 200 --outdir save
# python train.py --model classbalancing --batchsize 200 --outdir save

# python train.py --model cam --batchsize 50 --outdir save \
#   --pretrainedpath /n/fs/context-scr/COCOStuff/save/stage1/stage1_4.pth
# python train.py --model featuresplit --batchsize 200 --outdir save
