#!/bin/sh
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # number of tasks requested
#SBATCH --ntasks-per-node 1    # number of tasks per node
#SBATCH --exclude=node718      # exclude the node that often causes errors
#SBATCH -A visualai            # specify which group of nodes to use
#SBATCH --mem-per-cpu=16G      # memory per cpu-core (4G default)
#SBATCH --gres=gpu:1           # number of GPUs requested
#SBATCH -t 5:00:00             # time requested in hour:minute:second

source /n/fs/context-scr/context/bin/activate

python data_process.py

# python create_evaldata.py
#
# python calculate_alpha.py
#
# python stage1.py
#
# python stage2_cam.py
#
# python stage2_featuresplit.py
#
# python evaluate.py
