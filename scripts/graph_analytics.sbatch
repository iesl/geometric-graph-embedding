#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods
python3 src/box_training_methods/graph_modeling/graph_analytics.py