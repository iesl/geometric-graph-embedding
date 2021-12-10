#!/usr/bin/env bash

set -exu

sweep_id=$1
num_machines=${2:-20}
threads=${3:-2}
mem=${4:-10000}

TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

model_name="wandb"
dataset=$sweep_id
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/log

partition='1080ti-short'

mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --gres=gpu:1 \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --array=0-$num_machines \
            bin/run_sweep.sh $sweep_id $threads 100000
