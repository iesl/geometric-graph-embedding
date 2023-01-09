#!/usr/bin/env bash
# Copyright 2021 The Geometric Graph Embedding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


set -exu


sweep_id=$1
partition=$2
max_run=${3:-100}
num_machines=${4:-0}
threads=${5:-1}
mem=${6:-25000}


TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

model_name="wandb"
dataset=$sweep_id
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/log

partition=$partition

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
            --exclude=node026,node030,node040,node057,node059,node072,node095,node099,node123,node125,node167,node169,node176 \
            bin/run_sweep.sh $sweep_id $threads $max_run
