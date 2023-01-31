# Graph Modeling Scripts
In this directory are some small self-contained scripts used for orchestrating the sweeps for Bayesian hyperparameter tuning with Weights and Biases, submitting jobs on SLURM, and collating the data.

To submit multiple sweeps given a two-level directory of graph data:
`python scripts/auto_sweep.py --data_path data/graphs/ --model_type=box --dim=16 --partition=1080ti-long --max_run 100`
This will also save sweep configs to ./sweeps_config/

(There is also a file [box-training-methods](box-training-methods) which provides a script-like interface to the `box_training_methods` command resulting from installing this module. This is the entry point we use with wandb, and can also be useful as a target for your debugger.)
