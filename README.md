# Graph Modeling
This repository contains code which accompanies the paper [Capacity and Bias of Learned Geometric Embeddings for Directed Graphs (Boratko et al. 2021)](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html).

This code includes implementations of many geometric embedding methods:
- Vector Similarity and Distance
- Bilinear Vector Model [(Nickel et al. 2011)](https://openreview.net/forum?id=H14QEiZ_WS)
- ComplEx Embeddings [(Trouillon et al. 2016)](https://arxiv.org/abs/1606.06357)
- Order Embeddings [(Vendrov et al. 2015)](https://arxiv.org/abs/1511.06361) and Probabilistic Order Embeddings [(Lai and Hockenmaier 2017)](https://aclanthology.org/E17-1068.pdf)
- Hyperbolic Embeddings, including:
  - "Lorentzian" - uses the squared Lorentzian distance on the Hyperboloid as in [(Law et al. 2019)](http://proceedings.mlr.press/v97/law19a.html), trains undirected but uses the asymmetric score function from [(Nickel and Kiela 2017)](https://proceedings.neurips.cc/paper/2017/file/59dfa2df42d9e3d41f5b02bfc32229dd-Paper.pdf) to determine edge direction at inference
  - "Lorentzian Score" - uses the asymmetric score above directly in training loss 
  - "Lorentzian Distance" - Hyperbolic model for directed graphs as described in section 2.3 of [(Boratko et al. 2021)](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html)
- Hyperbolic Entailment Cones [(Ganea et al. 2018)](https://arxiv.org/abs/1804.01882)
- Gumbel Box Embeddings [(Dasgupta et al. 2020)](https://arxiv.org/abs/2010.04831)
- t-Box model as described in section 3 of [(Boratko et al. 2021)](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html)

It also provides a general-purpose pipeline to explore correlation between graph characteristics and models' learning capabilities.

## Installation

This repository makes use of submodules, to clone them you should use the `--recurse-submodules` flag, eg.
```bash
git clone <repo-url> --recurse-submodules
```
After cloning the repo, you should create an environment and install pytorch. For example,

```bash
conda create -n graph-modeling python=3.8
conda activate graph-modeling
conda install -c pytorch cudatoolkit=11.3 pytorch
```

You can then run `make all` to install the remaining modules and their dependencies. **Note:**
1. This will install Python modules, so you should run this command with the virtual environment created previously activated.
2. Certain graph generation methods (Kronecker and Price Network) will require additional dependencies to be compiled. In particular, Price requires that you use `conda`. If you are not interested in generating Kronecker or Price graphs you can skip this by using `make base` instead of `make all`.

## Usage

This module provides a command line interface available with `graph_modeling`.

Run `graph_modeling --help` to see available options.

### Generate Graphs
To generate a graph, run `graph_modeling generate <graph_type>`, eg. `graph_modeling generate scale-free-network`.

- `graph_modeling generate --help` provides a list of available graphs that can be generated
- `graph_modeling generate <graph_type> --help` provides a list of parameters for generation

By default, graphs will be output in `data/graphs`, using a subfolder for their graph type and parameter settings. You can override this with the `--outdir` parameter.

### Train Graph Representations
You can train graph representations using the `graph_modeling train` command, run `graph_modeling train --help` to see available options. The only required parameter is `--data_path`, which specifies either a specific graph file or a folder, in which case it will pick a graph in the folder uniformly randomly. The `--model` option allows for a selection of different embedding models. Most other options apply to every model (eg. `--dim`) or training in general (eg. `--log_batch_size`). Model-specific options are prefaced with the model name (eg. `--box_intersection_temp`). Please see the help text for the options for more details, and submit an issue if anything is unclear.

## Citation
If you found the code contained in this repository helpful in your research, please cite the following paper:

```
@inproceedings{boratko2021capacity,
  title={Capacity and Bias of Learned Geometric Embeddings for Directed Graphs},
  author={Boratko, Michael and Zhang, Dongxu and Monath, Nicholas and Vilnis, Luke and Clarkson, Kenneth L and McCallum, Andrew},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```


