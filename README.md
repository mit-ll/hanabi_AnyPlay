# Overview: Any-Play Learning Augmentation for Zero-Shot Coordination

![AnyPlay_diagram.png]()

This library implements the [Any-Play learning augmentation](https://arxiv.org/abs/2201.12436) in [Hanabi Learning Environment](https://github.com/hengyuan-hu/hanabi-learning-environment). Any-Play is an intrisictly-motivated, diversity-based augmentation for reinforcement learning algorithms (RL) that enables RL agents to effectively cooperate with novel, never-before-seen teammates on collaborative tasks; referred to as zero-shot coordination. This library is used to train and demonstrate such teaming in the cooperative card game Hanabi; although the Any-Play augmentation is environment agnostic, and therefore, could be applied to many other domains beyone Hanabi.

This library is a direct extension of the [`hanabi_SAD` library](https://github.com/facebookresearch/hanabi_SAD) and is modified and redistributed under the same [CC BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/)

# Citation

When using this library please cite the [Any-Play](https://arxiv.org/abs/2201.12436) paper as well as the orginal [Other-Play](https://arxiv.org/abs/2003.02979) and [Simplified Action Decoder](https://arxiv.org/abs/1912.02288) works on which this is based

Any-Play
```
@article{lucas2022any,
  title={Any-Play: An Intrinsic Augmentation for Zero-Shot Coordination},
  author={Lucas, Keane and Allen, Ross E},
  journal={arXiv preprint arXiv:2201.12436},
  year={2022}
}
```


Other-Play
```
@incollection{icml2020_5369,
 author = {Hu, Hengyuan and Peysakhovich, Alexander and Lerer, Adam and Foerster, Jakob},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {9396--9407},
 title = {\textquotedblleft Other-Play\textquotedblright  for Zero-Shot Coordination},
 year = {2020}
}

```

Simplfied Action Decoder
```
@inproceedings{
Hu2020Simplified,
title={Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning},
author={Hengyuan Hu and Jakob N Foerster},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=B1xm3RVtwB}
}
```

# Disclaimer

------

# Other-Play & Simplified Action Decoder in Hanabi

## Important Update, Mar-2021
We uploaded one off-belief-learning (OBL) model from our recent [
paper](https://arxiv.org/pdf/2103.04000.pdf). To get this model, go to
`hanabi_SAD/models` and run
```shell
wget https://dl.fbaipublicfiles.com/hanabi_op/obl.zip
unzip obl.zip
```
To use this model, go to `hanabi_SAD/pyhanabi` and run
```shell
python tools/eval_model.py --paper obl --num_game 5000
```

## Important Update, Feb-2021

We uploaded the models from the Other-Play paper. To get those models, run the
updated `download.sh` in the `models` folder. If you only need the Other-Play
models, you can download them by running the following command from the `models` folder
```shell
wget https://dl.fbaipublicfiles.com/hanabi_op/op.zip
unzip op.zip
```

We also include the model evaluation data in `models/op_raw_data.txt`. The data in
this file is used for Figure 4 and Table 1 in the paper.

We updated the evaluation script to allow both self-play and cross-play evaluation
using the new other-play models.
```shell
# assume current work directory is pyhanabi
# method can be sad, sad-op, sad-aux, sad-aux-op
# idx1/idx2 ranges from [0, 11], corresponding to the 12 models.
python tools/eval_model.py --paper op --method sad-aux --idx1 0 --idx2 0
```
The evaluation script assumes that the models are saved in the `$ROOT/models` folder.

The model used for human evaluation in the paper was `models/op/sad-aux-op/M1.pthw`, which
was the model with the highest cross-play score and trained with the best method.


## Important Update, Sep-2020

The repo has been updated to include Other-Play, auxiliary task, as well as improved
training infrastructure. The build process has also been significantly simplfied. It
is no longer necessary to build pytorch from source (thanks to changes in pytorch1.5)
and the code now works with newer version of pytorch and cuda.
It also avoids the hanging problem that may appear in
previous version of the codebase on certain hardware configuration.

## Introduction

This repo contains code and models for
["Other-Play" for Zero-Shot Coordination](https://arxiv.org/abs/2003.02979)
and [Simplified Action Decoder for Deep Multi-Agent
Reinforcement Learning](https://arxiv.org/abs/1912.02288).

To reference these works, please use:

Other-Play
```
@incollection{icml2020_5369,
 author = {Hu, Hengyuan and Peysakhovich, Alexander and Lerer, Adam and Foerster, Jakob},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {9396--9407},
 title = {\textquotedblleft Other-Play\textquotedblright  for Zero-Shot Coordination},
 year = {2020}
}

```

Simplfied Action Decoder
```
@inproceedings{
Hu2020Simplified,
title={Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning},
author={Hengyuan Hu and Jakob N Foerster},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=B1xm3RVtwB}
}
```

## Compile
We have been using `pytorch-1.5.1`, `cuda-10.1`, and `cudnn-v7.6.5` in our development environment.
Other settings may also work but we have not tested it extensively under different configurations.
We also use `conda/miniconda` to manage environments.
```bash
# create new conda env
conda create -n hanabi python=3.7
conda activate hanabi

# install pytorch
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install other dependencies
pip install numpy
pip install psutil

# if the current cmake version is < 3.15
conda install -c conda-forge cmake
```

### Clone & Build this repo
For convenience, add the following lines to your `.bashrc`,
after the line of `conda activate xxx`.

```bash
# set path
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1
```

Clone & build.
```bash
git clone --recursive https://github.com/facebookresearch/hanabi.git

cd hanabi
mkdir build
cd build
cmake ..
make -j10
```

## Run

`hanabi/pyhanabi/tools` contains some example scripts to launch training
runs. `dev.sh` is a fast lauching script for debugging. It needs 2 gpus to run,
1 for training and 1 for simulation. Other scripts are examples for a more formal
training run, they require 3 gpus, 1 for training and 2 for simulation.

The important flags are:

`--sad 1` to enable "Simplified Action Decoder";

`--pred_weight 0.25` to enable auxiliary task and multiply aux loss with 0.25;

`--shuffle_color 1` to enable other-play.

```bash
cd pyhanabi
sh tools/dev.sh
```

## Trained Models

Run the following command to download the trained models used to
produce tables in the paper.
```bash
cd model
sh download.sh
```
To evaluate a model, simply run
```bash
cd pyhanabi
python tools/eval_model.py --weight ../models/sad_2p_10.pthw --num_player 2
```

## Related Repos

The results on Hanabi can be further improved by running search on top
of our agents. Please refer to the [paper](https://arxiv.org/abs/1912.02318) and
[code](https://github.com/facebookresearch/Hanabi_SPARTA) for details.

We also open-sourced a single agent implementation of R2D2 tested on Atari
[here](https://github.com/facebookresearch/rela).

## Contribute

### Python
Use [`black`](https://github.com/psf/black) to format python code,
run `black *.py` before pushing

### C++
The root contains a `.clang-format` file that define the coding style of
this repo, run the following command before submitting PR or push
```bash
clang-format -i *.h
clang-format -i *.cc
```

## Copyright
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
