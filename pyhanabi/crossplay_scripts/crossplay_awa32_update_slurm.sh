#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#SBATCH --gres=gpu:volta:1
#SBATCH -p gaia 
#SBATCH -o slurm_out/out_crossplay_awa32_2p.txt
#SBATCH -e slurm_err/err_crossplay_awa32_2p.txt
#SBATCH --job-name=crossplay_awa32_2p
#SBATCH --exclusive

eval "$(conda shell.bash hook)";
conda activate py38;
python -u tools/eval_model.py --expfolder exps_awa32/ --expfolder ../models/op/ --num_player 2 --paper prebrief --total_game 2500 --save_file paired_results_jsons/paired_results_awa32.json