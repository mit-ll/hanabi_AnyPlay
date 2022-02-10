#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#SBATCH --gres=gpu:volta:1
#SBATCH -p gaia 
#SBATCH -o slurm_out/out_ext_xp_awa32_2p.txt
#SBATCH -e slurm_err/err_ext_xp_awa32_2p.txt
#SBATCH --job-name=ext_xp_awa32_2p
#SBATCH --exclusive

eval "$(conda shell.bash hook)";
conda activate py38;
python -u tools/eval_model.py --expfolder exps_awa32/ --expfolder ../models/ --num_player 2 --paper prebrief --total_game 2500 --save_file paired_results_jsons/ext_paired_results_awa32.json