#!/bin/bash


# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: CC-BY-NC-4.0

python -u tools/eval_model.py --expfolder exps_awa64/ --expfolder ../models/ --num_player 2 --paper prebrief --total_game 2500 --save_file paired_results_jsons/ext_paired_results_awa64.json
