#!/bin/bash


# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

python -u tools/eval_model.py --expfolder ../models/ --num_player 2 --paper prebrief --total_game 2500 --save_file ext_prior_crossplay_results.json
