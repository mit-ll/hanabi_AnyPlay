#!/bin/bash

# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: CC-BY-NC-4.0

python tools/eval_model.py \
       --weight exps/pid_2p_2ffskip \
       --num_player 2 \ 
       --paper prebrief \
       --total_game 100000
