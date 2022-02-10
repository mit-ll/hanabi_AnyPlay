#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

python -u selfplay_reloader.py \
       --save_dir exps_plot2/sad_aux_2ffsk \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method vdn \
       --sad 1 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 123456 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 3 \
       --act_device cuda:0,cuda:1 \
       --pred_weight 0.25 \
       --intent_weight 0 \
       --num_ff_layer 2 \
       --skip_connect \
