# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python tools/eval_model.py \
       --weight exps/pid_2p_2ffskip \
       --num_player 2 \ 
       --paper prebrief \
       --total_game 100000
