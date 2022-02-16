#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

python -u tools/eval_model.py --expfolder exps_awa64/ --expfolder ../models/op/ --num_player 2 --paper prebrief --total_game 2500 --save_file paired_results_jsons/paired_results_awa64.json