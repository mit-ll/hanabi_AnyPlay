# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import argparse
import pprint
import gc

import numpy as np
import torch
from torch import nn

import common_utils
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_obs", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--pred_weight", type=float, default=0)
    parser.add_argument("--use_pred_reward", action="store_true", default=False)
    parser.add_argument("--intent_weight", type=float, default=0)
    parser.add_argument("--intent_pred_input", type=str, default='lstm_o')
    parser.add_argument("--intent_arch", type=str, default='concat')
    
    parser.add_argument("--num_eps", type=int, default=80)

    parser.add_argument("--load_model", type=str, default="")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--eta", type=float, default=0.9, help="eta for aggregate priority")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)
    parser.add_argument("--intent_size", type=int, default=0)
    parser.add_argument("--use_xent_intent", action="store_true", default=False)
    parser.add_argument("--dont_onehot_xent_intent", action="store_true", default=False)
    parser.add_argument("--one_way_intent", action="store_true", default=False)
    parser.add_argument("--train_adapt", action="store_true", default=False)
    parser.add_argument("--use_player_id", action="store_true", default=False, \
                        help="whether to provide player ID as observations to agents")
    parser.add_argument("--shuf_pid", action="store_true", default=False, \
                        help="shuffles player ID; use_player_id must be True")

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument("--num_ff_layer", type=int, default=1)
    parser.add_argument("--skip_connect", action="store_true", default=False)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=500)#5000) #changing default to be shorter to get faster crossplay results
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    assert args.method in ["vdn", "iql"]
    assert not args.train_adapt or len(args.load_model) > 0
    return args


if __name__ == "__main__":
    args = parse_args()

    success = False
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    # sys.stdout = common_utils.Logger(logger_path)
    while not success:
        success = True
        gc.collect()

        arg_dict = vars(args)

        # call subprocess
        command_list = ["python", "-u", "selfplay.py"]

        for key in arg_dict.keys():
            if type(arg_dict[key]) is bool and not arg_dict[key]:
                continue
            elif type(arg_dict[key]) is bool and arg_dict[key]:
                command_list.append("--%s"%key)
            else:
                command_list.append("--%s"%(key))
                command_list.append("%s"%arg_dict[key])

        # command_args = " ".join(["--%s %s"%(key,arg_dict[key]) if arg_dict[key]=="True" else "--%s %s"%(key,arg_dict[key]) for key in arg_dict.keys()])
        # command_list += command_args.split(" ")
        # print(command_list)
        subprocess.call(command_list)

        last_line = ""
        if os.path.exists(logger_path):
            with open(logger_path, 'rb') as rf:        
                rf.seek(0,os.SEEK_END)
                # while rf.tell() > 0 and rf.read(7) != b'RELOAD:':
                while rf.tell() > 0 and rf.read(7) != b'RELOAD:':
                    # lol = rf.read(7) 
                    # print(lol)
                    # if lol :
                    #     break
                    rf.seek(rf.tell()-8)
                rf.seek(rf.tell()-7)
                if rf.tell() > 0:
                    last_line = rf.readline().decode().strip()
        if "LOAD:" in last_line:
            args.intent_weight = float(last_line.split("intent_weight to ")[1])
            args.seed = int(args.seed) + 1000000 #change seed too
            success = False

        gc.collect()
