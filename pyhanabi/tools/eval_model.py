# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import argparse
import sys, time, os

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import numpy as np
import torch
import r2d2
import utils
from eval import evaluate
from obl_model import obl_model
import json, hashlib


def load_sad_model(weight_files, device):
    agents = []
    for weight_file in weight_files:
        if "sad" in weight_file or "aux" in weight_file:
            sad = True
        else:
            sad = False

        state_dict = torch.load(weight_file, map_location=device)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        agent = r2d2.R2D2Agent(
            False, 3, 0.999, 0.9, device, input_dim, hid_dim, output_dim, 2, 5, 3, False, num_player=len(weight_files)
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents


def load_prebrief_model(weight_files, device, recv_or_send='recv'):
    agents = []
    max_intent_size = 0
    for weight_file in weight_files:
        if "obl.pthw" in weight_file:
            agents.append(obl_model)
            continue
        
        if "all_obl_models" in weight_file:
            state_dict = torch.load(weight_file)
            if "core_ffn.1.weight" in state_dict:
                state_dict.pop("core_ffn.1.weight")
                state_dict.pop("core_ffn.1.bias")
                state_dict.pop("core_ffn.3.weight")
                state_dict.pop("core_ffn.3.bias")
                state_dict.pop("pred_2nd.weight")
                state_dict.pop("pred_2nd.bias")
                state_dict.pop("pred_t.weight")
                state_dict.pop("pred_t.bias")

            obl_model.online_net.load_state_dict(state_dict)
            obl_model.sync_target_with_online()
            agents.append(obl_model)
            continue

        sad = "sad" in weight_file or "aux" in weight_file
        iql = "iql" in weight_file
        vdn = "vdn" in weight_file
        prior_model = "models" in weight_file
        cfg = utils.get_train_config(weight_file)
        if cfg is None:
            cfg = {}
        pid = "use_player_id" in cfg.keys() and cfg["use_player_id"]
        prebrief = "intent_size" in cfg.keys() and cfg["intent_size"] > 0
        
        num_ff_layer = 1 if "num_ff_layer" not in cfg.keys() else cfg["num_ff_layer"]
        skip_connect = "skip_connect" in cfg.keys() and cfg["skip_connect"]
        if prior_model and 'op/' in weight_file:
            idx = int(os.path.basename(weight_file).replace(".pthw","").replace("M",""))
            if idx >= 0 and idx < 3:
                num_ff_layer = 1
                skip_connect = False
            elif idx >= 3 and idx < 6:
                num_ff_layer = 1
                skip_connect = True
            elif idx >= 6 and idx < 9:
                num_ff_layer = 2
                skip_connect = False
            else:
                num_ff_layer = 2
                skip_connect = True

        if prior_model and 'sad_models/' in weight_file:
            num_ff_layer = 1
            skip_connect = False

        success = False
        state_dict = None
        tries = 0
        max_tries = 10
        while not success and tries < max_tries:
            try:
                state_dict = torch.load(weight_file, map_location=device)
                success = True
            except Exception as e:
                tries += 1
                print("Tried to load %s, retrying... "%weight_file)
                time.sleep(1)
                print(e)
                # if type(e) is EOFError:

        if "net.0.weight" in state_dict.keys():
            input_dim = state_dict["net.0.weight"].size()[1]
            hid_dim = state_dict["net.0.weight"].size()[0]
        elif "priv_net.0.weight" in state_dict.keys():
            input_dim = state_dict["priv_net.0.weight"].size()[1]
            hid_dim = state_dict["priv_net.0.weight"].size()[0]
        else:
            assert False, "missing first layer name"

        output_dim = state_dict["fc_a.weight"].size()[0]
        intent_size = 0 if "intent_size" not in cfg.keys() else cfg["intent_size"]
        intent_weight = 0. if "intent_weight" not in cfg.keys() else cfg["intent_weight"]
        max_intent_size = max(max_intent_size, intent_size)
        dont_onehot_xent_intent = "dont_onehot_xent_intent" not in cfg.keys() or cfg["dont_onehot_xent_intent"]
        use_xent_intent = "use_xent_intent" in cfg.keys() and cfg["use_xent_intent"]
        one_way_intent = "one_way_intent" in cfg.keys() and cfg["one_way_intent"]
        shuf_pid = "shuf_pid" in cfg.keys() and cfg["shuf_pid"]
        intent_pred_input = "lstm_o" if "intent_pred_input" not in cfg.keys() else cfg["intent_pred_input"]
        intent_arch = "concat" if "intent_arch" not in cfg.keys() else cfg["intent_arch"]
        player_embed_dim = 8 if pid else 0

        agent = r2d2.R2D2Agent(
            False, 3, 0.999, 0.9, device, input_dim-intent_size-player_embed_dim, \
            hid_dim, output_dim, 2, 5, intent_size, False, num_player=len(weight_files) if pid else None, \
            num_ff_layer=num_ff_layer, skip_connect=skip_connect, use_xent_intent=use_xent_intent, \
            intent_weight=intent_weight, dont_onehot_xent_intent=dont_onehot_xent_intent, one_way_intent=one_way_intent, \
            recv_or_send=recv_or_send, shuf_pid=shuf_pid,
            intent_pred_input=intent_pred_input, intent_arch=intent_arch
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents, max_intent_size

def load_op_model(method, idx1, idx2, device):
    """load op models, op models was trained only for 2 player
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    folder = os.path.join(root, "models", "op", method)
    agents = []
    for idx in [idx1, idx2]:
        if idx >= 0 and idx < 3:
            num_ff = 1
            skip_connect = False
        elif idx >= 3 and idx < 6:
            num_ff = 1
            skip_connect = True
        elif idx >= 6 and idx < 9:
            num_ff = 2
            skip_connect = False
        else:
            num_ff = 2
            skip_connect = True
        weight_file = os.path.join(folder, f"M{idx}.pthw")
        if not os.path.exists(weight_file):
            print(f"Cannot find weight at: {weight_file}")
            assert False

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]
        agent = r2d2.R2D2Agent(
            False,
            3,
            0.999,
            0.9,
            device,
            input_dim,
            hid_dim,
            output_dim,
            2,
            5,
            3,
            False,
            num_player=2,
            num_ff_layer=num_ff,
            skip_connect=skip_connect,
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents


def evaluate_agents(agents, num_game, seed, bomb, device, num_run=1, verbose=True, intent_size=0):
    num_player = len(agents)
    assert num_player > 1, "1 weight file per player"

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            True,  # in op paper, sad was a default
            device=device,
            intent_size=intent_size,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / len(scores)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate, "num games: %d"%(len(scores)))
    return mean, sem, perfect_rate

def path_to_model(exp_path):
    prior_models = "models" in exp_path
    return exp_path

def model_name(exp_path):
    if "models/op" in exp_path:
        return exp_path[exp_path.index("models")+10:].replace("/","_")
    elif "models/sad_models" in exp_path:
        return exp_path[exp_path.index("models")+18:].replace("/","_")
    elif "models/all_obl_models" in exp_path:
        afterOBL = exp_path.split("icml_OBL")[1]
        belief_level = int(afterOBL[0])
        folder = exp_path.split("/model0.")[0]
        model_id = folder[-1]
        return "OBL%d_%s"%(belief_level, model_id)
    elif "models/obl" in exp_path:
        return exp_path[exp_path.index("models")+11:].replace("/","_")
    else:
        return os.path.basename(os.path.dirname(exp_path))


def evaluate_agent_pairs(exp_dict, num_game, seed, bomb, device, num_run=1, prior_models=None, verbose=True, save_file="paired_results.json", recv_or_send="recv"):
    # Select which weight_files to pair up
    exps_to_pair = []
    model_hashes = {}
    for exp in exp_dict.keys():
        if "model0.pthw" in exp_dict[exp] and exp.split("_")[-1].isnumeric():
            load_success = False
            model_num = 0
            exp_file = os.path.join(exp,"model%d.pthw"%model_num)
            while not load_success and model_num < 5:
                try:
                    torch.load(exp_file, map_location=device)
                    load_success = True
                except:
                    print(exp_file + "failed to load. Trying diff one")
                    model_num += 1
                    exp_file = os.path.join(exp,"model%d.pthw"%model_num)
                    load_success = False
            assert model_num < 5, "Could not find loadable model"
            with open(exp_file, "rb") as rf:
                model_hashes[model_name(exp_file)] = str(hashlib.sha1(rf.read()).hexdigest())
            exps_to_pair.append(exp_file)
        elif "M0.pthw" in exp_dict[exp]:
            for i in range(12):
                exp_file = os.path.join(exp,"M%d.pthw"%i)
                with open(exp_file, "rb") as rf:
                    model_hashes[model_name(exp_file)] = str(hashlib.sha1(rf.read()).hexdigest())
                exps_to_pair.append(exp_file)
        elif "models/sad_models" in exp:
            for exp_basename in exp_dict[exp]:
                exp_file = os.path.join(exp,exp_basename)
                if '_2p_' in exp_file:
                    with open(exp_file, "rb") as rf:
                        model_hashes[model_name(exp_file)] = str(hashlib.sha1(rf.read()).hexdigest())
                    exps_to_pair.append(exp_file)
        elif "models/all_obl_models" in exp:
            assert 'model0.pthw' in exp_dict[exp]
            exp_file = os.path.join(exp, 'model0.pthw')
            with open(exp_file, "rb") as rf:
                model_hashes[model_name(exp_file)] = str(hashlib.sha1(rf.read()).hexdigest())
            exps_to_pair.append(exp_file)
        elif "models/obl" in exp:
            assert 'obl.pthw' in exp_dict[exp]
            exp_file = os.path.join(exp, 'obl.pthw')
            with open(exp_file, "rb") as rf:
                model_hashes[model_name(exp_file)] = str(hashlib.sha1(rf.read()).hexdigest())
            exps_to_pair.append(exp_file)

    exps_to_pair = sorted(exps_to_pair)

    results_dict = {}
    prior_results_dict = {}
    if os.path.exists(save_file):
        with open(save_file,'r') as rf:
            results_dict = json.loads(rf.read())
    if os.path.exists("ext_prior_crossplay_results.json"):
        with open("ext_prior_crossplay_results.json",'r') as rf:
            prior_results_dict = json.loads(rf.read())
    # Nested for-loop to compare all weight files against each other (and themselves)
    for exp0 in exps_to_pair:
        for exp1 in exps_to_pair:
            weight_files = [path_to_model(exp0),path_to_model(exp1)]
            agents, intent_size = load_prebrief_model(weight_files, device, recv_or_send=recv_or_send)
    
            if args.total_game is not None:
                args.num_run = args.total_game // args.num_game

            exp_name = "$$".join([model_name(exp0),model_name(exp1)])
            if exp_name in results_dict.keys() and \
               model_hashes[model_name(exp0)] == str(results_dict[exp_name][-2]) and \
               model_hashes[model_name(exp1)] == str(results_dict[exp_name][-1]):
                continue
            elif exp_name in prior_results_dict.keys() and \
                 model_hashes[model_name(exp0)] == str(prior_results_dict[exp_name][-2]) and \
                 model_hashes[model_name(exp1)] == str(prior_results_dict[exp_name][-1]):
                results_dict[exp_name] = prior_results_dict[exp_name]
                continue
            try:
                # fast evaluation for 5k games
                print("||".join([model_name(exp0),model_name(exp1)]) + ": ")
                eval_results = evaluate_agents(
                    agents, num_game, seed, bomb, num_run=num_run, device=device, verbose=True, intent_size=intent_size
                )
                results_dict[exp_name] = (*eval_results, model_hashes[model_name(exp0)], model_hashes[model_name(exp1)])

                #plot image and ASCII number grid describing results
                with open(save_file,'w') as wf:
                    json.dump(results_dict, wf)

            except Exception as e:
                print(model_name(exp0) + " / " + model_name(exp1) + " failed: ")
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="sad", type=str, help="sad/op/obl")
    parser.add_argument("--num_game", default=2500, type=int) #5000 as default was causing "Resource temporarility unavailable"
    parser.add_argument("--total_game", default=None, type=int)
    parser.add_argument("--recv_or_send", default="recv", type=str)
    parser.add_argument(
        "--num_run", default=1, type=int, help="total num game = num_game * num_run"
    )
    # config for model from sad paper
    parser.add_argument("--weight", action="append", default=None, type=str)
    parser.add_argument("--expfolder", action="append", default=None, type=str)
    parser.add_argument("--save_file", default="paired_results.json", type=str)
    parser.add_argument("--num_player", default=None, type=int)
    # config for model from op paper
    parser.add_argument(
        "--method", default="sad-aux-op", type=str, help="sad-aux-op/sad-aux/sad-op/sad"
    )
    parser.add_argument("--idx1", default=1, type=int, help="which model to use?")
    parser.add_argument("--idx2", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

    if args.expfolder:
        exp_dict = {}
        for expfolder in args.expfolder:
            for root, dirs, files in os.walk(expfolder):
                for file in files:
                    if ".pthw" in file:
                        if root not in exp_dict.keys():
                            exp_dict[root] = []
                        exp_dict[root].append(file)

        evaluate_agent_pairs(
        exp_dict, args.num_game, 1337, 0, num_run=args.num_run, save_file=args.save_file, device=args.device, recv_or_send=args.recv_or_send,
        )
        quit()


    intent_size = 0
    if len(args.weight) == 1:
        args.weight = args.weight[0]
    if args.paper == "sad":
        assert os.path.exists(args.weight)
        # we are doing self player, all players use the same weight
        weight_files = [args.weight for _ in range(args.num_player)]
        agents = load_sad_model(weight_files, args.device)
    elif args.paper == "prebrief":
        assert (type(args.weight) is list and all([os.path.exists(wt) for wt in args.weight])) or \
               os.path.exists(args.weight)
        # we are doing self player, all players use the same weight
        weight_files = args.weight
        while len(weight_files) < args.num_player:
            weight_files.append(weight_files[-1])
        agents, intent_size = load_prebrief_model(weight_files, args.device, recv_or_send=args.recv_or_send)
        
    elif args.paper == "op":
        agents = load_op_model(args.method, args.idx1, args.idx2, args.device)
    elif args.paper == "obl":
        agents = [obl_model, obl_model]

    if args.total_game is not None:
        args.num_run = args.total_game // args.num_game

    # fast evaluation for 2500 games
    evaluate_agents(
        agents, args.num_game, 1337, 0, num_run=args.num_run, device=args.device, intent_size=intent_size
    )
