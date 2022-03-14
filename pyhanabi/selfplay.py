# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import time
import os
import sys
import argparse
import pprint
import gc

import numpy as np
import torch
from torch import nn

from create import create_envs, create_threads, ActGroup
from eval import evaluate
import common_utils
import rela
import r2d2
import utils


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
    parser.add_argument("--num_epoch", type=int, default=5000)
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
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    success = False
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)
       
    success = True
    gc.collect()

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_eps
    )
    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.hand_size,
        args.intent_size,
        args.train_bomb,
        explore_eps,
        args.max_len,
        args.sad,
        args.shuffle_obs,
        args.shuffle_color,
    )

    agent = r2d2.R2D2Agent(
        (args.method == "vdn"),
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        games[0].feature_size(),
        args.rnn_hid_dim,
        games[0].num_action(),
        args.num_lstm_layer,
        args.hand_size,
        args.intent_size,
        False,  # uniform priority
        num_ff_layer = args.num_ff_layer,
        skip_connect = args.skip_connect,
        num_player = args.num_player if args.use_player_id else None,
        intent_weight=args.intent_weight,
        use_xent_intent=args.use_xent_intent,
        dont_onehot_xent_intent=args.dont_onehot_xent_intent,
        use_pred_reward=args.use_pred_reward,
        one_way_intent=args.one_way_intent,
        shuf_pid=args.shuf_pid,
        intent_pred_input=args.intent_pred_input,
        intent_arch = args.intent_arch,
    )
    agent.sync_target_with_online()

    if args.load_model:
        print("*****loading pretrained model*****")
        utils.load_weight(agent.online_net, args.load_model, args.train_device)
        print("*****done*****")

    agent = agent.to(args.train_device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    act_group = ActGroup(
        args.method,
        args.act_device,
        agent,
        args.num_thread,
        args.num_game_per_thread,
        args.multi_step,
        args.gamma,
        args.eta,
        args.max_len,
        args.num_player,
        replay_buffer,
    )

    assert args.shuffle_obs == False, 'not working with 2nd order aux'
    context, threads = create_threads(
        args.num_thread, args.num_game_per_thread, act_group.actors, games,
    )
    act_group.start()
    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()

    eval_agent = agent.clone(args.train_device, {"vdn": False})
    eval_runners = [
        rela.BatchRunner(eval_agent, "cuda:0", 1000, ["act"])
        for _ in range(args.num_player)
    ]

    trend_lookback = 5
    scores = [1e38]*trend_lookback
    intent_losses = [1e38]*trend_lookback
    baseline_intent_loss = 1e38
    long_term_baseline_intent_loss = 1e38

    for epoch in range(args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                act_group.update_model(agent)

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data")

            loss, priority = agent.loss(batch, args.pred_weight, args.intent_weight, stat)
            priority = rela.aggregate_priority(
                priority.cpu(), batch.seq_len.cpu(), args.eta
            )
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            replay_buffer.update_priority(priority)
            stopwatch.time("updating priority")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        tachometer.lap(
            act_group.actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
        )
        stopwatch.summary()
        stat.summary(epoch)

        context.pause()
        eval_seed = (9917 + epoch * 999999) % 7777777
        for runner in eval_runners:
            runner.update_model(agent)
        score, perfect, *_ = evaluate(
            None,
            1000,
            eval_seed,
            args.eval_bomb,
            0,  # explore eps
            args.sad,
            runners=eval_runners,
            intent_size=args.intent_size
        )
        if epoch > 0 and epoch % 50 == 0:
            force_save_name = "model_epoch%d" % epoch
        else:
            force_save_name = None
        model_saved = saver.save(
            None, agent.online_net.state_dict(), score, force_save_name=force_save_name
        )
        print(
            "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
            % (epoch, score, perfect * 100, model_saved)
        )


        # break training loop to adjust intent weight if needed
        if args.intent_weight > 0.:
            intent_loss = stat['intent1'].mean()
            scores.pop(0)
            scores.append(score)
            intent_losses.pop(0)
            intent_losses.append(intent_loss)
            if epoch == 1: # set baseline intent loss
                baseline_intent_loss = intent_loss
            if epoch <= 10 and intent_loss < long_term_baseline_intent_loss:
                long_term_baseline_intent_loss = intent_loss
            if epoch > 10:
                if (epoch > 100 and np.all(np.array(scores) < 2.0) and np.all(np.array(intent_losses) > long_term_baseline_intent_loss*0.99)) or \
                   ('after' in args.intent_arch and epoch > 50 and np.all(np.array(scores) < 2.0)):
                    print("last five scores are all below 1 - restart")
                    success = False
                    context.terminate()
                    while not context.terminated():
                        time.sleep(0.5)
                    for runner in eval_runners:
                        runner.stop()
                    del agent
                    del replay_buffer
                    del act_group
                    del context
                    del eval_agent
                    del eval_runners
                    del games
                    args.intent_weight /= 1.1
                    print("RELOAD: dec intent_weight to %6.5f"%args.intent_weight)
                    break
                if (np.all(np.array(scores) > 8.0) and np.all(np.array(intent_losses) > long_term_baseline_intent_loss*0.99)) or \
                   (epoch > 50 and np.all(np.array(scores) > 10.0) and np.all(np.array(intent_losses) > long_term_baseline_intent_loss*0.98)):
                    print("intent loss is not decreasing below baseline - restart")
                    success = False
                    context.terminate()
                    while not context.terminated():
                        time.sleep(0.5)
                    for runner in eval_runners:
                        runner.stop()
                    del agent
                    del replay_buffer
                    del act_group
                    del context
                    del eval_agent
                    del eval_runners
                    del games
                    args.intent_weight *= 1.1
                    print("RELOAD: inc intent_weight to %6.5f"%args.intent_weight)
                    break


        gc.collect()
        context.resume()
        print("==========")
