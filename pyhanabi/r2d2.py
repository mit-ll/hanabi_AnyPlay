# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from numpy import nanpercentile
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
import common_utils


class R2D2Net(torch.jit.ScriptModule):
    __constants__ = [
        "hid_dim",
        "out_dim",
        "num_lstm_layer",
        "hand_size",
        "intent_size",
        "skip_connect",
    ]

    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        intent_size,
        num_player,
        num_ff_layer,
        skip_connect,
        player_embed_dim = 8,
        train_intent_net = False,
        use_xent_intent = False,
        intent_pred_input = 'lstm_o',
        intent_arch = 'concat',
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = num_ff_layer
        self.num_lstm_layer = num_lstm_layer
        self.hand_size = hand_size
        self.intent_size = intent_size
        self.skip_connect = skip_connect
        self.num_player = num_player
        self.train_intent_net = train_intent_net
        self.use_xent_intent = use_xent_intent
        self.intent_pred_input = intent_pred_input
        self.intent_arch = intent_arch
        

        self.player_embed = None
        self.player_embed_dim = 0
        self.in_dim = in_dim

        if num_player is not None:
            self.player_embed = nn.Embedding(num_player, player_embed_dim)
            self.player_embed_dim = player_embed_dim
        
        self.intent_embed = None
        if self.intent_arch != 'concat':
            intent_embed_layers = [nn.Linear(self.intent_size + self.player_embed_dim, self.hid_dim), nn.ReLU()]
            self.intent_embed = nn.Sequential(*intent_embed_layers)

        # print("intent",self.intent_size)
        # print("player_embed_dim",self.player_embed_dim)
        # print("in_dim",self.in_dim)
        self.intent_net = None
        if self.train_intent_net and self.intent_size > 0:
            # intent_layers = [nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(), \
            intent_layers = [nn.Linear(self.in_dim + self.player_embed_dim - self.intent_size, self.hid_dim), nn.ReLU(), \
                            # nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(),
                             nn.Linear(self.hid_dim, self.intent_size)]
            if use_xent_intent:
                intent_layers.append(nn.Sigmoid())
            self.intent_net = nn.Sequential(*intent_layers)

        ff_layers = [nn.Linear(self.in_dim + self.player_embed_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred = nn.Linear(self.hid_dim, self.hand_size * 3)

        self.pred_intent = None
        if self.intent_size > 0:
            if self.intent_pred_input == 'state_pid':
                self.pred_intent = nn.Sequential(nn.Linear(self.in_dim + self.player_embed_dim, self.hid_dim), nn.ReLU(), nn.Linear(self.hid_dim, self.intent_size))
            else:                
                self.pred_intent = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(), nn.Linear(self.hid_dim, self.intent_size))

    def freeze_all_but_intent_net(self):
        #freeze all parameters besides intent_net
        for p in self.parameters():
            p.requires_grad = False
        for p in self.intent_net.parameters():
            p.requires_grad = True

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def adapt_intent(
        self, priv_s: torch.Tensor, player_ids:torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            player_ids = player_ids.unsqueeze(0)
            # c0 = hid["c0"][-1,...]
            # c0 = c0.unsqueeze(0)
            one_step = True
        else:
            # c0 = hid["c0"][:,-1,...]
            pass

        if self.num_player is not None:
            plyr_embed = self.player_embed(player_ids)
            priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
        # adapted_intent = self.intent_net(c0)
        
        adapted_intent = player_ids #this should never be the value, doing this to get JIT script to compile
        if self.intent_net is not None:
            # adapted_intent = self.intent_net(priv_s)
            adapted_intent = self.intent_net(torch.zeros_like(priv_s)) #doing this to see if we can just find optimal intent for player 0

        if one_step:
            adapted_intent = adapted_intent.squeeze(0)
        return adapted_intent

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, player_ids:torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % priv_s.dim()

        priv_s = priv_s.unsqueeze(0)
        player_ids = player_ids.unsqueeze(0)
        if self.num_player is not None:
            plyr_embed = self.player_embed(player_ids)
            priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
        int_embed = None
        if self.intent_embed is not None:
            int_embed = self.intent_embed(priv_s[...,-(self.intent_size + self.player_embed_dim):])        
        
        if 'only' in self.intent_arch:
            priv_s[...,-(self.intent_size + self.player_embed_dim):] = 0.

        #trim SAD observations down if model is actually IQL or VDN
        if self.intent_size == 0 and self.player_embed_dim == 0:
            priv_s = priv_s[...,:self.in_dim]
        
        x = self.net(priv_s)
        if int_embed is not None and 'before' in self.intent_arch:
            x = x + int_embed
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        if self.skip_connect:
            o = o + x
        if int_embed is not None and 'after' in self.intent_arch:
            o = o + int_embed
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        player_ids:torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            player_ids = player_ids.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True
        if self.num_player is not None:
            plyr_embed = self.player_embed(player_ids)
            priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
        
        int_embed = None
        if self.intent_embed is not None:
            int_embed = self.intent_embed(priv_s[...,-(self.intent_size + self.player_embed_dim):])

        if 'only' in self.intent_arch:
            priv_s[...,-(self.intent_size + self.player_embed_dim):] = 0.
            
        # if torch.any(priv_s > 1000):
        #     print("priv_s has large value")
        x = self.net(priv_s)
        # if torch.any(torch.isnan(x)):
        #     print("x is nan")
        #     if torch.any(priv_s > 1000):
        #         print("priv_s has large value")
        #     if torch.any(torch.isnan(priv_s)):
        #         print("priv_s is nan")
        #     # print("priv_s",priv_s)
        #     # print("x",x)

        if int_embed is not None and 'before' in self.intent_arch:
            x = x + int_embed

        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        if self.skip_connect:
            o = o + x
        if int_embed is not None and 'after' in self.intent_arch:
            o = o + int_embed
        # if torch.any(torch.isnan(o)):
        #     print ("o is nan")
        #     print("x",x)
        #     print("hid",hid)
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self._duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o, c

    @torch.jit.script_method
    def _duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    def cross_entropy(self, net, lstm_o, target_p, hand_slot_mask, seq_len):
        # target_p: [seq_len, batch, num_player, 5, 3]
        # hand_slot_mask: [seq_len, batch, num_player, 5]
        logit = net(lstm_o).view(target_p.size())
        q = nn.functional.softmax(logit, -1)
        logq = nn.functional.log_softmax(logit, -1)
        plogq = (target_p * logq).sum(-1)
        xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(
            min=1e-6
        )

        # print("xentsize", xent.size())
        if xent.dim() == 3:
            # [seq, batch, num_player]
            xent = xent.mean(2)

        # save before sum out
        seq_xent = xent
        xent = xent.sum(0)
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        return xent, avg_xent, q, seq_xent.detach()
    
    def intent_squared_loss(self, lstm_o: torch.Tensor, player_intents: torch.Tensor, own_intent_mask: torch.Tensor, seq_len: torch.Tensor):
        # target_p: [seq_len, batch, num_player, intent_size]
        # own_intent_mask: [(num_player-1) * num_player]
        
        # reshape/reorder target_p to [seq_len, batch, num_player * (num_player-1), intent_size]
        target_p = player_intents[:,:,own_intent_mask,:]
        target_p_size = target_p.size()
        max_seq_len = target_p_size[0]
        # print("target_p_size", target_p.size())
        logit = self.pred_intent(lstm_o).view(target_p.size())
        # print((target_p[0,0,0,0]))
         
        # shape should be [seq_len, batch, (num_player-1) * num_player]
        # (sqrt is over each other player, not all players)
        # squared_loss = nn.functional.mse_loss(logit, target_p)
        squared_loss = torch.sqrt(torch.square(logit - target_p).sum(-1) + 1e-10)
        # squared_loss = torch.square(logit - target_p).sum(-1)

        # mean over all players 
        # shape should be [seq_len, batch]
        # squared_loss = squared_loss.mean(-1)
        # print("SQUAREDLOSS:",squared_loss)
        # print("SQUAREDLOSS:",squared_loss.size())
        # print("squaredlosssize",squared_loss.size())
        if squared_loss.dim() == 3:
            # [seq, batch, num_player]
            squared_loss = squared_loss.mean(2)
            # print("ifstatement",squared_loss.size())  

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        # mask = (mask.unsqueeze(1) == seq_len.unsqueeze(0)-1).float() #This makes the loss and reward for intent only pertinent at the end
        squared_loss *= mask

        # save before sum out
        seq_squared_loss = squared_loss
        squared_loss = squared_loss.sum(0)
        assert squared_loss.size() == seq_len.size()
        avg_squared_loss = (squared_loss / seq_len).mean().item()
        # avg_squared_loss = (squared_loss).mean().item()
        return squared_loss, avg_squared_loss, logit, seq_squared_loss#.detach()
    
    def intent_crossentropy_loss(self, lstm_o: torch.Tensor, player_intents: torch.Tensor, own_intent_mask: torch.Tensor, seq_len: torch.Tensor):
        # target_p: [seq_len, batch, num_player, intent_size]
        # own_intent_mask: [(num_player-1) * num_player]
        
        #softmax logit output
        #get - log likelihood of correct intent

        # reshape/reorder target_p to [seq_len, batch, num_player * (num_player-1), intent_size]
        target_p = player_intents[:,:,own_intent_mask,:]
        target_p_size = target_p.size()
        max_seq_len = target_p_size[0] 
        target_onehot = nn.functional.one_hot(target_p.argmax(-1), num_classes=target_p_size[-1])
        # target_onehot = torch.zeros(target_p_size).to(device)
        # target_onehot[target_p.argmax(-1, keepdim=True)] = 1
        
        # print("target_p_size", target_p.size())
        logit = self.pred_intent(lstm_o).view(target_p_size)
        # print((target_p[0,0,0,0]))
        q = nn.functional.softmax(logit, -1)
        logq = nn.functional.log_softmax(logit, -1)
        plogq = (target_onehot * logq).sum(-1)
        xent = -(plogq).sum(-1)

        # shape should be [seq_len, batch, (num_player-1) * num_player]
        # mean over all players 
        if xent.dim() == 3:
            # [seq, batch, num_player]
            xent = xent.mean(2)
            # print("ifstatement",squared_loss.size())  

        # print("true: ", target_onehot[:6,0,0], "predicted: ", q[:6,0,0])

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        # mask = (mask.unsqueeze(1) == seq_len.unsqueeze(0)-1).float() #This makes the loss and reward for intent only pertinent at the end
        xent *= mask

        # save before sum out
        seq_xent = xent
        xent = xent.sum(0)
        # xent = xent.sum(0) / seq_len #attempting to normalize for episode length
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        # avg_xent = (xent).mean().item()
        return xent, avg_xent, q, seq_xent.detach()

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return self.cross_entropy(self.pred, lstm_o, target, hand_slot_mask, seq_len)

    def pred_loss_intent(self, lstm_o: torch.Tensor, target: torch.Tensor, own_intent_mask: torch.Tensor, seq_len: torch.Tensor):
        if self.use_xent_intent:
            return self.intent_crossentropy_loss(lstm_o, target, own_intent_mask, seq_len)
        else:
            return self.intent_squared_loss(lstm_o, target, own_intent_mask, seq_len)

class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["vdn", "multi_step", "gamma", "eta", "uniform_priority"]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        intent_size,
        uniform_priority,
        *,
        num_ff_layer=1,
        skip_connect=False,
        num_player=None,
        player_embed_dim = 8,
        use_xent_intent = False,
        intent_weight = 0.,
        dont_onehot_xent_intent = False,
        train_adapt = False,
        use_pred_reward = False,
        one_way_intent = False,
        recv_or_send = 'both',
        shuf_pid = False,
        intent_pred_input = 'lstm_o',
        intent_arch = 'concat',
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.online_net = R2D2Net(
            device,
            in_dim + intent_size,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            intent_size if num_player is None else intent_size*(num_player-1),
            num_player,
            num_ff_layer,
            skip_connect,
            player_embed_dim=player_embed_dim,
            train_intent_net=train_adapt,
            use_xent_intent=use_xent_intent,
            intent_pred_input=intent_pred_input,
            intent_arch=intent_arch,
        ).to(device)
        self.target_net = R2D2Net(
            device,
            in_dim + intent_size,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            intent_size if num_player is None else intent_size*(num_player-1),
            num_player,
            num_ff_layer,
            skip_connect,
            player_embed_dim=player_embed_dim,
            train_intent_net=train_adapt,
            use_xent_intent=use_xent_intent,
            intent_pred_input=intent_pred_input,
            intent_arch=intent_arch,
        ).to(device)
        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.uniform_priority = uniform_priority
        self.intent_size = intent_size
        self.use_xent_intent = use_xent_intent
        self.intent_weight = intent_weight
        self.train_adapt = train_adapt
        self.use_pred_reward = use_pred_reward
        self.dont_onehot_xent_intent = dont_onehot_xent_intent
        self.one_way_intent = one_way_intent
        self.recv_or_send = recv_or_send
        self.shuf_pid = shuf_pid
        self.player_ids_str = "player_ids" if not shuf_pid else "player_shuffled_ids"
        self.intent_pred_input = intent_pred_input
        self.intent_arch = intent_arch


    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.online_net.num_lstm_layer,
            self.online_net.hand_size,
            self.online_net.intent_size,
            self.uniform_priority,
            num_ff_layer=self.online_net.num_ff_layer,
            skip_connect=self.online_net.skip_connect,
            num_player=self.online_net.num_player,
            player_embed_dim=self.online_net.player_embed_dim,
            train_adapt=self.train_adapt,
            use_pred_reward=self.use_pred_reward,
            use_xent_intent=self.use_xent_intent,
            intent_weight=self.intent_weight,
            dont_onehot_xent_intent=self.dont_onehot_xent_intent,
            one_way_intent=self.one_way_intent,
            recv_or_send=self.recv_or_send,
            shuf_pid=self.shuf_pid,
            intent_pred_input=self.intent_pred_input,
            intent_arch=self.intent_arch,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        player_ids: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, player_ids, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        obsize, ibsize, num_player = 0, 0, 0
        if self.vdn:
            obsize, ibsize, num_player = obs["priv_s"].size()[:3]
            priv_s = obs["priv_s"].flatten(0, 2)
            legal_move = obs["legal_move"].flatten(0, 2)
            eps = obs["eps"].flatten(0, 2)
            assert self.player_ids_str in obs.keys()
            player_ids= obs[self.player_ids_str].flatten(0, 2)
            if "player_intents" in obs.keys() and self.intent_size > 0:
                player_intents = obs["player_intents"].flatten(0, 2)
                player_intents = player_intents[..., :self.intent_size]
            else:
                player_intents = None
        else:
            obsize, ibsize = obs["priv_s"].size()[:2]
            num_player = 1
            priv_s = obs["priv_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
            eps = obs["eps"].flatten(0, 1)
            assert self.player_ids_str in obs.keys()
            player_ids= obs[self.player_ids_str].flatten(0, 1)
            
            if "player_intents" in obs.keys() and self.intent_size > 0:
                player_intents= obs["player_intents"].flatten(0, 1)
                player_intents = player_intents[..., :self.intent_size]
            else:
                player_intents = None

        hid = {
            "h0": obs["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": obs["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        # if player_ids is not None:
        #     plyr_embed = self.player_embed(player_ids)
        #     priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
        if player_intents is not None:
            if self.train_adapt:
                adapted_intents = self.online_net.adapt_intent(priv_s, player_ids, hid)
                # print(self.player_ids_str,player_ids[:10])
                # print("priv_s",priv_s.size())
                # print("player_intents",player_intents.size())
                # print("adapted_intents",adapted_intents.size())
                # print("player_ids_expand",player_ids.unsqueeze(-1).expand(-1,self.intent_size)[:10])
                player_intents = torch.zeros_like(player_intents) #CHEATING
                player_intents[...,0] = 0.1 #CHEATING
                player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, adapted_intents, player_intents)#.detach()
            if self.recv_or_send == 'recv':
                player_ids[...] = 0
            if self.recv_or_send == 'send':
                player_ids[...] = 1
            if self.one_way_intent:
                # assert self.online_net.num_player is not None # make sure PID is being used / commented out to allow it not to be used
                # sender_intents = player_intents[...,1,:].expand()
                receiver_intents = torch.zeros_like(player_intents)
                if self.use_xent_intent:
                    receiver_intents[...,0] = 1.
                player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, receiver_intents, player_intents)
            if self.use_xent_intent and not self.dont_onehot_xent_intent:
                player_intents = nn.functional.one_hot(player_intents.argmax(-1), num_classes=player_intents.size()[-1])
            priv_s = torch.cat((priv_s, player_intents), dim=-1)

        greedy_action, new_hid = self.greedy_act(priv_s, player_ids, legal_move, hid)

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        # if torch.any(torch.isnan(rand)):
        #     print ("rand is nan",rand)
        assert rand.size() == eps.size()
        # rand = (rand < eps).long()
        # action = (greedy_action * (1 - rand) + random_action * rand).detach().long()
        action = torch.where(rand < eps, random_action, greedy_action).detach()
        # if torch.any(torch.isnan(action)):
        #     print ("action is nan",action)
        # if torch.any(torch.isnan(greedy_action)):
        #     print ("greedy_action is nan",greedy_action)

        if self.vdn:
            action = action.view(obsize, ibsize, num_player)
            greedy_action = greedy_action.view(obsize, ibsize, num_player)
            rand = rand.view(obsize, ibsize, num_player)
        else:
            action = action.view(obsize, ibsize)
            greedy_action = greedy_action.view(obsize, ibsize)
            rand = rand.view(obsize, ibsize)

        hid_shape = (
            obsize,
            ibsize * num_player,
            self.online_net.num_lstm_layer,
            self.online_net.hid_dim,
        )
        h0 = new_hid["h0"].transpose(0, 1).view(*hid_shape)
        c0 = new_hid["c0"].transpose(0, 1).view(*hid_shape)

        reply = {
            "a": action.detach().cpu(),
            "greedy_a": greedy_action.detach().cpu(),
            "h0": h0.contiguous().detach().cpu(),
            "c0": c0.contiguous().detach().cpu(),
        }
        return reply

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method is for inference on the Hanabi_SPARTA repo. It wraps the act method
        while adding inference-time observations into the obs dict

        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        assert len(obs["s"].size()) == 2
        assert obs["s"].size()[-1] == 838, "dim input not as expected"
        if "s" in obs.keys() and "priv_s" not in obs.keys():
            priv_s = obs["s"]
        else:
            priv_s = obs["priv_s"]
        player_ids = torch.zeros(priv_s.size()[:-1], dtype=torch.int32).to(priv_s.device)
        player_intents = torch.zeros(list(player_ids.size()) + [self.intent_size], dtype=torch.float32).to(priv_s.device)
        
        hid = {
            "h0": obs["h0"].transpose(0, 1).contiguous(),
            "c0": obs["c0"].transpose(0, 1).contiguous(),
        }

        if self.train_adapt:
            adapted_intents = self.online_net.adapt_intent(priv_s, player_ids, hid)
            player_intents = torch.zeros_like(player_intents).to(priv_s.device) #CHEATING
            player_intents[...,0] = 0.1 #CHEATING
            player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, adapted_intents, player_intents)#.detach()
        if self.recv_or_send == 'recv':
            player_ids[...] = 0
        if self.recv_or_send == 'send':
            player_ids[...] = 1
        if self.one_way_intent:
            # assert self.online_net.num_player is not None # make sure PID is being used / Commented out to allow to not use player-id
            # sender_intents = player_intents[...,1,:].expand()
            receiver_intents = torch.zeros_like(player_intents).to(priv_s.device)
            if self.use_xent_intent:
                receiver_intents[...,0] = 1.
            player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, receiver_intents, player_intents)
        if self.use_xent_intent and not self.dont_onehot_xent_intent:
            player_intents = nn.functional.one_hot(player_intents.argmax(-1), num_classes=player_intents.size()[-1])
        priv_s = torch.cat((priv_s, player_intents), dim=-1)

        action, new_hid = self.online_net.act(priv_s, player_ids, hid)

        h0 = new_hid["h0"]#.view(2,1,512) # hardcoding assuming Hanabi_SAD
        c0 = new_hid["c0"]#.view(2,1,512) # hardcoding assuming Hanabi_SAD

        reply = {
            "a": action.detach(),
            "h0": h0.transpose(0,1).contiguous().detach(),
            "c0": c0.transpose(0,1).contiguous().detach(),
        }
        return reply
        
        

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        compute priority for one batch
        """
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"]).detach().cpu()}

        obsize, ibsize, num_player = 0, 0, 0
        flatten_end = 0
        if self.vdn:
            obsize, ibsize, num_player = input_["priv_s"].size()[:3]
            flatten_end = 2
        else:
            obsize, ibsize = input_["priv_s"].size()[:2]
            num_player = 1
            flatten_end = 1

        priv_s = input_["priv_s"].flatten(0, flatten_end)
        legal_move = input_["legal_move"].flatten(0, flatten_end)
        online_a = input_["a"].flatten(0, flatten_end)
        
        assert self.player_ids_str in input_.keys()
        player_ids= input_[self.player_ids_str].flatten(0, flatten_end)
        player_intents = torch.zeros((1)) #placeholder to make jitscript not see this as optional
        if "player_intents" in input_.keys() and self.intent_size > 0:
            player_intents = input_["player_intents"].flatten(0, flatten_end)
            player_intents = player_intents[..., :self.intent_size]
 

        hid = {
            "h0": input_["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        next_hid = {
            "h0": input_["next_h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["next_c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        # if player_ids is not None:
        #     plyr_embed = self.player_embed(player_ids)
        #     priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
        if player_intents is not None and self.intent_size > 0:
            if self.train_adapt:
                adapted_intents = self.online_net.adapt_intent(priv_s, player_ids, hid)
                player_intents = torch.zeros_like(player_intents) #CHEATING
                player_intents[...,0] = 0.1 #CHEATING
                player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, adapted_intents, player_intents)#.detach()
            if self.recv_or_send == 'recv':
                player_ids[...] = 0
            if self.recv_or_send == 'send':
                player_ids[...] = 1
            if self.one_way_intent:
                # assert self.online_net.num_player is not None # make sure PID is being used / commented out to allow it not to be used
                receiver_intents = torch.zeros_like(player_intents)
                if self.use_xent_intent:
                    receiver_intents[...,0] = 1.
                player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, receiver_intents, player_intents)
            if self.use_xent_intent and not self.dont_onehot_xent_intent:
                player_intents = nn.functional.one_hot(player_intents.argmax(-1), num_classes=player_intents.size()[-1])
            priv_s = torch.cat((priv_s, player_intents), dim=-1)

        next_priv_s = input_["next_priv_s"].flatten(0, flatten_end)
        next_legal_move = input_["next_legal_move"].flatten(0, flatten_end)
        temperature = input_["temperature"].flatten(0, flatten_end)

        assert self.player_ids_str in input_.keys()
        next_player_ids= input_["next_"+self.player_ids_str].flatten(0, flatten_end)
        if "next_player_intents" in input_.keys() and self.intent_size > 0:
            next_player_intents= input_["next_player_intents"].flatten(0, flatten_end)
            next_player_intents = next_player_intents[..., :self.intent_size]
        else:
            next_player_intents = None

        # if next_player_ids is not None:
        #     next_plyr_embed = self.player_embed(next_player_ids)
        #     next_priv_s = torch.cat((next_priv_s, next_plyr_embed), dim=-1)
        if next_player_intents is not None:
            if self.train_adapt:
                next_adapted_intents = self.online_net.adapt_intent(next_priv_s, next_player_ids, next_hid)
                next_player_intents = torch.zeros_like(next_player_intents) #CHEATING
                next_player_intents[...,0] = 0.1 #CHEATING
                next_player_intents = torch.where(next_player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, next_adapted_intents, next_player_intents)#.detach()
            if self.recv_or_send == 'recv':
                next_player_ids[...] = 0
            if self.recv_or_send == 'send':
                next_player_ids[...] = 1
            if self.one_way_intent:
                # assert self.online_net.num_player is not None # make sure PID is being used / commented out to allow it not to be used
                next_receiver_intents = torch.zeros_like(next_player_intents)
                if self.use_xent_intent:
                    next_receiver_intents[...,0] = 1.
                next_player_intents = torch.where(next_player_ids.unsqueeze(-1).expand(-1,self.intent_size) == 0, next_receiver_intents, next_player_intents)
            if self.use_xent_intent and not self.dont_onehot_xent_intent:
                next_player_intents = nn.functional.one_hot(next_player_intents.argmax(-1), num_classes=next_player_intents.size()[-1])
            next_priv_s = torch.cat((next_priv_s, next_player_intents), dim=-1)

        reward = input_["reward"].flatten(0, 1)
        bootstrap = input_["bootstrap"].flatten(0, 1)

        online_qa, greedy_a, _, lstm_o, lstm_c = self.online_net(priv_s, player_ids, legal_move, online_a, hid)
        next_a, _ = self.greedy_act(next_priv_s, next_player_ids, next_legal_move, next_hid)
        target_qa, _, _, _, _ = self.target_net(
            next_priv_s, next_player_ids, next_legal_move, next_a, next_hid,
        )

        bsize = obsize * ibsize
        if self.vdn:
            # sum over action & player
            online_qa = online_qa.view(bsize, num_player).sum(1)
            target_qa = target_qa.view(bsize, num_player).sum(1)

        # intent_loss = 0.
        # seq_intent_loss = 0.
        # if self.intent_weight > 0 and self.intent_size > 0:
        #     # To add incentive to make behavior identifiable, 
        #     # add reward for low loss of other agent
        #     print("KEYS", input_.keys())
        #     intent_loss, seq_intent_loss, avg_intent_loss = self.aux_task_intent_jit(
        #         lstm_o,
        #         player_intents.view(obsize, ibsize, num_player, self.intent_size),
        #         # input_["seq_len"],
        #         1,
        #         reward.size()[1:],
        #     )
        #     # reward -= seq_intent_loss.detach() * self.intent_weight
        #     reward -= seq_intent_loss * self.intent_weight

        assert reward.size() == bootstrap.size()
        assert reward.size() == target_qa.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        priority = (target - online_qa).abs()
        priority = priority.view(obsize, ibsize).detach().cpu()
        return {"priority": priority}

    ############# python only functions #############
    def flat_4d(self, data):
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
        bsize = 0
        num_player = 0
        for k, v in data.items():
            if num_player == 0:
                bsize, num_player = v.size()[1:3]

            if v.dim() == 4:
                d0, d1, d2, d3 = v.size()
                data[k] = v.view(d0, d1 * d2, d3)
            elif v.dim() == 3:
                d0, d1, d2 = v.size()
                data[k] = v.view(d0, d1 * d2)
        return bsize, num_player

    def td_error(self, obs, hid, action, reward, terminal, bootstrap, seq_len, stat, intent_weight=0., pred_weight=0.):
        max_seq_len = obs["priv_s"].size(0)

        bsize, num_player = 0, 1
        if self.vdn:
            bsize, num_player = self.flat_4d(obs)
            self.flat_4d(action)

        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        action = action["a"]

        assert self.player_ids_str in obs.keys()
        player_ids = obs[self.player_ids_str]
        # print("PLAYER_IDS",player_ids[0,...])
        # plyr_embed = self.player_embed(obs[self.player_ids_str])
        # priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
        if "player_intents" in obs.keys() and self.intent_size > 0:
            player_intents = obs["player_intents"]
            player_intents = player_intents[..., :self.intent_size]
            if self.train_adapt:
                # h0 and c0 do not seem available here
                adapt_hid = {
                #     "h0": obs["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
                #     "c0": obs["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
                }
                # print(self.player_ids_str,player_ids.size())
                # print("priv_s",priv_s.size())
                # print("player_intents",player_intents.size())
                # print("player_ids_expand",player_ids.unsqueeze(-1).expand(-1,-1,self.intent_size)[:10])
                adapted_intents = self.online_net.adapt_intent(priv_s, player_ids, adapt_hid)
                # print("adapted_intents",adapted_intents.size())
                player_intents = torch.zeros_like(player_intents) #CHEATING
                player_intents[...,0] = 0.1 #CHEATING
                player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,-1,self.intent_size) == 0, adapted_intents, player_intents)#.detach()
            if self.recv_or_send == 'recv':
                player_ids[...] = 0
            if self.recv_or_send == 'send':
                player_ids[...] = 1
            if self.one_way_intent:
                # assert self.online_net.num_player is not None # make sure PID is being used / commented out to allow it not to be used
                receiver_intents = torch.zeros_like(player_intents)
                if self.use_xent_intent:
                    receiver_intents[...,0] = 1.
                player_intents = torch.where(player_ids.unsqueeze(-1).expand(-1,-1,self.intent_size) == 0, receiver_intents, player_intents)
                # print("PLAYER_IDS",player_ids.unsqueeze(-1).expand(-1,-1,self.intent_size))
                # print("PLAYER_INTENTS before", player_intents)
            if self.use_xent_intent and not self.dont_onehot_xent_intent:
                player_intents = nn.functional.one_hot(player_intents.argmax(-1), num_classes=player_intents.size()[-1])
                # print("PLAYER_INTENTS after", player_intents)
            priv_s = torch.cat((priv_s, player_intents), dim=-1)
        else:
            player_intents = None

        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, _, lstm_o, lstm_c = self.online_net(
            priv_s, player_ids, legal_move, action, hid
        )

        # if torch.any(torch.isnan(lstm_o)):
        #     print ("lstm_o is nan")
        #     print([('obs',obs),('action',action),("legal_move",legal_move),("hid",hid)])
        #     for dictname, dictthing in [('obs',obs),('action',action),("legal_move",legal_move),("hid",hid)]:
        #         if type(dictthing) is dict:
        #             for key in dictthing.keys():
        #                 if torch.any(torch.isnan(dictthing[key])):
        #                     print(dictname, key, dictthing[key])
        #         else:
        #             if torch.any(torch.isnan(dictthing)):
        #                 print(dictname, dictthing)

        with torch.no_grad():
            target_qa, _, _, _, _ = self.target_net(priv_s, player_ids, legal_move, greedy_a, hid)
            # assert target_q.size() == pa.size()
            # target_qe = (pa * target_q).sum(-1).detach()
            assert online_qa.size() == target_qa.size()

        if self.vdn:
            online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
            target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
            lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)
            lstm_c = lstm_c.view(self.online_net.num_lstm_layer, bsize, num_player, -1)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        target_qa = torch.cat(
            [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
        )
        target_qa[-self.multi_step :] = 0

        if pred_weight > 0 and self.use_pred_reward:
            if self.vdn:
                pred_loss, seq_pred_loss = self.aux_task_vdn(
                    lstm_o,
                    obs["own_hand"].view(max_seq_len, bsize, num_player, -1),
                    # batch.obs["temperature"], #not sure what this temperature variable it is not used
                    None,
                    seq_len,
                    reward.size()[1:],
                    stat,
                )
            else:
                pred_loss, seq_pred_loss = self.aux_task_iql(
                    lstm_o, obs["own_hand"].view(max_seq_len, bsize, num_player, -1), 
                    seq_len, 
                    reward.size()[1:], 
                    stat,
                )
            reward -= pred_weight * seq_pred_loss

        intent_loss = 0.
        seq_intent_loss = 0.
        if intent_weight > 0 and self.intent_size > 0:
            # To add incentive to make behavior identifiable, 
            # add reward for low loss of other agent
            if self.intent_pred_input == 'lstm_o':
                pred_input = lstm_o
            elif self.intent_pred_input == 'lstm_c':
                # lstm_c is of shape [num_lstm_layer, bsize, num_player, hid_dim]
                # transforming it to take only last_lstm layer and expand hid_dim to be seq_len
                # final shape will be [max_seq_len, bsize, num_player, hid_dim]
                pred_input = lstm_c[-1,...].unsqueeze(0).expand(max_seq_len, -1, -1, -1)
            elif self.intent_pred_input == 'state_pid' and self.player_ids_str in obs.keys():
                plyr_embed = self.online_net.player_embed(player_ids)
                priv_s = torch.cat((priv_s, plyr_embed), dim=-1)
                pred_input = priv_s


            intent_loss, seq_intent_loss = self.aux_task_intent(
                pred_input,
                player_intents.view(max_seq_len, bsize, num_player, self.intent_size),
                seq_len,
                reward.size()[1:],
                stat,
            )
            reward -= seq_intent_loss.detach() * intent_weight

        assert target_qa.size() == reward.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        
        # if intent_weight > 0 and self.intent_size > 0:
        #     err += seq_intent_loss * intent_weight

        if self.train_adapt:
            err = -online_qa # While this is not always greedy action, 
                             # I am hoping this works similar to actor-critic training
                             # where the intent_net is actor and frozen q-net is critic
            # err += 25. # Wanted to get err positive
            err /= 25. # Wanted to get err to similar magnitude as regular RL loss

        return err, lstm_o, (intent_loss, seq_intent_loss)

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, seq_pred_loss1 = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux1"].feed(avg_xent1)
        return pred_loss1, seq_pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, seq_pred_loss1 = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        rotate = [num_player - 1]
        rotate.extend(list(range(num_player - 1)))
        partner_hand = own_hand[:, :, rotate, :, :]
        partner_hand_slot_mask = partner_hand.sum(4)
        partner_belief1 = belief1[:, :, rotate, :, :].detach()
        if stat is not None:
            stat["aux1"].feed(avg_xent1)
        return pred_loss1, seq_pred_loss1
    
    def aux_task_intent_jit(self, lstm_o, player_intents, seq_len, rl_loss_size:List[int]
    ):# -> Tuple[torch.Tensor,torch.Tensor,float]:
        """auxilliary task to predict other agents intent vector"""
        seq_size, bsize, num_player, _ = player_intents.size()
        all_player_intents = player_intents.view(seq_size, bsize, num_player, \
                                                 -1)
        all_player_intents = all_player_intents[..., :self.online_net.intent_size]
        # own_intent_mask = 1 - torch.diag(torch.ones((num_player)))
        own_intent_mask = [i for j in range(num_player) for i in range(num_player) if i != j]
        # own_intent_mask = torch.zeros((num_player*(num_player-1)))
        # k = 0
        # for j in range(num_player):
        #     for i in range(num_player):
        #         if i != j:
        #             own_intent_mask[k] = i
        #             k += 1

        squared_loss, avg_squared_loss, _, seq_squared_loss = self.online_net.pred_loss_intent(
            lstm_o, all_player_intents, own_intent_mask, seq_len
        )
        assert squared_loss.size() == rl_loss_size
        
        return squared_loss, seq_squared_loss, avg_squared_loss

    def aux_task_intent(self, lstm_o, player_intents, seq_len, rl_loss_size:List[int], stat):
        squared_loss, seq_squared_loss, avg_squared_loss = self.aux_task_intent_jit(lstm_o, player_intents, seq_len, rl_loss_size)
        if stat is not None:
            stat["intent1"].feed(avg_squared_loss)
        return squared_loss, seq_squared_loss


    def loss(self, batch, pred_weight, intent_weight, stat):
        err, lstm_o, (intent_loss, seq_intent_loss) = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
            stat,
            # intent_weight=self.intent_weight, # keep this intent constant
            intent_weight=intent_weight,
            pred_weight=pred_weight
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["adapt_loss" if self.train_adapt else "rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        # priority = self.aggregate_priority(p, batch.seq_len)
        # print("IDs")
        # print(batch.obs[self.player_ids_str].shape)
        # print(batch.obs[self.player_ids_str][:,0,:])
        # print(batch.obs[self.player_ids_str][:,20,:])
        # print(batch.obs[self.player_ids_str][:,-1,:])
        
        # print("INTENTS")
        # print(batch.obs["player_intents"].shape)
        # print(batch.obs["player_intents"][:,0,:])
        # print(batch.obs["player_intents"][:,10,:])
        # print(batch.obs["player_intents"][:,20,:])
        # print(batch.obs["player_intents"][:,40,:])
        # print(batch.obs["player_intents"][:,-1,:])
        
        # quit()

        if pred_weight > 0:
            if self.vdn:
                pred_loss1, seq_pred_loss = self.aux_task_vdn(
                    lstm_o,
                    batch.obs["own_hand"],
                    # batch.obs["temperature"], #not sure what this temperature variable it is not used
                    None,
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + pred_weight * pred_loss1
            else:
                pred_loss, seq_pred_loss = self.aux_task_iql(
                    lstm_o, batch.obs["own_hand"], batch.seq_len, rl_loss.size(), stat,
                )
                loss = rl_loss + pred_weight * pred_loss
        else:
            loss = rl_loss

        if intent_weight > 0 and self.intent_size > 0 and not self.train_adapt:
            # intent_loss, seq_intent_loss = self.aux_task_intent(
            #         lstm_o,
            #         batch.obs["player_intents"],
            #         None,
            #         batch.seq_len,
            #         rl_loss.size(),
            #         stat,
            #         use_xent_intent=self.use_xent_intent,
            #     )
            
            # if torch.any(torch.isnan(lstm_o)):
            #     print ("lstm_o is nan")
            #     # for dictname, dictthing in [('obs',batch.obs),('h0',batch.h0),('action',batch.action),('reward',batch.reward),('terminal',batch.terminal),('bootstrap',batch.bootstrap),('seq_len',batch.seq_len)]:
            #     #     if type(dictthing) is dict:
            #     #         for key in dictthing.keys():
            #     #             if torch.any(torch.isnan(dictthing[key])):
            #     #                 print(dictname, key, dictthing[key])
            #     #     else:
            #     #         if torch.any(torch.isnan(dictthing)):
            #     #             print(dictname, dictthing)
            #     print("rl_loss", rl_loss)
            #     print("intent_loss", intent_loss)
            # loss = loss + intent_weight * intent_loss#.detach()
            # priority += intent_weight * seq_intent_loss.abs()
            loss += intent_weight * intent_loss#.detach()
            priority += intent_weight * seq_intent_loss.abs()

        return loss, priority
