# Run Tools

`hanabi/pyhanabi/tools` contains some example scripts to launch training runs. 

* `dev.sh` is a fast lauching script for debugging. It needs 2 gpus to run, 1 for training and 1 for simulation.
  * It launches 2-player self-play with IQL and some standard parameters

The important flags are:

`--sad 1` to enable "Simplified Action Decoder";

`--pred_weight 0.25` to enable auxiliary task and multiply aux loss with 0.25;

`--shuffle_color 1` to enable other-play.

```bash
cd pyhanabi
sh tools/dev.sh
```

* `eval_models` evaluates self-play with a given model
```
cd pyhanabi
python tools/eval_model.py --weight ../models/sad_2p_10.pthw --num_player 2
```

Other scripts are examples for a more formal training runs with specific types of training, they require 3 gpus, 1 for training and 2 for simulation.

