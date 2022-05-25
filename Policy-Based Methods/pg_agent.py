import argparse
import pprint

import ray
from ray import tune
from ray.rllib.agents.pg.pg import DEFAULT_CONFIG
from ray.rllib.agents.pg.pg import PGTrainer as trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='LunarLanderContinuous-v2', help="Gym environment name.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config_update = {
        "env": args.env,
        "num_gpus": 0,
        "num_workers": 12,
        "evaluation_num_workers": 10,
        "evaluation_interval": 1,
    }
    config.update(config_update)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    ray.init()
    tune.run(trainer, stop={"timesteps_total": 200000}, config=config)
