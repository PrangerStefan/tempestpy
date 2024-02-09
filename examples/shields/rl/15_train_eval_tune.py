import gymnasium as gym
import minigrid

import ray
from ray.tune import register_env
from ray.tune.experiment.trial import Trial
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print, UnifiedLogger, CSVLogger
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.air import session

from torch_action_mask_model import TorchActionMaskModel
from rllibutils import OneHotShieldingWrapper, MiniGridShieldingWrapper, shielding_env_creater
from utils import MiniGridShieldHandler, create_shield_query, parse_arguments, create_log_dir, ShieldingConfig, test_name

from torch.utils.tensorboard import SummaryWriter
from callbacks import CustomCallback


def register_minigrid_shielding_env(args):
    env_name = "mini-grid-shielding"
    register_env(env_name, shielding_env_creater)

    ModelCatalog.register_custom_model(
        "shielding_model",
        TorchActionMaskModel
    )

def trial_name_creator(trial : Trial):
    return "trial"


def ppo(args):
    register_minigrid_shielding_env(args)
    logdir = args.log_dir

    config = (PPOConfig()
        .rollouts(num_rollout_workers=args.workers)
        .resources(num_gpus=args.num_gpus)
        .environment( env="mini-grid-shielding",
                      env_config={"name": args.env,
                                  "args": args,
                                  "shielding": args.shielding is ShieldingConfig.Full or args.shielding is ShieldingConfig.Training,
                                  },)
        .framework("torch")
        .callbacks(CustomCallback)
        .evaluation(evaluation_config={
                                       "evaluation_interval": 1,
                                        "evaluation_duration": 10,
                                        "evaluation_num_workers":1,
                                        "env": "mini-grid-shielding",
                                        "env_config": {"name": args.env,
                                                       "args": args,
                                                       "shielding": args.shielding is ShieldingConfig.Full or args.shielding is ShieldingConfig.Evaluation}})
        .rl_module(_enable_rl_module_api = False)
        .debugging(logger_config={
            "type": UnifiedLogger,
            "logdir": logdir
        })
        .training(_enable_learner_api=False ,model={
            "custom_model": "shielding_model"
        }))

    tuner = tune.Tuner("PPO",
                       tune_config=tune.TuneConfig(
                           metric="episode_reward_mean",
                           mode="max",
                           num_samples=1,
                           trial_name_creator=trial_name_creator,

                       ),
                        run_config=air.RunConfig(
                                stop = {"episode_reward_mean": 1,
                                        "timesteps_total": args.steps,},
                                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True,
                                                                       num_to_keep=1,
                                                                       checkpoint_score_attribute="episode_reward_mean",
                                                                       ),

                               storage_path=F"{logdir}",
                               name=test_name(args),


                        ),
    param_space=config,)

    results = tuner.fit()
    best_result = results.get_best_result()

    import pprint

    metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

def main():
    ray.init(num_cpus=3)
    import argparse
    args = parse_arguments(argparse)

    ppo(args)

    ray.shutdown()

if __name__ == '__main__':
    main()
