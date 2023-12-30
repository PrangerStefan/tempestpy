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
from ray.air import session

from torch_action_mask_model import TorchActionMaskModel
from wrappers import OneHotShieldingWrapper, MiniGridShieldingWrapper
from helpers import parse_arguments, create_log_dir, ShieldingConfig, test_name
from shieldhandlers import MiniGridShieldHandler, create_shield_query

from torch.utils.tensorboard import SummaryWriter
from callbacks import MyCallbacks, ShieldInfoCallback


def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N3-v0")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.expname}_{args.grid_path}_{config.worker_index}.txt"
    args.prism_path = F"{args.expname}_{args.prism_path}_{config.worker_index}.prism"
    shielding = config.get("shielding", False)
    shield_creator = MiniGridShieldHandler(grid_file=args.grid_path,
                                           grid_to_prism_path=args.grid_to_prism_binary_path,
                                           prism_path=args.prism_path,
                                           formula=args.formula,
                                           shield_value=args.shield_value,
                                           prism_config=args.prism_config,
                                           shield_comparision=args.shield_comparision)

    prob_forward = args.prob_forward
    prob_direct = args.prob_direct
    prob_next = args.prob_next

    env = gym.make(name, randomize_start=True,probability_forward=prob_forward, probability_direct_neighbour=prob_direct, probability_next_neighbour=prob_next)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query ,mask_actions=shielding != ShieldingConfig.Disabled)

    env = OneHotShieldingWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )


    return env


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
        .callbacks(MyCallbacks, ShieldInfoCallback(logdir, [1,12]))
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
                                stop = {"episode_reward_mean": 94,
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

   # algo = Algorithm.from_checkpoint(best_result.checkpoint)


    # eval_log_dir = F"{logdir}-eval"

    # writer = SummaryWriter(log_dir=eval_log_dir)
    # csv_logger = CSVLogger(config=config, logdir=eval_log_dir)


    # for i in range(args.evaluations):
    #     eval_result = algo.evaluate()
    #     print(pretty_print(eval_result))
    #     print(eval_result)
    #     # logger.on_result(eval_result)

    #     csv_logger.on_result(eval_result)

    #     evaluation = eval_result['evaluation']
    #     epsiode_reward_mean = evaluation['episode_reward_mean']
    #     episode_len_mean = evaluation['episode_len_mean']
    #     print(epsiode_reward_mean)
    #     writer.add_scalar("evaluation/episode_reward_mean", epsiode_reward_mean, i)
    #     writer.add_scalar("evaluation/episode_len_mean", episode_len_mean, i)


def main():
    ray.init(num_cpus=3)
    import argparse
    args = parse_arguments(argparse)

    ppo(args)

    ray.shutdown()

if __name__ == '__main__':
    main()
