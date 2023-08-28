from typing import Dict
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


from datetime import datetime

import gymnasium as gym

import minigrid
import numpy as np

import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.models.preprocessors import get_preprocessor
from MaskModels import TorchActionMaskModel
from Wrapper import OneHotWrapper, MiniGridEnvWrapper
from helpers import extract_keys, parse_arguments, create_shield_dict

import matplotlib.pyplot as plt




class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode started Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        episode.user_data["count"] = 0
        # print(env.printGrid())
        # print(env.action_space.n)
        # print(env.actions)
        # print(env.mission)
        # print(env.observation_space)
        # img = env.get_frame()
        # plt.imshow(img)
        # plt.show()
    
       
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
         episode.user_data["count"] = episode.user_data["count"] + 1
         env = base_env.get_sub_environments()[0]
         #print(env.printGrid())
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode end Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        # print(env.printGrid())
        # print(episode.user_data["count"])
        
                    

def env_creater_custom(config):
    framestack = config.get("framestack", 4)
    shield = config.get("shield", {})
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    
    env = gym.make(name)
    keys = extract_keys(env)
    env = MiniGridEnvWrapper(env, shield=shield, keys=keys)
    # env = minigrid.wrappers.ImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = OneHotWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
    
    return env

def create_log_dir(args):
    return F"{args.log_dir}{datetime.now()}-{args.algorithm}-masking:{not args.no_masking}"


def register_custom_minigrid_env(args):
    env_name = "mini-grid"
    register_env(env_name, env_creater_custom)

    ModelCatalog.register_custom_model(
        "pa_model", 
        TorchActionMaskModel
    )


def ppo(args):
    
    ray.init(num_cpus=3)

    
    register_custom_minigrid_env(args)
    shield_dict = create_shield_dict(args)
    
    config = (PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .environment(env="mini-grid", env_config={"shield": shield_dict, "name": args.env})
        .framework("torch")       
        .callbacks(MyCallbacks)
        .rl_module(_enable_rl_module_api = False)
        .debugging(logger_config={
            "type": "ray.tune.logger.TBXLogger", 
            "logdir": create_log_dir(args)
        })
        .training(_enable_learner_api=False ,model={
            "custom_model": "pa_model",
            "custom_model_config" : {"shield": shield_dict, "no_masking": args.no_masking}            
        }))
    
    algo =(
        
        config.build()
    )
    
    # while not terminated and not truncated:
    #     action = algo.compute_single_action(obs)
    #     obs, reward, terminated, truncated = env.step(action)
    
    for i in range(30):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            
    ray.shutdown()


def dqn(args):
    register_custom_minigrid_env(args)
    shield_dict = create_shield_dict(args)

    
    config = DQNConfig()
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)
    config = config.environment(env="mini-grid", env_config={"shield": shield_dict, "name": args.env })
    config = config.framework("torch")
    config = config.callbacks(MyCallbacks)
    config = config.rl_module(_enable_rl_module_api = False)
    config = config.debugging(logger_config={
            "type": "ray.tune.logger.TBXLogger", 
            "logdir": create_log_dir(args)
        })
    config = config.training(hiddens=[], dueling=False, model={    
            "custom_model": "pa_model",
            "custom_model_config" : {"shield": shield_dict, "no_masking": args.no_masking}
    })
    
    algo = (
        config.build()
    )
         
    for i in range(args.iterations):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            print("Saving checkpoint")
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            
    ray.shutdown()


def main():
    import argparse
    args = parse_arguments(argparse)

    if args.algorithm == "ppo":
        ppo(args)
    elif args.algorithm == "dqn":
        dqn(args)


   


if __name__ == '__main__':
    main()