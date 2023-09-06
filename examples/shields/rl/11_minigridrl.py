# from typing import Dict
# from ray.rllib.env.base_env import BaseEnv
# from ray.rllib.evaluation import RolloutWorker
# from ray.rllib.evaluation.episode import Episode
# from ray.rllib.evaluation.episode_v2 import EpisodeV2
# from ray.rllib.policy import Policy
# from ray.rllib.utils.typing import PolicyID

import gymnasium as gym

import minigrid
# import numpy as np

# import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog


from TorchActionMaskModel import TorchActionMaskModel
from Wrappers import OneHotShieldingWrapper, MiniGridShieldingWrapper
from helpers import parse_arguments, create_log_dir
from ShieldHandlers import MiniGridShieldHandler

import matplotlib.pyplot as plt

from ray.tune.logger import TBXLogger   

# class MyCallbacks(DefaultCallbacks):
#     def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
#         # print(F"Epsiode started Environment: {base_env.get_sub_environments()}")
#         env = base_env.get_sub_environments()[0]
#         episode.user_data["count"] = 0
#         # print("On episode start print")
#         # print(env.printGrid())
#         # print(worker)
#         # print(env.action_space.n)
#         # print(env.actions)
#         # print(env.mission)
#         # print(env.observation_space)
#         # img = env.get_frame()
#         # plt.imshow(img)
#         # plt.show()
    
       
#     def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
#          episode.user_data["count"] = episode.user_data["count"] + 1
#          env = base_env.get_sub_environments()[0]
#         # print(env.printGrid())
    
#     def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
#         # print(F"Epsiode end Environment: {base_env.get_sub_environments()}")
#         env = base_env.get_sub_environments()[0]
#         #print("On episode end print")
#         #print(env.printGrid())
        
                    

def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.grid_path}_{config.worker_index}.txt"
    args.prism_path = F"{args.prism_path}_{config.worker_index}.prism"
    
    shield_creator = MiniGridShieldHandler(args.grid_path, args.grid_to_prism_binary_path, args.prism_path, args.formula)
    
    env = gym.make(name)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator)
    # env = minigrid.wrappers.ImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = OneHotShieldingWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
    
    
    return env



def register_minigrid_shielding_env(args):
    env_name = "mini-grid"
    register_env(env_name, shielding_env_creater)

    ModelCatalog.register_custom_model(
        "shielding_model", 
        TorchActionMaskModel
    )


def ppo(args):
    register_minigrid_shielding_env(args)
    
    config = (PPOConfig()
        .rollouts(num_rollout_workers=args.workers)
        .resources(num_gpus=0)
        .environment(env="mini-grid", env_config={"name": args.env, "args": args})
        .framework("torch")
        #.callbacks(MyCallbacks)
        .rl_module(_enable_rl_module_api = False)
        .debugging(logger_config={
            "type": TBXLogger, 
            "logdir": create_log_dir(args)
        })
        .training(_enable_learner_api=False ,model={
            "custom_model": "shielding_model",
            "custom_model_config" : {"no_masking": args.no_masking}            
        }))
    
    algo =(
        
        config.build()
    )
    
    algo.eva
    
    for i in range(args.iterations):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            

def dqn(args):
    register_minigrid_shielding_env(args)

    
    config = DQNConfig()
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=args.workers)
    config = config.environment(env="mini-grid", env_config={"name": args.env, "args": args })
    config = config.framework("torch")
    #config = config.callbacks(MyCallbacks)
    config = config.rl_module(_enable_rl_module_api = False)
    config = config.debugging(logger_config={
            "type": TBXLogger, 
            "logdir": create_log_dir(args)
        })
    config = config.training(hiddens=[], dueling=False, model={    
            "custom_model": "shielding_model",
            "custom_model_config" : {"no_masking": args.no_masking}
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
            

def main():
    import argparse
    args = parse_arguments(argparse)

    if args.algorithm == "ppo":
        ppo(args)
    elif args.algorithm == "dqn":
        dqn(args)


   


if __name__ == '__main__':
    main()