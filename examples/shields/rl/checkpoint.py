

import gymnasium as gym
import minigrid

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.algorithm import Algorithm

from torch_action_mask_model import TorchActionMaskModel
from wrappers import OneHotShieldingWrapper, MiniGridShieldingWrapper
from helpers import parse_arguments, create_log_dir, ShieldingConfig
from shieldhandlers import MiniGridShieldHandler, create_shield_query
from callbacks import MyCallbacks

from ray.tune.logger import TBXLogger   
import imageio

import matplotlib.pyplot as plt


def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaSlipperyS12-v2")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.grid_path}_{config.worker_index}.txt"
    args.prism_path = F"{args.prism_path}_{config.worker_index}.prism"
    
    shield_creator = MiniGridShieldHandler(args.grid_path, args.grid_to_prism_binary_path, args.prism_path, args.formula)
    
    env = gym.make(name)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query)
    # env = minigrid.wrappers.ImgObsWrapper(env)
    # env = ImgObsWrapper(env)
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
    
import argparse
args = parse_arguments(argparse)
    
register_minigrid_shielding_env(args)
    
# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:
path_to_checkpoint = '/home/tknoll/Documents/Projects/log_results/PPO-shielding:full-evaluations:10-steps:20000-env:MiniGrid-LavaSlipperyS12-v2/PPO/PPO_mini-grid-shielding_8cd74_00000_0_2023-09-13_14-10-38/checkpoint_000005'


algo = Algorithm.from_checkpoint(path_to_checkpoint)

# Continue training.
name = "MiniGrid-LavaSlipperyS12-v2"
shield_creator = MiniGridShieldHandler(F"./{args.grid_path}_1.txt", args.grid_to_prism_binary_path, F"./{args.prism_path}_1.prism", args.formula)

env = gym.make(name)
env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query)
# env = minigrid.wrappers.ImgObsWrapper(env)
# env = ImgObsWrapper(env)
env = OneHotShieldingWrapper(env,
                    0,
                    framestack=4
                    )

episode_reward = 0
terminated = truncated = False

obs, info = env.reset()
i = 0
filenames = []
while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    filename = F"./frames/{i}.jpg"
    img = env.get_frame()
    plt.imsave(filename, img)
    filenames.append(filename)
    i  = i + 1
    
import imageio
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./movie.gif', images)
    