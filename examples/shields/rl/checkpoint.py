

import gymnasium as gym
import minigrid

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.algorithm import Algorithm

from torch_action_mask_model import TorchActionMaskModel
from rllibutils import OneHotShieldingWrapper, MiniGridShieldingWrapper
from utils import parse_arguments, create_log_dir, ShieldingConfig
from utils import MiniGridShieldHandler, create_shield_query
from callbacks import CustomCallback

from ray.tune.logger import TBXLogger   
import imageio
import os

import matplotlib.pyplot as plt


def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaSlipperyS12-v2")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.grid_path}_{config.worker_index}.txt"
    args.prism_path = F"{args.prism_path}_{config.worker_index}.prism"
    
    shield_creator = MiniGridShieldHandler(args.grid_path, args.grid_to_prism_binary_path, args.prism_path, args.formula)
    
    env = gym.make(name, randomize_start=False)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query, mask_actions=False)
    # env = minigrid.wrappers.ImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = OneHotShieldingWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
    
    env.randomize_start = False

    
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
# checkpoints = [('/home/knolli/Documents/University/Thesis/log_results/sh:none-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_low.yaml/checkpoint_000030', 'No_shield'),
#                         ("/home/knolli/Documents/University/Thesis/log_results/Relative_06/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_high.yaml/checkpoint_000030", "Rel_06_high"),
#                         ("/home/knolli/Documents/University/Thesis/log_results/Relative_06/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_medium.yaml/checkpoint_000030", "Rel_06_med"),
#                         ("/home/knolli/Documents/University/Thesis/log_results/Relative_06/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_low.yaml/checkpoint_000030", "Rel_06_low"),
#                         ("/home/knolli/Documents/University/Thesis/log_results/RELATIVE_1/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_high.yaml/checkpoint_000016", "Rel_1_high"),
#                         ("/home/knolli/Documents/University/Thesis/log_results/RELATIVE_1/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_medium.yaml/checkpoint_000030", "Rel_1_med"),
#                         ("/home/knolli/Documents/University/Thesis/log_results/RELATIVE_1/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:adv_config_slippery_low.yaml/checkpoint_000030", "Rel_1_low")]
checkpoints = [
    # ('/home/knolli/Documents/University/Thesis/log_results/sh:none-value:0.9-env:MiniGrid-LavaSlipperyS12-v2-conf:slippery_high_pro.yaml/checkpoint_000070', "no_shielding"),
                # ('/home/knolli/Documents/University/Thesis/log_results/sh:full-value:0.9-env:MiniGrid-LavaSlipperyS12-v2-conf:slippery_high_pro.yaml/checkpoint_000070', "shielding_09"),
                # ('/home/knolli/Documents/University/Thesis/log_results/sh:full-value:1.0-env:MiniGrid-LavaSlipperyS12-v2-conf:slippery_high_pro.yaml/checkpoint_000070', "shielding_1")]
('/home/knolli/Documents/University/Thesis/logresults/exp/trial_0_2024-01-09_22-39-43/checkpoint_000002', 'v3')]

# checkpoints = [('/home/knolli/Documents/University/Thesis/log_results/sh:full-env:MiniGrid-LavaSlipperyS12-v2-conf:slippery_high_prob.yaml/checkpoint_000060', "Shielded_Gif")]
for path_to_checkpoint, gif_name in checkpoints:
    algo = Algorithm.from_checkpoint(path_to_checkpoint)
    policy = algo.get_policy()
    # Continue training.
    name = "MiniGrid-LavaSlipperyS12-v0"
    shield_creator = MiniGridShieldHandler(F"./{args.grid_path}_1.txt", args.grid_to_prism_binary_path, F"./{args.prism_path}_1.prism", args.formula)

    env = gym.make(name, randomize_start=False, probability_forward=3/9, probability_direct_neighbour=5/9, probability_next_neighbour=7/9,)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query, mask_actions=True)
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
        policy_actions = policy.compute_single_action(obs)
        # print(f'Policy actions {policy_actions}')
        # print(f'Policy actions {policy_actions.logits}')
        policy_action = policy_actions[2]['action_dist_inputs'].argmax()
        # print(f'The action is: {action} vs policy action {policy_action}')
        
        if policy_action != action:
            print('policy action deviated')
            action = policy_action
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
    imageio.mimsave(F'./{gif_name}.gif', images)

    for filename in filenames:
        os.remove(filename)
    