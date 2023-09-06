from typing import Dict, Optional, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import stormpy
import stormpy.core
import stormpy.simulator

from collections import deque

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files
import os

import gymnasium as gym

import minigrid
import numpy as np

import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved, framework_iterator
from ray import tune, air
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from ray.rllib.utils.numpy import one_hot
from ray.rllib.algorithms import ppo

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.models.preprocessors import get_preprocessor

import matplotlib.pyplot as plt

import argparse
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from examples.shields.rl.Wrappers import OneHotShieldingWrapper


torch, nn = try_import_torch()

class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()



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
        #  print(env.env.env.printGrid())
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode end Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        # print(env.env.env.printGrid())
        # print(episode.user_data["count"])
        
    
    


def parse_arguments(argparse):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="gym environment to load", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--env", help="gym environment to load", default="MiniGrid-LavaCrossingS9N1-v0")
    parser.add_argument("--seed", type=int, help="seed for environment", default=1)
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32)
    parser.add_argument("--agent_view", default=False, action="store_true", help="draw the agent sees")
    parser.add_argument("--grid_path", default="Grid.txt")
    parser.add_argument("--prism_path", default="Grid.PRISM")
    
    args = parser.parse_args()
    
    return args


def env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    # name = config.get("name", "MiniGrid-Empty-8x8-v0")
    framestack = config.get("framestack", 4)
    
    env = gym.make(name)
    # env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    env = minigrid.wrappers.ImgObsWrapper(env)
    env = OneHotShieldingWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
      

    return env


def create_shield(grid_file, prism_path):
    os.system(F"/home/tknoll/Documents/main -v 'agent' -i {grid_file} -o {prism_path}")
    
    f = open(prism_path, "a")
    f.write("label \"AgentIsInLava\" = AgentIsInLava;")
    f.close()
    
    
    program = stormpy.parse_prism_program(prism_path)
    formula_str = "Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"
    
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)
    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
    
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.1) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    
    assert result.has_scheduler
    assert result.has_shield
    shield = result.shield

    stormpy.shields.export_shield(model, shield,"Grid.shield")
    
    return shield.construct(), model

def export_grid_to_text(env, grid_file):
    f = open(grid_file, "w")
    print(env)
    f.write(env.printGrid(init=True))
    # f.write(env.pprint_grid())
    f.close()

def create_environment(args):
    env_id= args.env
    env = gym.make(env_id)
    env.reset()
    return env


def main():
    args = parse_arguments(argparse)

    env = create_environment(args)
    ray.init(num_cpus=3)

    # print(env.pprint_grid())
    # print(env.printGrid(init=False))
    
    grid_file = args.grid_path
    export_grid_to_text(env, grid_file)
    
    prism_path = args.prism_path
    shield, model = create_shield(grid_file, prism_path)
    
    for state_id in model.states:
        choices = shield.get_choice(state_id)
        print(F"Allowed choices in state {state_id}, are {choices.choice_map} ")
        
    env_name = "mini-grid"
    register_env(env_name, env_creater)
    
  
    algo =(
        PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .environment(env="mini-grid")
        .framework("torch")
        .callbacks(MyCallbacks)
        .training(model={
            "fcnet_hiddens": [256,256],
            "fcnet_activation": "relu",
            
        })
        .build()
    )
    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset()
    
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

if __name__ == '__main__':
    main()