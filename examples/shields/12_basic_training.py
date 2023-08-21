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

from ray.rllib.models.preprocessors import get_preprocessor

import matplotlib.pyplot as plt

import argparse


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
        
    
       

class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32
        )

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    print(
                        "100-average dist travelled={}".format(
                            np.mean(self.x_y_delta_buffer)
                        )
                    )
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

      
        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
      
        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)



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
    env = OneHotWrapper(env,
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