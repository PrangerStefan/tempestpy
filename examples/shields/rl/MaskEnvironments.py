import random
import minigrid

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from Wrapper import OneHotWrapper


class ParametricActionsMiniGridEnv(gym.Env):
    """Parametric action version of MiniGrid.

    """

    def __init__(self, config):
       
        name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
        self.left_action_embed = np.random.randn(2)
        self.right_action_embed = np.random.randn(2)
        framestack = config.get("framestack", 4)
        
        # env = gym.make(name)
        # env = minigrid.wrappers.ImgObsWrapper(env)
        # env = OneHotWrapper(env,
        #                 config.vector_index if hasattr(config, "vector_index") else 0,
        #                 framestack=framestack
        #                 )
        self.wrapped = gym.make(name)
        # self.observation_space = Dict(
        #     {
        #          "action_mask": None,
        #          "avail_actions": None,
        #         "cart": self.wrapped.observation_space,
        #     }
        # )
        print(F"Wrapped environment is {self.wrapped}")
        self.step_count = 0
        self.action_space = self.wrapped.action_space
        self.observation_space = self.wrapped.observation_space
        
        
    def update_avail_actions(self):
        self.action_assignments = np.array(
            [[0.0, 0.0]] * self.action_space.n, dtype=np.float32
        )
        self.action_mask = np.array([0.0] * self.action_space.n, dtype=np.int8)
        self.left_idx, self.right_idx = random.sample(range(self.action_space.n), 2)
        self.action_assignments[self.left_idx] = self.left_action_embed
        self.action_assignments[self.right_idx] = self.right_action_embed
        self.action_mask[self.left_idx] = 1
        self.action_mask[self.right_idx] = 1

    def reset(self, *, seed=None, options=None):
        self.update_avail_actions()
        obs, infos = self.wrapped.reset()
        return obs, infos
        return {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": obs,
        }, infos

    def step(self, action):
        if action == self.left_idx:
            actual_action = 0
        elif action == self.right_idx:
            actual_action = 1
        else:
            actual_action = 0
            # raise ValueError(
            #     "Chosen action was not one of the non-zero action embeddings",
            #     action,
            #     self.action_assignments,
            #     self.action_mask,
            #     self.left_idx,
            #     self.right_idx,
            # )
        orig_obs, rew, done, truncated, info = self.wrapped.step(actual_action)
        self.update_avail_actions()
        self.action_mask = self.action_mask.astype(np.int8)
        print(F"Info is {info}")
        info["Hello" : "Ich kenn mich nix aus"]
        return orig_obs, rew, done, truncated, info
        obs = {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": orig_obs,
        }
        return obs, rew, done, truncated, info

   