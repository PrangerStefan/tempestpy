import gymnasium as gym
import numpy as np
import random

from utils import MiniGridShieldHandler, common_parser

class MiniGridSbShieldingWrapper(gym.core.Wrapper):
    def __init__(self,
                 env,
                 shield_handler : MiniGridShieldHandler,
                 create_shield_at_reset = True,
                 mask_actions=True,
                 ):
        super().__init__(env)
        self.shield_handler = shield_handler
        self.mask_actions = mask_actions
        self.create_shield_at_reset = create_shield_at_reset

        shield = self.shield_handler.create_shield(env=self.env)
        self.shield = shield

    def create_action_mask(self):
        try:
            return self.shield[self.env.get_symbolic_state()]
        except:
            return [1.0] * 3 + [1.0] * 4

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)

        if self.create_shield_at_reset and self.mask_actions:
            shield = self.shield_handler.create_shield(env=self.env)
            self.shield = shield
        return obs, infos

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)

        return obs, rew, done, truncated, info

def parse_sb3_arguments():
    parser = common_parser()
    args = parser.parse_args()

    return args
