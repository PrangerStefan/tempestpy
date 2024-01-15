import gymnasium as gym
import numpy as np
import random

from utils import MiniGridShieldHandler, common_parser
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import Image

class MiniGridSbShieldingWrapper(gym.core.Wrapper):
    def __init__(self,
                 env,
                 shield_handler : MiniGridShieldHandler,
                 create_shield_at_reset = True,
                 ):
        super().__init__(env)
        self.shield_handler = shield_handler
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

        if self.create_shield_at_reset:
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

class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self):
        image = self.training_env.render(mode="rgb_array")
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self):
        return True


class InfoCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.sum_goal = 0
        self.sum_lava = 0
        self.sum_collisions = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        if infos["reached_goal"]:
            self.sum_goal += 1
        if infos["ran_into_lava"]:
            self.sum_lava += 1
        self.logger.record("info/sum_reached_goal", self.sum_goal)
        self.logger.record("info/sum_ran_into_lava", self.sum_lava)
        if "collision" in infos:
            if infos["collision"]:
                self.sum_collision += 1
            self.logger.record("info/sum_collision", sum_collisions)
        return True
