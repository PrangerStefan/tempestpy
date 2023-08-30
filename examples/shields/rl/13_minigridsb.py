from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from gymnasium.spaces import Dict, Box

from minigrid.core.actions import Actions

import numpy as np
import time

from helpers import create_shield_dict, parse_arguments, extract_keys, get_action_index_mapping, create_log_dir

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0, env=None):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        
        
    def _on_step(self) -> bool:
        #print(self.env.printGrid())
        return super()._on_step()


class MiniGridEnvWrapper(gym.core.Wrapper):
    def __init__(self, env, args=None, no_masking=False):
        super(MiniGridEnvWrapper, self).__init__(env)
        self.max_available_actions = env.action_space.n
        self.observation_space = env.observation_space.spaces["image"]
        
        self.args = args
        self.no_masking = no_masking

    def create_action_mask(self):
        if self.no_masking:
            return  np.array([1.0] * self.max_available_actions, dtype=np.int8)
        
        
        coordinates = self.env.agent_pos
        view_direction = self.env.agent_dir

        key_text = ""

        # only support one key for now
        if self.keys:
            key_text = F"!Agent_has_{self.keys[0]}_key\t& "


        if self.env.carrying and self.env.carrying.type == "key":
            key_text = F"Agent_has_{self.env.carrying.color}_key\t& "

        #print(F"Agent pos is {self.env.agent_pos} and direction {self.env.agent_dir} ")
        cur_pos_str = f"[{key_text}!AgentDone\t& xAgent={coordinates[0]}\t& yAgent={coordinates[1]}\t& viewAgent={view_direction}]"

        allowed_actions = []


        # Create the mask
        # If shield restricts action mask only valid with 1.0
        # else set all actions as valid
        mask = np.array([0.0] * self.max_available_actions, dtype=np.int8)

        if cur_pos_str in self.shield and self.shield[cur_pos_str]:
             allowed_actions = self.shield[cur_pos_str]
             for allowed_action in allowed_actions:
                 index =  get_action_index_mapping(allowed_action[1])
                 if index is None:
                     assert(False)
                 mask[index] = 1.0
        else:
            # print(F"Not in shield {cur_pos_str}")
            for index, x in enumerate(mask):
                mask[index] = 1.0
                
        front_tile = self.env.grid.get(self.env.front_pos[0], self.env.front_pos[1])

        if front_tile is not None and front_tile.type == "key":
            mask[Actions.pickup] = 1.0
            
        if self.env.carrying:
            mask[Actions.drop] = 1.0
            
        if front_tile and front_tile.type == "door":
            mask[Actions.toggle] = 1.0            
    
        

        return mask

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        keys = extract_keys(self.env)
        shield = create_shield_dict(self.env, self.args)
        
        self.keys = keys
        self.shield = shield
        return obs["image"], infos

    def step(self, action):
      #  print(F"Performed action in step: {action}")
        orig_obs, rew, done, truncated, info = self.env.step(action)

        #print(F"Original observation is {orig_obs}")
        obs = orig_obs["image"]

        #print(F"Info is {info}")
        return obs, rew, done, truncated, info



def mask_fn(env: gym.Env):
    return env.create_action_mask()
    


def main():
    import argparse
    args = parse_arguments(argparse)
    
    
    env = gym.make(args.env, render_mode="rgb_array")
    env = MiniGridEnvWrapper(env,args=args, no_masking=args.no_masking)
    env = ActionMasker(env, mask_fn)
    callback = CustomCallback(1, env)
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=create_log_dir(args))
    
    iterations = args.iterations
    
    if iterations < 10_000:
        iterations = 10_000
    
    model.learn(iterations, callback=callback)
 
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), 10)
    
    vec_env = model.get_env()
    obs = vec_env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        action_masks = None
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        # action, _states = model.predict(obs, deterministic=True)
        # obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.2)
    
    

if __name__ == '__main__':
    main()