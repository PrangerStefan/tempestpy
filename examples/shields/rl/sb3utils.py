import gymnasium as gym
import numpy as np
import random

from utils import MiniGridShieldHandler, create_shield_query

class MiniGridSbShieldingWrapper(gym.core.Wrapper):
    def __init__(self, 
                 env, 
                 shield_creator : MiniGridShieldHandler,
                 shield_query_creator,
                 create_shield_at_reset = True,
                 mask_actions=True,
                 ):
        super(MiniGridSbShieldingWrapper, self).__init__(env)
        self.max_available_actions = env.action_space.n
        self.observation_space = env.observation_space.spaces["image"]
        
        self.shield_creator = shield_creator
        self.mask_actions = mask_actions
        self.shield_query_creator = shield_query_creator

    def create_action_mask(self):
        if not self.mask_actions:
            return  np.array([1.0] * self.max_available_actions, dtype=np.int8)
               
        cur_pos_str = self.shield_query_creator(self.env)
        
        allowed_actions = []

        # Create the mask
        # If shield restricts actions, mask only valid actions with 1.0
        # else set all actions valid
        mask = np.array([0.0] * self.max_available_actions, dtype=np.int8)

        if cur_pos_str in self.shield and self.shield[cur_pos_str]:
            allowed_actions = self.shield[cur_pos_str]
            for allowed_action in allowed_actions:
                index =  get_action_index_mapping(allowed_action.labels)
                if index is None:
                     assert(False)
                              
                mask[index] = random.choices([0.0, 1.0], weights=(1 - allowed_action.prob, allowed_action.prob))[0]
        else:
            for index, x in enumerate(mask):
                mask[index] = 1.0
        
        front_tile = self.env.grid.get(self.env.front_pos[0], self.env.front_pos[1])

            
        if front_tile and front_tile.type == "door":
            mask[Actions.toggle] = 1.0            
            
        return mask  
    

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
      
        shield = self.shield_creator.create_shield(env=self.env)
        
        self.shield = shield
        return obs["image"], infos

    def step(self, action):
        orig_obs, rew, done, truncated, info = self.env.step(action)
        obs = orig_obs["image"]
        
        return obs, rew, done, truncated, info

