from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym

from minigrid.core.actions import Actions

import time

from helpers import parse_arguments, create_log_dir, ShieldingConfig
from shieldhandlers import MiniGridShieldHandler, create_shield_query
from wrappers import MiniGridSbShieldingWrapper

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0, env=None):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        
        
    def _on_step(self) -> bool:
        print(self.env.printGrid())
        return super()._on_step()




def mask_fn(env: gym.Env):
    return env.create_action_mask()
    


def main():
    import argparse
    args = parse_arguments(argparse)
    
    args.grid_path = F"{args.grid_path}.txt"
    args.prism_path = F"{args.prism_path}.prism"
    
    shield_creator = MiniGridShieldHandler(args.grid_path, args.grid_to_prism_binary_path, args.prism_path, args.formula)
    
    env = gym.make(args.env, render_mode="rgb_array")
    env = MiniGridSbShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query, mask_actions=args.shielding == ShieldingConfig.Full)
    env = ActionMasker(env, mask_fn)
    callback = CustomCallback(1, env)
    model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.4, verbose=1, tensorboard_log=create_log_dir(args))
    
    iterations = args.iterations
    
    if iterations < 10_000:
        iterations = 10_000
    
    model.learn(iterations, callback=callback)
 
  #W  mean_reward, std_reward = evaluate_policy(model, model.get_env())
    
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