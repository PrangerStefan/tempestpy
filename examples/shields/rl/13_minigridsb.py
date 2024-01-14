from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import gymnasium as gym

from minigrid.core.actions import Actions

import time

from utils import MiniGridShieldHandler, create_log_dir, ShieldingConfig
from sb3utils import MiniGridSbShieldingWrapper, parse_sb3_arguments

GRID_TO_PRISM_BINARY="/home/spranger/research/tempestpy/Minigrid2PRISM/build/main"

def mask_fn(env: gym.Env):
    return env.create_action_mask()


def main():
    args = parse_sb3_arguments()

    formula = args.formula
    shield_value = args.shield_value
    shield_comparison = args.shield_comparison

    shield_handler = MiniGridShieldHandler(GRID_TO_PRISM_BINARY, args.grid_file, args.prism_output_file, formula, shield_value=shield_value, shield_comparison=shield_comparison)
    env = gym.make(args.env, render_mode="rgb_array")
    env = MiniGridSbShieldingWrapper(env, shield_handler=shield_handler, create_shield_at_reset=False, mask_actions=args.shielding == ShieldingConfig.Full)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.4, verbose=1, tensorboard_log=create_log_dir(args))

    steps = args.steps


    model.learn(steps)


    print("Learning done, hit enter")
    input("")
    vec_env = model.get_env()
    obs = vec_env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        action_masks = None
        action, _states = model.predict(obs, action_masks=action_masks)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        # action, _states = model.predict(obs, deterministic=True)
        # obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.2)



if __name__ == '__main__':
    main()
