from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

import gymnasium as gym

from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

import time

from utils import MiniGridShieldHandler, create_log_dir, ShieldingConfig, MiniWrapper
from sb3utils import MiniGridSbShieldingWrapper, parse_sb3_arguments, ImageRecorderCallback, InfoCallback
from stable_baselines3.common.callbacks import EvalCallback

import os

GRID_TO_PRISM_BINARY=os.getenv("M2P_BINARY")
def mask_fn(env: gym.Env):
    return env.create_action_mask()


def main():
    args = parse_sb3_arguments()

    formula = args.formula
    shield_value = args.shield_value
    shield_comparison = args.shield_comparison

    shield_handler = MiniGridShieldHandler(GRID_TO_PRISM_BINARY, args.grid_file, args.prism_output_file, formula, shield_value=shield_value, shield_comparison=shield_comparison)
    env = gym.make(args.env, render_mode="rgb_array")
    env = RGBImgObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    env = MiniWrapper(env)
    env = MiniGridSbShieldingWrapper(env, shield_handler=shield_handler, create_shield_at_reset=False, mask_actions=args.shielding == ShieldingConfig.Full)
    env = ActionMasker(env, mask_fn)
    logDir = create_log_dir(args)
    model = MaskablePPO("CnnPolicy", env, verbose=1, tensorboard_log=logDir)

    evalCallback = EvalCallback(env, best_model_save_path=logDir,
                                log_path=logDir, eval_freq=max(500,  int(args.steps/30)),
                                deterministic=True, render=False)
    steps = args.steps

    model.learn(steps,callback=[ImageRecorderCallback(), InfoCallback()])


    #vec_env = model.get_env()
    #obs = vec_env.reset()
    #terminated = truncated = False
    #while not terminated and not truncated:
    #    action_masks = None
    #    action, _states = model.predict(obs, action_masks=action_masks)
    #    print(action)
    #    obs, reward, terminated, truncated, info = env.step(action)
    #    # action, _states = model.predict(obs, deterministic=True)
    #    # obs, rewards, dones, info = vec_env.step(action)
    #    vec_env.render("human")
    #    time.sleep(0.2)



if __name__ == '__main__':
    main()
