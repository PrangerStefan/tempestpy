from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import Logger, CSVOutputFormat, TensorBoardOutputFormat, HumanOutputFormat

import gymnasium as gym

from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

import time

from utils import MiniGridShieldHandler, create_log_dir, ShieldingConfig, MiniWrapper, expname, shield_needed, shielded_evaluation
from sb3utils import MiniGridSbShieldingWrapper, parse_sb3_arguments, ImageRecorderCallback, InfoCallback
from stable_baselines3.common.callbacks import EvalCallback

import os, sys
from copy import deepcopy

GRID_TO_PRISM_BINARY=os.getenv("M2P_BINARY")
def mask_fn(env: gym.Env):
    return env.create_action_mask()

def nomask_fn(env: gym.Env):
    return [1.0] * 7


def main():
    args = parse_sb3_arguments()


    formula = args.formula
    shield_value = args.shield_value
    shield_comparison = args.shield_comparison
    log_dir = create_log_dir(args)
    #new_logger = Logger(log_dir, output_formats=[CSVOutputFormat(os.path.join(log_dir, f"progress_{expname(args)}.csv")), TensorBoardOutputFormat(log_dir)])
    new_logger = Logger(log_dir, output_formats=[CSVOutputFormat(os.path.join(log_dir, f"progress_{expname(args)}.csv")), TensorBoardOutputFormat(log_dir), HumanOutputFormat(sys.stdout)])


    if shield_needed(args.shielding):
        shield_handler = MiniGridShieldHandler(GRID_TO_PRISM_BINARY, args.grid_file, args.prism_output_file, formula, shield_value=shield_value, shield_comparison=shield_comparison, nocleanup=args.nocleanup)


    env = gym.make(args.env, render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = MiniWrapper(env)
    eval_env = deepcopy(env)
    eval_env.disable_random_start()
    if args.shielding == ShieldingConfig.Full:
        env = MiniGridSbShieldingWrapper(env, shield_handler=shield_handler, create_shield_at_reset=False)
        env = ActionMasker(env, mask_fn)
        eval_env = MiniGridSbShieldingWrapper(eval_env, shield_handler=shield_handler, create_shield_at_reset=False)
        eval_env = ActionMasker(eval_env, mask_fn)
    elif args.shielding == ShieldingConfig.Training:
        env = MiniGridSbShieldingWrapper(env, shield_handler=shield_handler, create_shield_at_reset=False)
        env = ActionMasker(env, mask_fn)
        eval_env = ActionMasker(eval_env, nomask_fn)
    elif args.shielding == ShieldingConfig.Evaluation:
        env = ActionMasker(env, nomask_fn)
        eval_env = MiniGridSbShieldingWrapper(eval_env, shield_handler=shield_handler, create_shield_at_reset=False)
        eval_env = ActionMasker(eval_env, mask_fn)
    elif args.shielding == ShieldingConfig.Disabled:
        env = ActionMasker(env, nomask_fn)
        eval_env = ActionMasker(eval_env, nomask_fn)
    else:
        assert(False) # TODO Do something proper
    model = MaskablePPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, device="auto")
    model.set_logger(new_logger)
    steps = args.steps


    # Evaluation
    eval_freq=max(500, int(args.steps/30))
    n_eval_episodes=5
    render_freq = eval_freq
    if shielded_evaluation(args.shielding):
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        evalCallback = MaskableEvalCallback(eval_env, best_model_save_path=log_dir,
                                            log_path=log_dir, eval_freq=eval_freq,
                                            deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
        imageAndVideoCallback = ImageRecorderCallback(eval_env, render_freq, n_eval_episodes=1, evaluation_method=evaluate_policy, log_dir=log_dir, deterministic=True, verbose=0)
    else:
        from stable_baselines3.common.evaluation import evaluate_policy
        evalCallback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                    log_path=log_dir, eval_freq=eval_freq,
                                    deterministic=True, render=False, n_eval_episodes=n_eval_episodes)

        imageAndVideoCallback = ImageRecorderCallback(eval_env, render_freq, n_eval_episodes=1, evaluation_method=evaluate_policy, log_dir=log_dir, deterministic=True, verbose=0)


    model.learn(steps,callback=[imageAndVideoCallback, InfoCallback(), evalCallback])
    model.save(f"{log_dir}/{expname(args)}")


if __name__ == '__main__':
    main()
