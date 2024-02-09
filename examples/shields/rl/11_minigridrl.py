import gymnasium as gym
import minigrid

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog


from torch_action_mask_model import TorchActionMaskModel
from rllibutils import OneHotShieldingWrapper, MiniGridShieldingWrapper, shielding_env_creater
from utils import MiniGridShieldHandler, create_shield_query, parse_arguments, create_log_dir, ShieldingConfig
from callbacks import CustomCallback

from ray.tune.logger import TBXLogger   


def register_minigrid_shielding_env(args):
    env_name = "mini-grid-shielding"
    register_env(env_name, shielding_env_creater)

    ModelCatalog.register_custom_model(
        "shielding_model", 
        TorchActionMaskModel
    )


def ppo(args):
    train_batch_size = 4000
    register_minigrid_shielding_env(args)
    
    config = (PPOConfig()
        .rollouts(num_rollout_workers=args.workers)
        .resources(num_gpus=0)
        .environment(env="mini-grid-shielding", env_config={"name": args.env, "args": args, "shielding": args.shielding is ShieldingConfig.Full or args.shielding is ShieldingConfig.Training})
        .framework("torch")
        .callbacks(CustomCallback)
        .rl_module(_enable_rl_module_api = False)
        .debugging(logger_config={
            "type": TBXLogger, 
            "logdir": create_log_dir(args)
        })    
        # .exploration(exploration_config={"exploration_fraction": 0.1})
        .training(_enable_learner_api=False ,
            model={"custom_model": "shielding_model"},
            train_batch_size=train_batch_size))
    # config.entropy_coeff =  0.05
    algo =(   
        config.build()
    )   
    
    
    iterations = int((args.steps / train_batch_size)) + 1
    for i in range(iterations):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    
    algo.save()
            

def dqn(args):
    train_batch_size = 4000
    register_minigrid_shielding_env(args)

    
    config = DQNConfig()
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=args.workers)
    config = config.environment(env="mini-grid-shielding", env_config={"name": args.env, "args": args })
    config = config.framework("torch")
    config = config.callbacks(CustomCallback)
    config = config.rl_module(_enable_rl_module_api = False)
    config = config.debugging(logger_config={
            "type": TBXLogger, 
            "logdir": create_log_dir(args)
        })
    config = config.training(hiddens=[], dueling=False, train_batch_size=train_batch_size, model={    
            "custom_model": "shielding_model"
    })
    
    algo = (
        config.build()
    )

    iterations = int((args.steps / train_batch_size)) + 1
    for i in range(iterations):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            print("Saving checkpoint")
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            

def main():
    import argparse
    args = parse_arguments(argparse)

    if args.algorithm == "PPO":
        ppo(args)
    elif args.algorithm == "DQN":
        dqn(args)


   


if __name__ == '__main__':
    main()