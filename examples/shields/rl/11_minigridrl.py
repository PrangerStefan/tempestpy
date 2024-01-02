import gymnasium as gym
import minigrid

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog


from torch_action_mask_model import TorchActionMaskModel
from wrappers import OneHotShieldingWrapper, MiniGridShieldingWrapper
from helpers import parse_arguments, create_log_dir, ShieldingConfig
from shieldhandlers import MiniGridShieldHandler, create_shield_query
from callbacks import MyCallbacks

from ray.tune.logger import TBXLogger   

def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.grid_path}_{config.worker_index}_{args.prism_config}.txt"
    args.prism_path = F"{args.prism_path}_{config.worker_index}_{args.prism_config}.prism"
    
    prob_forward = args.prob_forward
    prob_direct = args.prob_direct
    prob_next = args.prob_next

    shield_creator = MiniGridShieldHandler(args.grid_path, 
                                            args.grid_to_prism_binary_path,
                                            args.prism_path, 
                                            args.formula,
                                            args.shield_value,
                                            args.prism_config,
                                            shield_comparision=args.shield_comparision)

    env = gym.make(name, randomize_start=True,probability_forward=prob_forward, probability_direct_neighbour=prob_direct, probability_next_neighbour=prob_next)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, 
                                   shield_query_creator=create_shield_query,
                                   mask_actions=args.shielding != ShieldingConfig.Disabled,
                                   create_shield_at_reset=args.shield_creation_at_reset)
    # env = minigrid.wrappers.ImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = OneHotShieldingWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
    
    
    return env



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
        .callbacks(MyCallbacks)
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
    config = config.callbacks(MyCallbacks)
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