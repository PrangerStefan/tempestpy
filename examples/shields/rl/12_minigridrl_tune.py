import gymnasium as gym
import minigrid

from ray import tune, air
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog


from torch_action_mask_model import TorchActionMaskModel
from wrappers import OneHotShieldingWrapper, MiniGridShieldingWrapper
from helpers import parse_arguments, create_log_dir, ShieldingConfig
from shieldhandlers import MiniGridShieldHandler, create_shield_query
from callbacks import MyCallbacks

from torch.utils.tensorboard import SummaryWriter
from ray.tune.logger import TBXLogger, UnifiedLogger, CSVLogger

def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.grid_path}_{config.worker_index}.txt"
    args.prism_path = F"{args.prism_path}_{config.worker_index}.prism"
    
    shield_creator = MiniGridShieldHandler(args.grid_path, args.grid_to_prism_binary_path, args.prism_path, args.formula)
    
    env = gym.make(name)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query)
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
        .training(_enable_learner_api=False ,model={
            "custom_model": "shielding_model"
        }))
    
    return config
    
            

def dqn(args):
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
    config = config.training(hiddens=[], dueling=False, model={    
            "custom_model": "shielding_model"
    })
    
    return config
            

def main():
    import argparse
    args = parse_arguments(argparse)

    if args.algorithm == "PPO":
        config = ppo(args)
    elif args.algorithm == "DQN":
        config = dqn(args)
        
    logdir = create_log_dir(args)
        
    tuner = tune.Tuner(args.algorithm,
                        tune_config=tune.TuneConfig(
                            metric="episode_reward_mean",
                            mode="max",
                            num_samples=1,
                            
                        ),
                        run_config=air.RunConfig(
                                stop = {"episode_reward_mean": 94,
                                        "timesteps_total": 12000,
                                        "training_iteration": args.iterations}, 
                                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True, num_to_keep=2 ),
                                storage_path=F"{logdir}"
                        ),
                        param_space=config,
                    )

    tuner.fit()
 
   


if __name__ == '__main__':
    main()