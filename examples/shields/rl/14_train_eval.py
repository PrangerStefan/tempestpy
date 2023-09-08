import gymnasium as gym
import minigrid

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print, UnifiedLogger, CSVLogger
from ray.rllib.models import ModelCatalog


from torch_action_mask_model import TorchActionMaskModel
from wrappers import OneHotShieldingWrapper, MiniGridShieldingWrapper
from helpers import parse_arguments, create_log_dir, ShieldingConfig
from shieldhandlers import MiniGridShieldHandler, create_shield_query

from callbacks import MyCallbacks

from torch.utils.tensorboard import SummaryWriter


  

def shielding_env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    args = config.get("args", None)
    args.grid_path = F"{args.grid_path}_{config.worker_index}.txt"
    args.prism_path = F"{args.prism_path}_{config.worker_index}.prism"
    
    shielding = config.get("shielding", False)
    shield_creator = MiniGridShieldHandler(args.grid_path, args.grid_to_prism_binary_path, args.prism_path, args.formula)
    
    env = gym.make(name)
    env = MiniGridShieldingWrapper(env, shield_creator=shield_creator, shield_query_creator=create_shield_query ,mask_actions=shielding)

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
        .environment( env="mini-grid-shielding",
                      env_config={"name": args.env, "args": args, "shielding": args.shielding is ShieldingConfig.Full or args.shielding is ShieldingConfig.Training})
        .framework("torch")
        .callbacks(MyCallbacks)
        .evaluation(evaluation_config={ 
                                       "evaluation_interval": 1,
                                        "evaluation_duration": 10,
                                        "evaluation_num_workers":1,
                                        "env": "mini-grid-shielding", 
                                        "env_config": {"name": args.env, "args": args, "shielding": args.shielding is ShieldingConfig.Full or args.shielding is ShieldingConfig.Evaluation}})        
        .rl_module(_enable_rl_module_api = False)
        .debugging(logger_config={
            "type": UnifiedLogger, 
            "logdir": create_log_dir(args)
        })
        .training(_enable_learner_api=False ,model={
            "custom_model": "shielding_model"      
        }))
    
    algo =(
        
        config.build()
    )
    
    iterations = args.iterations
    
    
    
    for i in range(iterations):
        algo.train()
    
        if i % 5 == 0:
            algo.save()
        
    eval_log_dir = F"{create_log_dir(args)}-eval"
        
    writer = SummaryWriter(log_dir=eval_log_dir)
    csv_logger = CSVLogger(config=config, logdir=eval_log_dir)
    
    for i in range(iterations):
        eval_result = algo.evaluate()
        print(pretty_print(eval_result))
        print(eval_result)
        # logger.on_result(eval_result)

        csv_logger.on_result(eval_result)
        
        evaluation = eval_result['evaluation']
        epsiode_reward_mean = evaluation['episode_reward_mean']
        episode_len_mean = evaluation['episode_len_mean']
        print(epsiode_reward_mean)
        writer.add_scalar("evaluation/episode_reward_mean", epsiode_reward_mean, i)
        writer.add_scalar("evaluation/episode_len_mean", episode_len_mean, i)

        
        
    writer.close()


def main():
    import argparse
    args = parse_arguments(argparse)

    ppo(args)
   


if __name__ == '__main__':
    main()