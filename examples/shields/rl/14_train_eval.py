import gymnasium as gym
import minigrid

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print, UnifiedLogger, CSVLogger
from ray.rllib.models import ModelCatalog


from torch_action_mask_model import TorchActionMaskModel
from rllibutils import OneHotShieldingWrapper, MiniGridShieldingWrapper, shielding_env_creater
from utils import MiniGridShieldHandler, create_shield_query, parse_arguments, create_log_dir, ShieldingConfig

from callbacks import CustomCallback

from torch.utils.tensorboard import SummaryWriter

def register_minigrid_shielding_env(args):
    env_name = "mini-grid-shielding"
    register_env(env_name, shielding_env_creater)

    ModelCatalog.register_custom_model(
        "shielding_model", 
        TorchActionMaskModel
    )


def ppo(args):
    register_minigrid_shielding_env(args)
    train_batch_size = 4000
    config = (PPOConfig()
        .rollouts(num_rollout_workers=args.workers)
        .resources(num_gpus=0)
        .environment( env="mini-grid-shielding",
                      env_config={"name": args.env, "args": args, "shielding": args.shielding is ShieldingConfig.Full or args.shielding is ShieldingConfig.Training})
        .framework("torch")
        .callbacks(CustomCallback)
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
        }, train_batch_size=train_batch_size))
    
    algo =(
        
        config.build()
    )
    
    
    iterations = int((args.steps / train_batch_size)) + 1
    
    for i in range(iterations):
        algo.train()
    
        if i % 5 == 0:
            algo.save()
        
    eval_log_dir = F"{create_log_dir(args)}-eval"
        
    writer = SummaryWriter(log_dir=eval_log_dir)
    csv_logger = CSVLogger(config=config, logdir=eval_log_dir)
    
    for i in range(evaluations):
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