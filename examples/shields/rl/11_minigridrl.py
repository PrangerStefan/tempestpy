from typing import Dict, Optional, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import stormpy
import stormpy.core
import stormpy.simulator
from datetime import datetime

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files
import os

import gymnasium as gym

import minigrid
import numpy as np

import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.utils.test_utils import check_learning_achieved, framework_iterator
from ray import tune, air
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.models.preprocessors import get_preprocessor
from MaskModels import TorchActionMaskModel
from Wrapper import OneHotWrapper, MiniGridEnvWrapper

import matplotlib.pyplot as plt

import argparse



class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode started Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        episode.user_data["count"] = 0
        # print(env.printGrid())
        # print(env.action_space.n)
        # print(env.actions)
        # print(env.mission)
        # print(env.observation_space)
        # img = env.get_frame()
        # plt.imshow(img)
        # plt.show()
       
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
         episode.user_data["count"] = episode.user_data["count"] + 1
         env = base_env.get_sub_environments()[0]
         #print(env.env.env.printGrid())
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode end Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        # print(env.env.env.printGrid())
        # print(episode.user_data["count"])
        
    
       

def parse_arguments(argparse):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="gym environment to load", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--env", 
                        help="gym environment to load", 
                        default="MiniGrid-LavaCrossingS9N1-v0", 
                        choices=[
                                "MiniGrid-LavaCrossingS9N1-v0",
                                "MiniGrid-DoorKey-8x8-v0", 
                                "MiniGrid-Dynamic-Obstacles-8x8-v0",
                                "MiniGrid-Empty-Random-6x6-v0",
                                "MiniGrid-Fetch-6x6-N2-v0", 
                                "MiniGrid-FourRooms-v0", 
                                "MiniGrid-KeyCorridorS6R3-v0", 
                                "MiniGrid-GoToDoor-8x8-v0",
                                "MiniGrid-LavaGapS7-v0",
                                "MiniGrid-SimpleCrossingS9N3-v0",
                                "MiniGrid-BlockedUnlockPickup-v0",
                                "MiniGrid-LockedRoom-v0",
                                "MiniGrid-ObstructedMaze-1Dlh-v0",
                                "MiniGrid-DoorKey-16x16-v0",
                                "MiniGrid-RedBlueDoors-6x6-v0",])
    
   # parser.add_argument("--seed", type=int, help="seed for environment", default=None)
    parser.add_argument("--grid_path", default="Grid.txt")
    parser.add_argument("--prism_path", default="Grid.PRISM")
    parser.add_argument("--no_masking", default=False)
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--log_dir", default="../log_results/")
    
    args = parser.parse_args()
    
    return args


def env_creater_custom(config):
    framestack = config.get("framestack", 4)
    shield = config.get("shield", {})
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    
    env = gym.make(name)
    env = MiniGridEnvWrapper(env, shield=shield)
    # env = minigrid.wrappers.ImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = OneHotWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
    
    return env

def env_creater_cart(config):
    return gym.make("CartPole-v1")

def env_creater(config):
    name = config.get("name", "MiniGrid-LavaCrossingS9N1-v0")
    framestack = config.get("framestack", 4)
    
    env = gym.make(name)
    env = minigrid.wrappers.ImgObsWrapper(env)
    env = OneHotWrapper(env,
                        config.vector_index if hasattr(config, "vector_index") else 0,
                        framestack=framestack
                        )
      
    print(F"Created Minigrid Environment is {env}")

    return env



def create_shield(grid_file, prism_path):
    os.system(F"/home/tknoll/Documents/main -v 'agent' -i {grid_file} -o {prism_path}")
    
    f = open(prism_path, "a")
    f.write("label \"AgentIsInLava\" = AgentIsInLava;")
    f.close()
    
    
    program = stormpy.parse_prism_program(prism_path)
    formula_str = "Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"
    #formula_str = "Pmax=? [G \"AgentIsInGoalAndNotDone\"]"
    
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)
    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
    
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.1) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    
    assert result.has_scheduler
    assert result.has_shield
    shield = result.shield
    
    action_dictionary = {}
    shield_scheduler = shield.construct()
    
    for stateID in model.states:
        choice = shield_scheduler.get_choice(stateID)
        choices = choice.choice_map
        state_valuation = model.state_valuations.get_string(stateID)

        actions_to_be_executed = [(choice[1] ,model.choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1]))) for choice in choices]

        action_dictionary[state_valuation] = actions_to_be_executed

    stormpy.shields.export_shield(model, shield, "Grid.shield")
    return action_dictionary

def export_grid_to_text(env, grid_file):
    f = open(grid_file, "w")
    # print(env)
    f.write(env.printGrid(init=True))
    # f.write(env.pprint_grid())
    f.close()

def create_environment(args):
    env_id= args.env
    env = gym.make(env_id)
    env.reset()
    return env


def register_custom_minigrid_env(args):
    env_name = "mini-grid"
    register_env(env_name, env_creater_custom)

    ModelCatalog.register_custom_model(
        "pa_model", 
        TorchActionMaskModel
    )
        
def create_shield_dict(args):
    env = create_environment(args)
    # print(env.pprint_grid())
    # print(env.printGrid(init=False))
    
    grid_file = args.grid_path
    export_grid_to_text(env, grid_file)
    
    prism_path = args.prism_path
    shield_dict = create_shield(grid_file, prism_path)
    #shield_dict = {state.id : shield.get_choice(state).choice_map for state in model.states}
   
    #print(F"Shield dictionary {shield_dict}")
    # for state_id in model.states:
    #     choices = shield.get_choice(state_id)
    #     print(F"Allowed choices in state {state_id}, are {choices.choice_map} ")
    
    return shield_dict

def ppo(args):
    
    ray.init(num_cpus=3)

    
    register_custom_minigrid_env(args)
    shield_dict = create_shield_dict(args)
    
    config = (PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .environment(env="mini-grid", env_config={"shield": shield_dict})
        .framework("torch")       
        .callbacks(MyCallbacks)
        .rl_module(_enable_rl_module_api = False)
        .debugging(logger_config={
            "type": "ray.tune.logger.TBXLogger", 
            "logdir": F"{args.log_dir}{datetime.now()}-{args.algorithm}"
        })
        .training(_enable_learner_api=False ,model={
            "custom_model": "pa_model",
            "custom_model_config" : {"shield": shield_dict, "no_masking": args.no_masking}            
        }))
    
    algo =(
        
        config.build()
    )
    
    # while not terminated and not truncated:
    #     action = algo.compute_single_action(obs)
    #     obs, reward, terminated, truncated = env.step(action)
    
    for i in range(30):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            
    ray.shutdown()


def dqn(args):
    register_custom_minigrid_env(args)
    shield_dict = create_shield_dict(args)

    
    config = DQNConfig()
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)
    config = config.environment(env="mini-grid", env_config={"shield": shield_dict })
    config = config.framework("torch")
    config = config.callbacks(MyCallbacks)
    config = config.rl_module(_enable_rl_module_api = False)
    config = config.debugging(logger_config={
            "type": "ray.tune.logger.TBXLogger", 
            "logdir": F"{args.log_dir}{datetime.now()}-{args.algorithm}"
        })
    config = config.training(hiddens=[], dueling=False, model={    
            "custom_model": "pa_model",
            "custom_model_config" : {"shield": shield_dict, "no_masking": args.no_masking}
    })
    
    algo = (
        config.build()
    )
         
    for i in range(30):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            
    ray.shutdown()


def main():
    args = parse_arguments(argparse)

    if args.algorithm == "ppo":
        ppo(args)
    elif args.algorithm == "dqn":
        dqn(args)


   


if __name__ == '__main__':
    main()