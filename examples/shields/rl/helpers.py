import minigrid
from minigrid.core.actions import Actions

from datetime import datetime
from enum import Enum

import os

import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

class ShieldingConfig(Enum):
    Training = 'training'
    Evaluation = 'evaluation'
    Disabled = 'none'
    Full = 'full'
    
    def __str__(self) -> str:
        return self.value


def extract_keys(env):
    keys = []
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            obj = env.grid.get(i,j)
            
            if obj and obj.type == "key":
                keys.append((obj, i, j))
    
    if env.carrying and env.carrying.type == "key":
        keys.append((env.carrying, -1, -1))
    
    return keys

def extract_doors(env):
    doors = []
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            obj = env.grid.get(i,j)
            
            if obj and obj.type == "door":
                doors.append(obj)
                
    return doors

def create_log_dir(args):
    return F"{args.log_dir}sh:{args.shielding}-env:{args.env}"

def test_name(args):
    return F"{args.expname}"

def get_action_index_mapping(actions):
    for action_str in actions:
        if "move" in action_str:
            return Actions.forward
        elif "left" in action_str:
            return Actions.left
        elif "right" in action_str:
            return Actions.right
        elif "pickup" in action_str:
            return Actions.pickup
        elif "done" in action_str:
            return Actions.done    
        elif "drop" in action_str:
            return Actions.drop
        elif "toggle" in action_str:
            return Actions.toggle
        elif "unlock" in action_str:
            return Actions.toggle
    
    return Actions.done



def parse_arguments(argparse):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="gym environment to load", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--env", 
                        help="gym environment to load", 
                        default="MiniGrid-LavaCrossingS9N1-v0", 
                        choices=[
                                "MiniGrid-LavaCrossingS9N1-v0",
                                "MiniGrid-LavaCrossingS9N3-v0",
                                "MiniGrid-LavaSlipperyS12-v0",
                                "MiniGrid-LavaSlipperyS12-v1",
                                "MiniGrid-LavaSlipperyS12-v2",
                                "MiniGrid-LavaSlipperyS12-v3",
                                "MiniGrid-DoorKey-8x8-v0",
                                "MiniGrid-DoubleDoor-16x16-v0",
                                "MiniGrid-DoubleDoor-12x12-v0",
                                "MiniGrid-DoubleDoor-10x8-v0",
                                "MiniGrid-SingleDoor-7x6-v0",
                                # "MiniGrid-LockedRoom-v0",
                                # "MiniGrid-FourRooms-v0", 
                                # "MiniGrid-LavaGapS7-v0",
                                # "MiniGrid-SimpleCrossingS9N3-v0",
                                # "MiniGrid-DoorKey-16x16-v0",
                                # "MiniGrid-Empty-Random-6x6-v0",    
                                ])
    
   # parser.add_argument("--seed", type=int, help="seed for environment", default=None)
    parser.add_argument("--grid_to_prism_binary_path", default="./main")
    parser.add_argument("--grid_path", default="grid")
    parser.add_argument("--prism_path", default="grid")
    parser.add_argument("--algorithm", default="PPO", type=str.upper , choices=["PPO", "DQN"])
    parser.add_argument("--log_dir", default="../log_results/")
    parser.add_argument("--evaluations", type=int, default=10 )
    parser.add_argument("--formula", default="Pmax=? [G !\"AgentIsInLavaAndNotDone\"]")  # formula_str = "Pmax=? [G ! \"AgentIsInGoalAndNotDone\"]"
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shielding", type=ShieldingConfig, choices=list(ShieldingConfig), default=ShieldingConfig.Full)
    parser.add_argument("--steps", default=20_000, type=int)
    parser.add_argument("--expname", default="exp")
    parser.add_argument("--shield_creation_at_reset", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    return args
