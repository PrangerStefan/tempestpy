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
    #print(env.grid)
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            obj = env.grid.get(i,j)
            
            if obj and obj.type == "key":
                keys.append(obj.color)
    
    return keys

def create_log_dir(args):
    return F"{args.log_dir}{args.algorithm}-shielding:{args.shielding}-iterations:{args.iterations}"


def get_action_index_mapping(actions):
    for action_str in actions:
        if "left" in action_str:
            return Actions.left
        elif "right" in action_str:
            return Actions.right
        elif "east" in action_str: 
            return Actions.forward
        elif "south" in action_str:
            return Actions.forward
        elif "west" in action_str:
            return Actions.forward
        elif "north" in action_str:
            return Actions.forward
        elif "pickup" in action_str:
            return Actions.pickup
        elif "done" in action_str:
            return Actions.done
    
    
    raise ValueError(F"Action string {action_str} not supported")



def parse_arguments(argparse):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="gym environment to load", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--env", 
                        help="gym environment to load", 
                        default="MiniGrid-LavaCrossingS9N1-v0", 
                        choices=[
                                "MiniGrid-LavaCrossingS9N1-v0",
                                "MiniGrid-LavaCrossingS9N3-v0",
                                # "MiniGrid-DoorKey-8x8-v0", 
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
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--log_dir", default="../log_results/")
    parser.add_argument("--iterations", type=int, default=30 )
    parser.add_argument("--formula", default="Pmax=? [G !\"AgentIsInLavaAndNotDone\"]")  # formula_str = "Pmax=? [G ! \"AgentIsInGoalAndNotDone\"]"
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shielding", type=ShieldingConfig, choices=list(ShieldingConfig), default=ShieldingConfig.Full)

    
    args = parser.parse_args()
    
    return args
