import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

from enum import Enum
from abc import ABC


from minigrid.core.actions import Actions

import os
import time
class Action():
    def __init__(self, idx, prob=1, labels=[]) -> None:
        self.idx = idx
        self.prob = prob
        self.labels = labels

class ShieldHandler(ABC):
    def __init__(self) -> None:
        pass
    def create_shield(self, **kwargs) -> dict:
        pass

class MiniGridShieldHandler(ShieldHandler):
    def __init__(self, grid_file, grid_to_prism_path, prism_path, formula, shield_value=0.9 ,prism_config=None, shield_comparision='relative') -> None:
        self.grid_file = grid_file
        self.grid_to_prism_path = grid_to_prism_path
        self.prism_path = prism_path
        self.formula = formula
        self.prism_config = prism_config
        self.shield_value = shield_value
        self.shield_comparision = shield_comparision
    
    def __export_grid_to_text(self, env):
        f = open(self.grid_file, "w")
        f.write(env.printGrid(init=True))
        f.close()

    
    def __create_prism(self):
        # result = os.system(F"{self.grid_to_prism_path} -v 'Agent,Blue' -i {self.grid_file} -o {self.prism_path} -c adv_config.yaml")
        if self.prism_config is None:
            result = os.system(F"{self.grid_to_prism_path} -v 'Agent' -i {self.grid_file} -o {self.prism_path}")
        else:
            result = os.system(F"{self.grid_to_prism_path} -v 'Agent' -i {self.grid_file} -o {self.prism_path} -c {self.prism_config}")
        # result = os.system(F"{self.grid_to_prism_path} -v 'Agent' -i {self.grid_file} -o {self.prism_path} -c adv_config.yaml")

        assert result == 0, "Prism file could not be generated"

        f = open(self.prism_path, "a")
        f.close()

    def __create_shield_dict(self):
        print(self.prism_path)
        program = stormpy.parse_prism_program(self.prism_path)

        shield_comp = stormpy.logic.ShieldComparison.RELATIVE

        if self.shield_comparision == 'absolute':
            shield_comp = stormpy.logic.ShieldComparison.ABSOLUTE

        shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, shield_comp, self.shield_value) 

        
        formulas = stormpy.parse_properties_for_prism_program(self.formula, program)
        options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
        options.set_build_state_valuations(True)
        options.set_build_choice_labels(True)
        options.set_build_all_labels()
        model = stormpy.build_sparse_model_with_options(program, options)
        
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
        
        assert result.has_shield
        shield = result.shield
        action_dictionary = {}
        shield_scheduler = shield.construct()
        state_valuations = model.state_valuations
        choice_labeling = model.choice_labeling
        stormpy.shields.export_shield(model, shield, "myshield")
        
        for stateID in model.states:
            choice = shield_scheduler.get_choice(stateID)
            choices = choice.choice_map
            state_valuation = state_valuations.get_string(stateID)
            actions_to_be_executed = [Action(idx= choice[1], prob=choice[0], labels=choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1]))) for choice in choices]
            action_dictionary[state_valuation] = actions_to_be_executed
            
        return action_dictionary
    
    
    def create_shield(self, **kwargs):
        env = kwargs["env"]
        self.__export_grid_to_text(env)
        self.__create_prism()
       
        return self.__create_shield_dict()
        
def create_shield_query(env):
    coordinates = env.env.agent_pos
    view_direction = env.env.agent_dir
        
    keys = extract_keys(env)
    doors = extract_doors(env)
    adversaries = extract_adversaries(env)
    
    
    if env.carrying:
        agent_carrying = F"Agent_is_carrying_object\t"
    else:
        agent_carrying = "!Agent_is_carrying_object\t"
    
    key_positions = []
    agent_key_status = []
    
    for key in keys:    
        key_color = key[0].color
        key_x = key[1]
        key_y = key[2]
        if env.carrying and env.carrying.type == "key":
            agent_key_text = F"Agent_has_{env.carrying.color}_key\t& "
            key_position = F"xKey{key_color}={key_x}\t& yKey{key_color}={key_y}\t"
        else:
            agent_key_text = F"!Agent_has_{key_color}_key\t& "
            key_position = F"xKey{key_color}={key_x}\t& yKey{key_color}={key_y}\t"
        
        key_positions.append(key_position)            
        agent_key_status.append(agent_key_text)
    
    if key_positions:
        key_positions[-1] = key_positions[-1].strip()
    
    door_status = []
    for door in doors:
        status = ""
        if door.is_open:
            status = F"!Door{door.color}locked\t& Door{door.color}open\t&"
        elif door.is_locked:
            status = F"Door{door.color}locked\t& !Door{door.color}open\t&"
        else:
            status = F"!Door{door.color}locked\t& !Door{door.color}open\t&"
            
        door_status.append(status)
    
    adv_status = []
    adv_positions = []
    
    for adversary in adversaries:
        status = ""
        position = ""
        
        if adversary.carrying:
            carrying = F"{adversary.name}_is_carrying_object\t"
        else:
            carrying = F"!{adversary.name}_is_carrying_object\t"
            
        status = F"{carrying}& !{adversary.name}Done\t& "
        position = F"x{adversary.name}={adversary.cur_pos[1]}\t& y{adversary.name}={adversary.cur_pos[0]}\t& view{adversary.name}={adversary.adversary_dir}"
        adv_status.append(status)
        adv_positions.append(position)

    door_status_text = ""

    if door_status:
        door_status_text = F"& {''.join(door_status)}\t"
    
    adv_status_text = ""
    
    if adv_status:    
        adv_status_text = F"& {''.join(adv_status)}"
        
    adv_positions_text = ""
    
    if adv_positions:
        adv_positions_text = F"\t& {''.join(adv_positions)}"
        
    key_positions_text = ""
    
    if key_positions:
        key_positions_text = F"\t& {''.join(key_positions)}"
    
    move_text = ""
    
    if adversaries:
        move_text = F"move=0\t& "
    
    agent_position = F"& xAgent={coordinates[0]}\t& yAgent={coordinates[1]}\t& viewAgent={view_direction}"    
    query = f"[{agent_carrying}& {''.join(agent_key_status)}!AgentDone\t{adv_status_text}{move_text}{door_status_text}{agent_position}{adv_positions_text}{key_positions_text}]"

    return query
  

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
    # TODO Maybe need to add ordering of keys so it matches the order in the shield
    return keys

def extract_doors(env):
    doors = []
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            obj = env.grid.get(i,j)
            
            if obj and obj.type == "door":
                doors.append(obj)
                
    return doors

def extract_adversaries(env):
    adv = []
    
    if not hasattr(env, "adversaries") or not env.adversaries:
        return []
    
    for color, adversary in env.adversaries.items():
        adv.append(adversary)
    
    
    return adv

def create_log_dir(args):
    return F"{args.log_dir}sh:{args.shielding}-value:{args.shield_value}-comp:{args.shield_comparision}-env:{args.env}-conf:{args.prism_config}"

def test_name(args):
    return F"{args.expname}"

def get_action_index_mapping(actions):
    for action_str in actions:
        if not "Agent" in action_str:
            continue
        
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

    raise ValueError("No action mapping found")


def parse_arguments(argparse):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="gym environment to load", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--env",
                        help="gym environment to load",
                        default="MiniGrid-LavaSlipperyCliffS12-v2",
                        choices=[
                                "MiniGrid-Adv-8x8-v0",
                                "MiniGrid-AdvSimple-8x8-v0",
                                "MiniGrid-LavaCrossingS9N1-v0",
                                "MiniGrid-LavaCrossingS9N3-v0",
                                "MiniGrid-LavaSlipperyCliffS12-v0",
                                "MiniGrid-LavaFaultyS12-30-v0",
                                ])

   # parser.add_argument("--seed", type=int, help="seed for environment", default=None)
    parser.add_argument("--grid_to_prism_binary_path", default="./main")
    parser.add_argument("--grid_path", default="grid")
    parser.add_argument("--prism_path", default="grid")
    parser.add_argument("--algorithm", default="PPO", type=str.upper , choices=["PPO", "DQN"])
    parser.add_argument("--log_dir", default="../log_results/")
    parser.add_argument("--evaluations", type=int, default=30 )
    parser.add_argument("--formula", default="Pmax=? [G !\"AgentIsInLavaAndNotDone\"]")  # formula_str = "Pmax=? [G ! \"AgentIsInGoalAndNotDone\"]"
    # parser.add_argument("--formula", default="<<Agent>> Pmax=? [G <= 4 !\"AgentRanIntoAdversary\"]")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num_gpus", type=float, default=0)
    parser.add_argument("--shielding", type=ShieldingConfig, choices=list(ShieldingConfig), default=ShieldingConfig.Full)
    parser.add_argument("--steps", default=20_000, type=int)
    parser.add_argument("--expname", default="exp")
    parser.add_argument("--shield_creation_at_reset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prism_config",  default=None)
    parser.add_argument("--shield_value", default=0.9, type=float)
    parser.add_argument("--probability_displacement", default=1/4, type=float)
    parser.add_argument("--probability_intended", default=3/4, type=float)
    parser.add_argument("--probability_turn_displacement", default=0/4, type=float)
    parser.add_argument("--probability_turn_intended", default=4/4, type=float)
    parser.add_argument("--shield_comparision", default='relative', choices=['relative', 'absolute'])
    # parser.add_argument("--random_starts", default=1, type=int)
    args = parser.parse_args()
    
    return args
