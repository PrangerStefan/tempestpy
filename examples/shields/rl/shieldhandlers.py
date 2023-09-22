import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files


from helpers import extract_doors, extract_keys, extract_adversaries
from abc import ABC

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
    def __init__(self, grid_file, grid_to_prism_path, prism_path, formula) -> None:
        self.grid_file = grid_file
        self.grid_to_prism_path = grid_to_prism_path
        self.prism_path = prism_path
        self.formula = formula
    
    def __export_grid_to_text(self, env):
        f = open(self.grid_file, "w")
        f.write(env.printGrid(init=True))
        f.close()

    
    def __create_prism(self):
        result = os.system(F"{self.grid_to_prism_path} -v 'Agent,Blue' -i {self.grid_file} -o {self.prism_path}")
    
        assert result == 0, "Prism file could not be generated"
    
        f = open(self.prism_path, "a")
        f.write("label \"AgentIsInLava\" = AgentIsInLava;")
        f.write("label \"AgentIsInGoal\" = AgentIsInGoal;")
        f.close()
        
    def __create_shield_dict(self):
        print(self.prism_path)
        program = stormpy.parse_prism_program(self.prism_path)
        shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.9) 
        
        formulas = stormpy.parse_properties_for_prism_program(self.formula, program)
        options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
        options.set_build_state_valuations(True)
        options.set_build_choice_labels(True)
        options.set_build_all_labels()
        model = stormpy.build_sparse_model_with_options(program, options)
        
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
        
        assert result.has_scheduler
        assert result.has_shield
        shield = result.shield
        stormpy.shields.export_shield(model, shield, "Grid.shield")
        action_dictionary = {}
        shield_scheduler = shield.construct()
        state_valuations = model.state_valuations
        choice_labeling = model.choice_labeling
        
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
    
    agent_position = F"xAgent={coordinates[0]}\t& yAgent={coordinates[1]}\t& viewAgent={view_direction}"    
    query = f"[{agent_carrying}& {''.join(agent_key_status)}!AgentDone\t{adv_status_text}{door_status_text}{agent_position}{adv_positions_text}{key_positions_text}]"

    return query
    