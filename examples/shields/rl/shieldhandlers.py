import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

from abc import ABC

import os

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
        result = os.system(F"{self.grid_to_prism_path} -v 'agent' -i {self.grid_file} -o {self.prism_path}")
    
        assert result == 0, "Prism file could not be generated"
    
        f = open(self.prism_path, "a")
        f.write("label \"AgentIsInLava\" = AgentIsInLava;")
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
        
        for stateID in model.states:
            choice = shield_scheduler.get_choice(stateID)
            choices = choice.choice_map
            state_valuation = model.state_valuations.get_string(stateID)

            #actions_to_be_executed = [(choice[1] ,model.choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1]))) for choice in choices]
            actions_to_be_executed = [Action(idx= choice[1], prob=choice[0], labels=model.choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1]))) for choice in choices]

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

    key_text = ""

    # only support one key for now
   
    #print(F"Agent pos is {self.env.agent_pos} and direction {self.env.agent_dir} ")
    cur_pos_str = f"[{key_text}!AgentDone\t& xAgent={coordinates[0]}\t& yAgent={coordinates[1]}\t& viewAgent={view_direction}]"

    return cur_pos_str
    