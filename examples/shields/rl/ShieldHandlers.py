import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

from abc import ABC

import os

class ShieldHandler(ABC):
    def __init__(self) -> None:
        pass
    def create_shield(self, **kwargs):
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
        os.system(F"{self.grid_to_prism_path} -v 'agent' -i {self.grid_file} -o {self.prism_path}")
    
        f = open(self.prism_path, "a")
        f.write("label \"AgentIsInLava\" = AgentIsInLava;")
        f.close()
        
    def __create_shield_dict(self):
        program = stormpy.parse_prism_program(self.prism_path)
        shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.1) 
        
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
    
    
    def create_shield(self, **kwargs):
        env = kwargs["env"]
        self.__export_grid_to_text(env)
        self.__create_prism()
       
        return self.__create_shield_dict()
            