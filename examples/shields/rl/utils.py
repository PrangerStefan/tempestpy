import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

from enum import Enum
from abc import ABC

import re
import sys


from minigrid.core.actions import Actions
from minigrid.core.state import to_state

import os
import time

import argparse

def tic():
    #Homemade version of matlab tic and toc functions: https://stackoverflow.com/a/18903019
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

class ShieldingConfig(Enum):
    Training = 'training'
    Evaluation = 'evaluation'
    Disabled = 'none'
    Full = 'full'

    def __str__(self) -> str:
        return self.value

class ShieldHandler(ABC):
    def __init__(self) -> None:
        pass
    def create_shield(self, **kwargs) -> dict:
        pass

class MiniGridShieldHandler(ShieldHandler):
    def __init__(self, grid_to_prism_binary, grid_file, prism_path, formula, prism_config=None, shield_value=0.9, shield_comparison='absolute') -> None:
        self.grid_file = grid_file
        self.grid_to_prism_binary = grid_to_prism_binary
        self.prism_path = prism_path
        self.prism_config = prism_config

        self.formula = formula
        shield_comparison = stormpy.logic.ShieldComparison.ABSOLUTE if shield_comparison == "absolute" else stormpy.logic.ShieldComparison.RELATIVE
        self.shield_expression = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, shield_comparison, shield_value)


    def __export_grid_to_text(self, env):
        f = open(self.grid_file, "w")
        f.write(env.printGrid(init=True))
        f.close()


    def __create_prism(self):
        if self.prism_config is None:
            result = os.system(F"{self.grid_to_prism_binary} -i {self.grid_file} -o {self.prism_path}")
        else:
            result = os.system(F"{self.grid_to_prism_binary} -i {self.grid_file} -o {self.prism_path} -c {self.prism_config}")

        assert result == 0, "Prism file could not be generated"

    def __create_shield_dict(self):
        program = stormpy.parse_prism_program(self.prism_path)

        formulas = stormpy.parse_properties_for_prism_program(self.formula, program)
        options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
        options.set_build_state_valuations(True)
        options.set_build_choice_labels(True)
        options.set_build_all_labels()
        print(f"LOG: Starting with explicit model creation...")
        tic()
        model = stormpy.build_sparse_model_with_options(program, options)
        toc()

        print(f"LOG: Starting with model checking...")
        tic()
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=self.shield_expression)
        toc()

        assert result.has_shield
        shield = result.shield
        action_dictionary = dict()
        shield_scheduler = shield.construct()
        state_valuations = model.state_valuations
        choice_labeling = model.choice_labeling

        #stormpy.shields.export_shield(model, shield, "current.shield")

        for stateID in model.states:
            choice = shield_scheduler.get_choice(stateID)
            choices = choice.choice_map
            state_valuation = state_valuations.get_string(stateID)
            ints = dict(re.findall(r'([a-zA-Z][_a-zA-Z0-9]+)=([a-zA-Z0-9]+)', state_valuation))
            booleans = dict(re.findall(r'(\!?)([a-zA-Z][_a-zA-Z0-9]+)[\s\t]', state_valuation)) #TODO does not parse everything correctly?

            if int(ints.get("previousActionAgent", 3)) != 3:
                continue
            if int(ints.get("clock", 0)) != 0:
                continue
            state = to_state(ints, booleans)
            action_dictionary[state] = get_allowed_actions_mask([choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1])) for choice in choices])

        return action_dictionary


    def create_shield(self, **kwargs):
        env = kwargs["env"]
        self.__export_grid_to_text(env)
        self.__create_prism()

        return self.__create_shield_dict()


def create_log_dir(args):
    return F"{args.log_dir}sh:{args.shielding}-value:{args.shield_value}-comp:{args.shield_comparison}-env:{args.env}-conf:{args.prism_config}"

def test_name(args):
    return F"{args.expname}"

def get_allowed_actions_mask(actions):
    action_mask = [0.0] * 3 + [1.0] * 4
    actions_labels = [label for labels in actions for label in list(labels)]
    for action_label in actions_labels:
        if "move" in action_label:
            action_mask[2] = 1.0
        elif "left" in action_label:
            action_mask[0] = 1.0
        elif "right" in action_label:
            action_mask[1] = 1.0
    return action_mask

def common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="gym environment to load",
                        default="MiniGrid-LavaSlipperyCliff-16x12-v0")

    parser.add_argument("--grid_file", default="grid.txt")
    parser.add_argument("--prism_output_file", default="grid.prism")
    parser.add_argument("--log_dir", default="../log_results/")
    parser.add_argument("--formula", default="Pmax=? [G !AgentIsOnLava]")
    parser.add_argument("--shielding", type=ShieldingConfig, choices=list(ShieldingConfig), default=ShieldingConfig.Full)
    parser.add_argument("--steps", default=20_000, type=int)
    parser.add_argument("--shield_creation_at_reset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prism_config",  default=None)
    parser.add_argument("--shield_value", default=0.9, type=float)
    parser.add_argument("--shield_comparison", default='absolute', choices=['relative', 'absolute'])
    return parser
