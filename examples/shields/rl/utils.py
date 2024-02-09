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
import tempfile, datetime, shutil

import gymnasium as gym

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

def shield_needed(shielding):
    return shielding in [ShieldingConfig.Training, ShieldingConfig.Evaluation, ShieldingConfig.Full]

def shielded_evaluation(shielding):
    return shielding in [ShieldingConfig.Evaluation, ShieldingConfig.Full]

def shielded_training(shielding):
    return shielding in [ShieldingConfig.Training, ShieldingConfig.Full]

class ShieldHandler(ABC):
    def __init__(self) -> None:
        pass
    def create_shield(self, **kwargs) -> dict:
        pass

class MiniGridShieldHandler(ShieldHandler):
    def __init__(self, grid_to_prism_binary, grid_file, prism_path, formula, prism_config=None, shield_value=0.9, shield_comparison='absolute', nocleanup=False) -> None:
        self.tmp_dir_name = f"shielding_files_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}_{next(tempfile._get_candidate_names())}"
        os.mkdir(self.tmp_dir_name)
        self.grid_file = self.tmp_dir_name + "/" + grid_file
        self.grid_to_prism_binary = grid_to_prism_binary
        self.prism_path = self.tmp_dir_name + "/" + prism_path
        self.prism_config = prism_config

        self.formula = formula
        shield_comparison = stormpy.logic.ShieldComparison.ABSOLUTE if shield_comparison == "absolute" else stormpy.logic.ShieldComparison.RELATIVE
        self.shield_expression = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, shield_comparison, shield_value)

        self.nocleanup = nocleanup
    def __del__(self):
        if not self.nocleanup:
            shutil.rmtree(self.tmp_dir_name)

    def __export_grid_to_text(self, env):
        with open(self.grid_file, "w") as f:
            f.write(env.printGrid(init=True))


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

        #stormpy.shields.export_shield(model, shield, self.tmp_dir_name + "/current.shield")

        print(f"LOG: Starting to translate shield...")
        tic()
        for stateID in model.states:
            choice = shield_scheduler.get_choice(stateID)
            choices = choice.choice_map
            state_valuation = state_valuations.get_string(stateID)
            ints = dict(re.findall(r'([a-zA-Z][_a-zA-Z0-9]+)=(-?[a-zA-Z0-9]+)', state_valuation))
            booleans = re.findall(r'(\!?)([a-zA-Z][_a-zA-Z0-9]+)[\s\t]+', state_valuation)
            booleans = {b[1]: False if b[0] == "!" else True for b in booleans}
            if int(ints.get("previousActionAgent", 3)) != 3:
                continue
            if int(ints.get("clock", 0)) != 0:
                continue
            state = to_state(ints, booleans)
            action_dictionary[state] = get_allowed_actions_mask([choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1])) for choice in choices])

        toc()
        return action_dictionary


    def create_shield(self, **kwargs):
        env = kwargs["env"]
        self.__export_grid_to_text(env)
        self.__create_prism()

        return self.__create_shield_dict()

def expname(args):
    return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}_{args.env}_{args.shielding}_{args.shield_comparison}_{args.shield_value}_{args.expname_suffix}"

def create_log_dir(args):
    log_dir = f"{args.log_dir}/{expname(args)}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_allowed_actions_mask(actions):
    action_mask = [0.0] * 7
    actions_labels = [label for labels in actions for label in list(labels)]
    for action_label in actions_labels:
        if "move" in action_label:
            action_mask[2] = 1.0
        elif "left" in action_label:
            action_mask[0] = 1.0
        elif "right" in action_label:
            action_mask[1] = 1.0
        elif "pickup" in action_label:
            action_mask[3] = 1.0
        elif "drop" in action_label:
            action_mask[4] = 1.0
        elif "toggle" in action_label:
            action_mask[5] = 1.0
        elif "done" in action_label:
            action_mask[6] = 1.0
    return action_mask

def common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="gym environment to load",
                        choices=gym.envs.registry.keys(),
                        default="MiniGrid-LavaSlipperyCliff-16x13-v0")

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
    parser.add_argument("--nocleanup", action=argparse.BooleanOptionalAction)
    parser.add_argument("--expname_suffix", default="")
    return parser

class MiniWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.transpose(1,0,2), info

    def observations(self, obs):
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.transpose(1,0,2), reward, terminated, truncated, info
