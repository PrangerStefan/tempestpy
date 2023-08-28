import minigrid
from minigrid.core.actions import Actions
import gymnasium as gym

import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

import os

   
def extract_keys(env):
    env.reset()
    keys = []
    print(env.grid)
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            obj = env.grid.get(i,j)
            
            if obj and obj.type == "key":
                keys.append(obj.color)
    
    return keys


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
    parser.add_argument("--grid_to_prism_path", default="./main")
    parser.add_argument("--grid_path", default="Grid.txt")
    parser.add_argument("--prism_path", default="Grid.PRISM")
    parser.add_argument("--no_masking", default=False)
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--log_dir", default="../log_results/")
    parser.add_argument("--iterations", type=int, default=30 )
    
    args = parser.parse_args()
    
    return args



def create_environment(args):
    env_id= args.env
    env = gym.make(env_id)
    env.reset()
    return env


def export_grid_to_text(env, grid_file):
    f = open(grid_file, "w")
    # print(env)
    f.write(env.printGrid(init=True))
    f.close()


def create_shield(grid_to_prism_path, grid_file, prism_path):
    os.system(F"{grid_to_prism_path} -v 'agent' -i {grid_file} -o {prism_path}")
    
    f = open(prism_path, "a")
    f.write("label \"AgentIsInLava\" = AgentIsInLava;")
    f.close()
    
    
    program = stormpy.parse_prism_program(prism_path)
    formula_str = "Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"
    # formula_str = "Pmax=? [G ! \"AgentIsInGoalAndNotDone\"]"
    # shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY,
    #                                                       stormpy.logic.ShieldComparison.ABSOLUTE, 0.9) 
 
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.1) 
    # shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.9) 
    
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)
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

        
def create_shield_dict(args):
    env = create_environment(args)
    # print(env.printGrid(init=False))
    
    grid_file = args.grid_path
    grid_to_prism_path = args.grid_to_prism_path
    export_grid_to_text(env, grid_file)
    
    prism_path = args.prism_path
    shield_dict = create_shield(grid_to_prism_path ,grid_file, prism_path)
    #shield_dict = {state.id : shield.get_choice(state).choice_map for state in model.states}
   
    #print(F"Shield dictionary {shield_dict}")
    # for state_id in model.states:
    #     choices = shield.get_choice(state_id)
    #     print(F"Allowed choices in state {state_id}, are {choices.choice_map} ")
    
    return shield_dict

