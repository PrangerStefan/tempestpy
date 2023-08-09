import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files

import random


def optimal_shield_simulator():
    path = stormpy.examples.files.prism_smg_lights
    formula_str = "<optimal, Optimal> <<shield>> R{\"differenceWithInterferenceCost\"}min=? [ LRA ]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
   
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)

    assert result.has_scheduler
    assert result.has_shield
   
    shield = result.shield
    
    scheduler = shield.construct()
    simulator = stormpy.simulator.create_simulator(model)#, seed=42)

    print(simulator)
    while not simulator.is_done():
        current_state = simulator.get_current_state()
        state_string = model.state_valuations.get_string(current_state)
     #   print(F"Simulator is in state {state_string}.")  
        temp = scheduler.get_choice(current_state)  
      #  print(F"Correction map is {temp.choice_map}")
     #   print([model.get_label_of_choice(current_state, x) for x in simulator.available_actions()])
        print(F"Available actions {simulator.available_actions()}")
        for action in simulator.available_actions():
            print(F"Action: {action} ActionLabel: {model.get_label_of_choice(current_state, action)}")
        observation, reward = simulator.step()
    


if __name__ == '__main__':
    optimal_shield_simulator()