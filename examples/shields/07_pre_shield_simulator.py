import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields

import stormpy.examples
import stormpy.examples.files

import random

"""
Simulating a model with the usage of a pre shield
"""

def example_pre_shield_simulator():
    path = stormpy.examples.files.prism_mdp_cliff_walking
    formula_str = "Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)

    initial_state = model.initial_states[0]
    assert initial_state == 0
    
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.9) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    
    assert result.has_scheduler
    assert result.has_shield
    
    shield = result.shield

    pre_scheduler = shield.construct()

    simulator = stormpy.simulator.create_simulator(model, seed=42)

    while not simulator.is_done():
        current_state = simulator.get_current_state()
        state_string = model.state_valuations.get_string(current_state)
        print(F"Simulator is in state {state_string}.")
        choices = [x for x in pre_scheduler.get_choice(current_state).choice_map if x[0] > 0]
        choice_labels =  [model.choice_labeling.get_labels_of_choice(model.get_choice_index(current_state, choice[1])) for choice in choices]
        
        if not choices:
            break

        index = random.randint(0, len(choices) - 1)
        selected_action = choices[index]
        choice_label = model.choice_labeling.get_labels_of_choice(model.get_choice_index(current_state, selected_action[1]))
        print(F"Allowed Choices are {choice_labels}. Selected Action: {choice_label}")
        observation, reward = simulator.step(selected_action[1])

        


if __name__ == '__main__':
    example_pre_shield_simulator()
