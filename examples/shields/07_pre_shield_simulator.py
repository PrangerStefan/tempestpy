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
    path = stormpy.examples.files.prism_mdp_lava_simple
    formula_str = "<ShieldFileName, PreSafety, gamma=0.9> Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)

    initial_state = model.initial_states[0]
    assert initial_state == 0
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
    assert result.has_scheduler
    assert result.has_shield
    
    shield = result.shield

    pre_scheduler = shield.construct()

    simulator = stormpy.simulator.create_simulator(model, seed=42)
    final_outcomes = dict()
    for n in range(1000):
        while not simulator.is_done():
            current_state = simulator.get_current_state()
            choices = pre_scheduler.get_choice(current_state).choice_map
            index = random.randint(0, len(choices) - 1)
            selected_action = choices[index]
            state_string = model.state_valuations.get_string(current_state)
            print(F"Simulator is in state {state_string}. Allowed Choices are {choices}. Selected Action: {selected_action}")
            observation, reward = simulator.step(selected_action[1])
        if observation not in final_outcomes:
            final_outcomes[observation] = 1
        else:
            final_outcomes[observation] += 1
        simulator.restart()



if __name__ == '__main__':
    example_pre_shield_simulator()
