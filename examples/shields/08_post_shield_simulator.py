import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields

import stormpy.examples
import stormpy.examples.files


"""
Simulating a model with the usage of a post shield
"""

def example_post_shield_simulator():
    path = stormpy.examples.files.prism_mdp_lava_simple
    formula_str = "Pmax=? [G !\"AgentIsInLavaAndNotDone\"]; Pmax=? [ F \"AgentIsInLavaAndNotDone\" ];"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)

    initial_state = model.initial_states[0]
    assert initial_state == 0
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.POST_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.9) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    result2 = stormpy.model_checking(model, formulas[1], extract_scheduler=True)

    assert result.has_shield
    assert result2.has_scheduler

    shield = result.shield
    scheduler = result2.scheduler

    post_scheduler = shield.construct()

    simulator = stormpy.simulator.create_simulator(model, seed=42)

    while not simulator.is_done():
        current_state = simulator.get_current_state()
        state_string = model.state_valuations.get_string(current_state)
        # print(F"Simulator is in state {state_string}.")

        sched_choice = scheduler.get_choice(current_state).get_deterministic_choice()
        # print(F"Scheduler choice {model.choice_labeling.get_labels_of_choice(model.get_choice_index(current_state, sched_choice))}")

        corrections = post_scheduler.get_choice(current_state).choice_map
        # print(corrections)
        correction_labels =  [(model.get_label_of_choice(current_state, correction[0]),  model.get_label_of_choice(current_state, correction[1])) for correction in corrections]
        # print(F"Correction Choices are {correction_labels}.")
        
        applied_correction = next((x[1] for x in corrections if x[0] == sched_choice), None)

        if applied_correction != None and applied_correction != sched_choice:
            print(F"Correction applied changed choice {sched_choice} to {applied_correction}")
            sched_choice = applied_correction

        observation, reward = simulator.step(sched_choice)




if __name__ == '__main__':
    example_post_shield_simulator()