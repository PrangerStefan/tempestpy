import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files
import random


def optimal_shield_03():
    path = stormpy.examples.files.prism_smg_robot
    formula_str = "<path_correction, Optimal> <<sh>> R{\"travel_costs\"}min=? [ LRA ]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
   
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
    assert result.has_scheduler
    print(F"Check Scheduler: {result.has_scheduler}")
    print(F"Check Shield: {result.has_schield}")

    print(type(result))
   
    shield = result.shield
    scheduler = result.scheduler
    
    print(type(shield))

    assert scheduler.memoryless
    assert scheduler.deterministic

    constructed_shield = shield.construct()

    print(type(constructed_shield))
    
    stormpy.shields.export_shieldDouble(model, shield)
    
    # for state in model.states:
    #     choice = scheduler.get_choice(state)
    #     action = choice.get_deterministic_choice()
    #     print("In state {} choose action {}".format(state, action))

    # dtmc = model.apply_scheduler(scheduler)
    # print(dtmc)



if __name__ == '__main__':
    optimal_shield_03()