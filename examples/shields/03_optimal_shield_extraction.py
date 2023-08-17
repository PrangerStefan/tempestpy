import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files


def optimal_shield_extraction():
    path = stormpy.examples.files.prism_smg_robot
    formula_str = "<Optimal> <<sh>> R{\"travel_costs\"}min=? [ LRA ]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
   
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.OPTIMAL) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    
    assert result.has_scheduler
    assert result.has_shield
   
    shield = result.shield

    state_ids = [x for x in model.states]
    scheduler = shield.construct()

    for state_id in state_ids[0:50]:
        choices = scheduler.get_choice(state_id)
        print(F"Corrections in state {state_id}, are {choices.choice_map} ")
    


if __name__ == '__main__':
    optimal_shield_extraction()