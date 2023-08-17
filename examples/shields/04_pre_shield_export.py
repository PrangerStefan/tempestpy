import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files

"""

Example of exporting a Pre Safety Shield
to a file

"""

def pre_schield():
    path = stormpy.examples.files.prism_mdp_lava_simple
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
    
    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.Absolute, 0.9) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    
    assert result.has_scheduler
    assert result.has_shield
    
    shield = result.shield

    stormpy.shields.export_shield(model, shield, "pre.shield")
    


if __name__ == '__main__':
    pre_schield()