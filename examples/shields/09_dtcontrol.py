import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files

from stormpy.dtcontrol import export_decision_tree

def export_shield_as_dot():
    path = stormpy.examples.files.prism_mdp_lava_simple
    formula_str = "<pre, PreSafety, lambda=0.9> Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    options.set_build_with_choice_origins(True)
    model = stormpy.build_sparse_model_with_options(program, options)

    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True) #, shielding_expression=shield_specification)
    
    assert result.has_shield

    shield = result.shield
    stormpy.shields.export_shieldDouble(model, shield, "preshield.storm.json")


    export_decision_tree(result.shield)



if __name__ == '__main__':
    export_shield_as_dot()