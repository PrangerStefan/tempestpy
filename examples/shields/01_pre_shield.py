import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files
import random


def pre_schield_01():
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
    assert result.has_schield
    
    shield = result.shield
    
    lookup = stormpy.shields.create_shield_action_lookup(model, shield)
    query = list(lookup.keys())[0]
    
    print(query)
    print(lookup[query])


if __name__ == '__main__':
    pre_schield_01()