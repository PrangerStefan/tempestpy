import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files

"""

Example for the extraction of a Pre Safety Shield 
from a model checking result and querying the shield 
for allowed choices in a state.

"""

def pre_shield_extraction():
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
    #shield_specification = ShieldingExpression(type="PreSafety", gamma=0.8) TODO Parameter for shield expression would be nice to have
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True) #, shielding_expression=shield_specification)
    assert result.has_scheduler
    assert result.has_shield
    
    shield = result.shield
    
    for state_id in model.states:
        choices = shield.construct().get_choice(state_id)
        print(F"Allowed choices in state {state_id}, are {choices.choice_map} ")


if __name__ == '__main__':
    pre_shield_extraction()