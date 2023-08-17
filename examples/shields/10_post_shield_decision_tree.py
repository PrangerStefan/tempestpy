import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files

from stormpy.decision_tree import create_decision_tree

def export_shield_as_dot():
    path = stormpy.examples.files.prism_mdp_lava_simple
    formula_str = "Pmax=? [G !\"AgentIsInLavaAndNotDone\"]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    options.set_build_with_choice_origins(True)
    model = stormpy.build_sparse_model_with_options(program, options)

    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.POST_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.9) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)

    assert result.has_shield

    shield = result.shield
    filename = "postshield.storm.json"
    filename2 = "postshield.shield"
    stormpy.shields.export_shield(model, shield, filename)
    stormpy.shields.export_shield(model, shield, filename2)
    
    output_folder = "post_trees"
    name = 'post_my_output'
    suite = create_decision_tree(filename, name=name , output_folder=output_folder, export_pdf=True)
    suite.display_html()

if __name__ == '__main__':
    export_shield_as_dot()