import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files


"""
Example of exporting a Optimal Shield
to a file
"""

def optimal_shield_export():
    path = stormpy.examples.files.prism_smg_lights
    formula_str = "<optimal, Optimal> <<shield>> R{\"differenceWithInterferenceCost\"}min=? [ LRA ]"

    program = stormpy.parse_prism_program(path)
    formulas = stormpy.parse_properties_for_prism_program(formula_str, program)

    options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
    options.set_build_state_valuations(True)
    options.set_build_choice_labels(True)
    options.set_build_all_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
   
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
    assert result.has_scheduler
    assert result.has_shield

    shield = result.shield
    
    stormpy.shields.export_shieldDouble(model, shield, "optimal.shield")


if __name__ == '__main__':
    optimal_shield_export()