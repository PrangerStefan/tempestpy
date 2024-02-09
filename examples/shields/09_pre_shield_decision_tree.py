import stormpy
import stormpy.core
import stormpy.simulator


import stormpy.shields

import stormpy.examples
import stormpy.examples.files

from sklearn.linear_model import LogisticRegression
from dtcontrol.benchmark_suite import BenchmarkSuite
from dtcontrol.decision_tree.decision_tree import DecisionTree
from dtcontrol.decision_tree.determinization.max_freq_determinizer import MaxFreqDeterminizer
from dtcontrol.decision_tree.impurity.entropy import Entropy
from dtcontrol.decision_tree.impurity.multi_label_entropy import MultiLabelEntropy
from dtcontrol.decision_tree.splitting.axis_aligned import AxisAlignedSplittingStrategy
from dtcontrol.decision_tree.splitting.linear_classifier import LinearClassifierSplittingStrategy


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

    shield_specification = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, stormpy.logic.ShieldComparison.RELATIVE, 0.9) 
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=shield_specification)
    
    assert result.has_shield

    shield = result.shield
    filename = "preshield.storm.json"
    stormpy.shields.export_shield(model, shield, filename)
    
    output_folder = "pre_trees"
    name = 'pre_my_output'

    aa = AxisAlignedSplittingStrategy()
    aa.priority = 1      

    classifiers = [DecisionTree([aa], Entropy(), name)]

    suite = create_decision_tree(filename, name=name , output_folder=output_folder, export_pdf=True, classifiers=classifiers)
    suite.display_html()

if __name__ == '__main__':
    export_shield_as_dot()