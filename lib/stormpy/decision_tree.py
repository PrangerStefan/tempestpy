from sklearn.linear_model import LogisticRegression
from dtcontrol.benchmark_suite import BenchmarkSuite
from dtcontrol.decision_tree.decision_tree import DecisionTree
from dtcontrol.decision_tree.determinization.max_freq_determinizer import MaxFreqDeterminizer
from dtcontrol.decision_tree.impurity.entropy import Entropy
from dtcontrol.decision_tree.impurity.multi_label_entropy import MultiLabelEntropy
from dtcontrol.decision_tree.splitting.axis_aligned import AxisAlignedSplittingStrategy
from dtcontrol.decision_tree.splitting.linear_classifier import LinearClassifierSplittingStrategy

import pydot

def create_decision_tree(filename, name, output_folder, 
                         timeout=60*60*2,
                         benchmark_file='benchmark',
                         save_folder='saved_classifiers',
                         export_pdf=False,
                         classifiers=None):
    
    suite = BenchmarkSuite(timeout=timeout,
                            save_folder=save_folder,
                            output_folder=output_folder,
                            benchmark_file=benchmark_file,
                            rerun=True)

    suite.add_datasets([filename])  


    if classifiers is None:        
        aa = AxisAlignedSplittingStrategy()
        aa.priority = 1      

        classifiers = [DecisionTree([aa], Entropy(), name)]

    suite.benchmark(classifiers)
    if export_pdf:
        for dataset in suite.datasets:
            for classifier in classifiers:
                filename = suite.get_filename(output_folder, dataset=dataset , classifier=classifier, extension='.dot')
                (graph,) = pydot.graph_from_dot_file(filename)
                graph.write_pdf(F'{name}.pdf')


    return suite
