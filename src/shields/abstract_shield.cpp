#include "abstract_shield.h"
#include "storm/shields/AbstractShield.h"

#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/Distribution.h"

template <typename ValueType, typename IndexType>
void define_abstract_shield(py::module& m) {
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;
    py::class_<AbstractShield>(m, "AbstractShield")
        .def("compute_row_group_size", &AbstractShield::computeRowGroupSizes)
        .def("get_class_name", &AbstractShield::getClassName)
        .def("get_optimization_direction", &AbstractShield::getOptimizationDirection)
    ;
}

template void define_abstract_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m);
