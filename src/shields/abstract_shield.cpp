#include "abstract_shield.h"
#include "storm/shields/AbstractShield.h"

#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/Distribution.h"

#include "storm/api/export.h"


template <typename ValueType, typename IndexType>
void define_abstract_shield(py::module& m, std::string vt_suffix) {
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;
    std::string shieldClassName = std::string("AbstractShield") + vt_suffix;

    py::class_<AbstractShield, std::shared_ptr<AbstractShield>> shield(m, shieldClassName.c_str());
        shield
        .def("compute_row_group_size", &AbstractShield::computeRowGroupSizes)
        .def("get_class_name", &AbstractShield::getClassName)
        .def("get_optimization_direction", &AbstractShield::getOptimizationDirection)
      ;
}

template void define_abstract_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m, std::string vt_suffix);
template void define_abstract_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m, std::string vt_suffix);