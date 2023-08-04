#include "pre_shield.h"

#include "storm/shields/PreShield.h"
#include "storm/shields/AbstractShield.h"


#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/Distribution.h"


template <typename ValueType, typename IndexType>
void define_pre_shield(py::module& m) {
    using PreShield = tempest::shields::PreShield<ValueType, IndexType>;
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;
    py::class_<PreShield, AbstractShield>(m, "PreShield")
    ;
}

template void define_pre_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m);
