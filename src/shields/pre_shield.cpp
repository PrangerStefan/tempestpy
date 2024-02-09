#include "pre_shield.h"

#include "storm/shields/PreShield.h"
#include "storm/shields/AbstractShield.h"


#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/Distribution.h"


template <typename ValueType, typename IndexType>
void define_pre_shield(py::module& m, std::string vt_suffix) {
    using PreShield = tempest::shields::PreShield<ValueType, IndexType>;
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;

    std::string shieldClassName = std::string("PreShield") + vt_suffix;


    py::class_<PreShield, AbstractShield, std::shared_ptr<PreShield>>(m, shieldClassName.c_str())
    .def("construct", &PreShield::construct, "Construct the shield")
    ;
}

template void define_pre_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m, std::string vt_suffix);
template void define_pre_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m, std::string vt_suffix);