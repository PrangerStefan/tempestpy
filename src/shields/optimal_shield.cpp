#include "optimal_shield.h"

#include "storm/shields/AbstractShield.h"
#include "storm/shields/OptimalShield.h"

template <typename ValueType, typename IndexType>
void define_optimal_shield(py::module& m, std::string vt_suffix) {
    using OptimalShield = tempest::shields::OptimalShield<ValueType, IndexType>;
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;

    std::string shieldClassName = std::string("OptimalShield") + vt_suffix;

    py::class_<OptimalShield, AbstractShield, std::shared_ptr<OptimalShield>>(m, shieldClassName.c_str())
        .def("construct", &OptimalShield::construct, "Construct the shield")
    ;
}

template void define_optimal_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m, std::string vt_suffix);
template void define_optimal_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m, std::string vt_suffix);