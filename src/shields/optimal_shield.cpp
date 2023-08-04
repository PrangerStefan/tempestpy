#include "optimal_shield.h"

#include "storm/shields/AbstractShield.h"
#include "storm/shields/OptimalShield.h"

template <typename ValueType, typename IndexType>
void define_optimal_shield(py::module& m) {
    using OptimalShield = tempest::shields::OptimalShield<ValueType, IndexType>;
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;

    py::class_<OptimalShield, AbstractShield>(m, "OptimalShield")
    ;
}

template void define_optimal_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m);
template void define_optimal_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m);