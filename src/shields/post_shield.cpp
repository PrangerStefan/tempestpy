#include "post_shield.h"

#include "storm/shields/AbstractShield.h"
#include "storm/shields/PostShield.h"

template <typename ValueType, typename IndexType>
void define_post_shield(py::module& m) {
    using PostShield = tempest::shields::PostShield<ValueType, IndexType>;
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;
    
    py::class_<PostShield, AbstractShield>(m, "PostShield")
    ;
}


template void define_post_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m);
template void define_post_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m);