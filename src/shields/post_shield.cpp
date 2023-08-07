#include "post_shield.h"

#include "storm/shields/AbstractShield.h"
#include "storm/shields/PostShield.h"

template <typename ValueType, typename IndexType>
void define_post_shield(py::module& m, std::string vt_suffix) {
    using PostShield = tempest::shields::PostShield<ValueType, IndexType>;
    using AbstractShield = tempest::shields::AbstractShield<ValueType, IndexType>;

    std::string shieldClassName = std::string("PostShield") + vt_suffix;
    
    py::class_<PostShield, AbstractShield, std::shared_ptr<PostShield>>(m, shieldClassName.c_str())
        .def("construct", &PostShield::construct, "Construct the shield")
    ;
}


template void define_post_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m, std::string vt_suffix);
template void define_post_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m, std::string vt_suffix);