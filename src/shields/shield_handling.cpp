#include "shield_handling.h"

#include "storm/shields/ShieldHandling.h"
#include "storm/api/export.h"

template <typename ValueType, typename IndexType>
void define_shield_handling(py::module& m, std::string vt_suffix) {
    std::string shieldHandlingname = std::string("export_shield");

    m.def(shieldHandlingname.c_str(), &storm::api::exportShield<ValueType, IndexType>, py::arg("model"), py::arg("shield"), py::arg("filename"));
    }

template void define_shield_handling<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m, std::string vt_suffix);
template void define_shield_handling<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(py::module& m, std::string vt_suffix);
