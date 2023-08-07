#include "shield_handling.h"

#include "storm/shields/ShieldHandling.h"
#include "storm/api/export.h"

template <typename ValueType, typename IndexType>
void define_shield_handling(py::module& m) {
    m.def("export_shield", &storm::api::exportShield<ValueType, IndexType>, py::arg("model"), py::arg("shield"));
}

template void define_shield_handling<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m);
