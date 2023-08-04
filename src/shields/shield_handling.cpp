#include "shield_handling.h"

#include "storm/shields/ShieldHandling.h"

template <typename ValueType, typename IndexType>
void define_shield_handling(py::module& m) {
    m.def("create_shield", &tempest::shields::createShield<ValueType, IndexType>, "hi");
}

template void define_shield_handling<double, typename storm::storage::SparseMatrix<double>::index_type>(py::module& m);
