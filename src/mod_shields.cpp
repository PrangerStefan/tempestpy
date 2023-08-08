#include "common.h"
#include "shields/abstract_shield.h"
#include "shields/optimal_shield.h"
#include "shields/post_shield.h"
#include "shields/pre_shield.h"
#include "shields/shield_handling.h"


#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/Distribution.h"

PYBIND11_MODULE(shields, m) {
    m.doc() = "shields";

#ifdef STORMPY_DISABLE_SIGNATURE_DOC
    py::options options;
    options.disable_function_signatures();
#endif
    define_abstract_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(m, "Double");
    define_abstract_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(m, "Exact");
    define_pre_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(m, "Double");
    define_pre_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(m, "Exact");
    define_post_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(m, "Double");
    define_post_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(m, "Exact");
    define_optimal_shield<double, typename storm::storage::SparseMatrix<double>::index_type>(m, "Double");
    define_optimal_shield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(m, "Exact");
    define_shield_handling<double, typename storm::storage::SparseMatrix<double>::index_type>(m, "Double");
    define_shield_handling<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>(m, "Exact");
}
