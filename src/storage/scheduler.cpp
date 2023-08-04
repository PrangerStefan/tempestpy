#include "scheduler.h"
#include "src/helpers.h"

#include "storm/storage/Scheduler.h"
#include "storm/storage/PostScheduler.h"
#include "storm/storage/PreScheduler.h"

template<typename ValueType>
void define_scheduler(py::module& m, std::string vt_suffix) {
    using Scheduler = storm::storage::Scheduler<ValueType>;
    using SchedulerChoice = storm::storage::SchedulerChoice<ValueType>;
    using PreScheduler = storm::storage::PreScheduler<ValueType>;
    using PostScheduler = storm::storage::PostScheduler<ValueType>;


    std::string schedulerClassName = std::string("Scheduler") + vt_suffix;
    py::class_<Scheduler, std::shared_ptr<storm::storage::Scheduler<ValueType>>> scheduler(m, schedulerClassName.c_str(), "A Finite Memory Scheduler");
    scheduler
            .def("__str__", [](Scheduler const& s) {
                std::stringstream str;
                s.printToStream(str);
                return str.str();
            })
            .def_property_readonly("memoryless", &Scheduler::isMemorylessScheduler, "Is the scheduler memoryless?")
            .def_property_readonly("memory_size", &Scheduler::getNumberOfMemoryStates, "How much memory does the scheduler take?")
            .def_property_readonly("deterministic", &Scheduler::isDeterministicScheduler, "Is the scheduler deterministic?")
            .def_property_readonly("partial", &Scheduler::isPartialScheduler, "Is the scheduler partial?")
            .def("get_choice", &Scheduler::getChoice, py::arg("state_index"), py::arg("memory_index") = 0)
            .def("compute_action_support", &Scheduler::computeActionSupport, "nondeterministic_choice_indices"_a)
    ;

    std::string schedulerChoiceClassName = std::string("SchedulerChoice") + vt_suffix;
    py::class_<SchedulerChoice> schedulerChoice(m, schedulerChoiceClassName.c_str(), "A choice of a finite memory scheduler");
    schedulerChoice
        .def_property_readonly("defined", &SchedulerChoice::isDefined, "Is the choice defined by the scheduler?")
        .def_property_readonly("deterministic", &SchedulerChoice::isDeterministic, "Is the choice deterministic (given by a Dirac distribution)?")
        .def("get_deterministic_choice", &SchedulerChoice::getDeterministicChoice, "Get the deterministic choice")
        .def("get_choice", &SchedulerChoice::getChoiceAsDistribution, "Get the distribution over the actions")
        .def("__str__", &streamToString<SchedulerChoice>);

    std::string preSchedulerClassName = std::string("PreScheduler") + vt_suffix;
    py::class_<PreScheduler> preScheduler(m, preSchedulerClassName.c_str(), "A pre scheduler");
        preScheduler
            .def_property_readonly("memoryless", &PreScheduler::isMemorylessScheduler, "is the pre scheduler memoryless?")
            .def_property_readonly("memory_size", &PreScheduler::getNumberOfMemoryStates, "How much memory does the scheduler take?")

    ;

    std::string postSchedulerClassName = std::string("PostScheduler") + vt_suffix;
    py::class_<PostScheduler> postScheduler(m, postSchedulerClassName.c_str(), "A post scheduler");
        postScheduler
            .def_property_readonly("memoryless", &PostScheduler::isMemorylessScheduler, "is the pre scheduler memoryless?")
          
    ;

}


template void define_scheduler<double>(py::module& m, std::string vt_suffix);
template void define_scheduler<storm::RationalNumber>(py::module& m, std::string vt_suffix);