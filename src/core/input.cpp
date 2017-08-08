#include "input.h"
#include "src/helpers.h"

void define_property(py::module& m) {
    py::class_<storm::jani::Property>(m, "Property", "Property")
        .def(py::init<std::string const&, std::shared_ptr<storm::logic::Formula const> const&, std::string const&>(), "Construct property from formula", py::arg("name"), py::arg("formula"), py::arg("comment") = "")
        .def_property_readonly("name", &storm::jani::Property::getName, "Obtain the name of the property")
        .def_property_readonly("raw_formula", &storm::jani::Property::getRawFormula, "Obtain the formula directly")
        .def("__str__", &streamToString<storm::jani::Property>)
    ;
}


// Define python bindings
void define_input(py::module& m) {

    // Parse Prism program
    m.def("parse_prism_program", &storm::api::parseProgram, "Parse Prism program", py::arg("path"));
    // Parse Jani model
    m.def("parse_jani_model", &storm::api::parseJaniModel, "Parse Jani model", py::arg("path"));

    // PrismType
    py::enum_<storm::prism::Program::ModelType>(m, "PrismModelType", "Type of the prism model")
        .value("DTMC", storm::prism::Program::ModelType::DTMC)
        .value("CTMC", storm::prism::Program::ModelType::CTMC)
        .value("MDP", storm::prism::Program::ModelType::MDP)
        .value("CTMDP", storm::prism::Program::ModelType::CTMDP)
        .value("MA", storm::prism::Program::ModelType::MA)
        .value("UNDEFINED", storm::prism::Program::ModelType::UNDEFINED)
    ;

    // PrismProgram
    py::class_<storm::prism::Program>(m, "PrismProgram", "Prism program")
        .def_property_readonly("nr_modules", &storm::prism::Program::getNumberOfModules, "Number of modules")
        .def_property_readonly("model_type", &storm::prism::Program::getModelType, "Model type")
        .def_property_readonly("has_undefined_constants", &storm::prism::Program::hasUndefinedConstants, "Flag if program has undefined constants")
        .def_property_readonly("undefined_constants_are_graph_preserving", &storm::prism::Program::undefinedConstantsAreGraphPreserving, "Flag if the undefined constants do not change the graph structure")
        .def("__str__", &streamToString<storm::prism::Program>)
    ;

    // JaniType
    py::enum_<storm::jani::ModelType>(m, "JaniModelType", "Type of the Jani model")
        .value("DTMC", storm::jani::ModelType::DTMC)
        .value("CTMC", storm::jani::ModelType::CTMC)
        .value("MDP", storm::jani::ModelType::MDP)
        .value("CTMDP", storm::jani::ModelType::CTMDP)
        .value("MA", storm::jani::ModelType::MA)
        .value("LTS", storm::jani::ModelType::LTS)
        .value("TA", storm::jani::ModelType::TA)
        .value("PTA", storm::jani::ModelType::PTA)
        .value("STA", storm::jani::ModelType::STA)
        .value("HA", storm::jani::ModelType::HA)
        .value("PHA", storm::jani::ModelType::PHA)
        .value("SHA", storm::jani::ModelType::SHA)
        .value("UNDEFINED", storm::jani::ModelType::UNDEFINED)
    ;

    // Jani Model
    py::class_<storm::jani::Model>(m, "JaniModel", "Jani Model")
        .def_property_readonly("name", &storm::jani::Model::getName, "Name of the jani model")
        .def_property_readonly("model_type", &storm::jani::Model::getModelType, "Model type")
        .def_property_readonly("has_undefined_constants", &storm::jani::Model::hasUndefinedConstants, "Flag if Jani model has undefined constants")
        .def_property_readonly("undefined_constants_are_graph_preserving", &storm::jani::Model::undefinedConstantsAreGraphPreserving, "Flag if the undefined constants do not change the graph structure")
    ;
    
    // SymbolicModelDescription
    py::class_<storm::storage::SymbolicModelDescription>(m, "SymbolicModelDescription", "Symbolic description of model")
        .def(py::init<storm::prism::Program const&>(), "Construct from Prism program", py::arg("prism_program"))
        .def(py::init<storm::jani::Model const&>(), "Construct from Jani model", py::arg("jani_model"))
        .def_property_readonly("is_prism_program", &storm::storage::SymbolicModelDescription::isPrismProgram, "Flag if program is in Prism format")
        .def_property_readonly("is_jani_model", &storm::storage::SymbolicModelDescription::isJaniModel, "Flag if program is in Jani format")
        .def("parse_constant_definitions", &storm::storage::SymbolicModelDescription::parseConstantDefinitions, "Parse given constant definitions", py::arg("String containing constants and their values"))
        .def("instantiate_constants", [](storm::storage::SymbolicModelDescription const& description, std::map<storm::expressions::Variable, storm::expressions::Expression> const& constantDefinitions) {
                return description.preprocess(constantDefinitions);
            }, "Instantiate constants in symbolic model description", py::arg("constant_definitions"))
    ;

    // PrismProgram and JaniModel can be converted into SymbolicModelDescription
    py::implicitly_convertible<storm::prism::Program, storm::storage::SymbolicModelDescription>();
    py::implicitly_convertible<storm::jani::Model, storm::storage::SymbolicModelDescription>();
}
