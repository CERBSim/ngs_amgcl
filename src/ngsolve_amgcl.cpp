#include <comp.hpp>
#include <python_comp.hpp>

#include "amgcl_precond.hpp"

using namespace ngsamgcl;

PYBIND11_MODULE(ngsolve_amgcl, m) {
  m.doc() = "NGSolve-AMGCL: Algebraic Multigrid preconditioner for NGSolve using AMGCL";

  py::class_<AMGCLOptions>(m, "AMGCLOptions")
    .def(py::init<>())
    .def_readwrite("coarsening", &AMGCLOptions::coarsening,
                   "Coarsening strategy: 'smoothed_aggregation', 'ruge_stuben'")
    .def_readwrite("relaxation", &AMGCLOptions::relaxation,
                   "Relaxation type: 'spai0', 'gauss_seidel', 'ilu0', 'damped_jacobi'")
    .def_readwrite("coarsening_aggr_eps_strong", &AMGCLOptions::coarsening_aggr_eps_strong,
                   "Strength threshold for aggregation")
    .def_readwrite("npre", &AMGCLOptions::npre, "Number of pre-smoothing steps")
    .def_readwrite("npost", &AMGCLOptions::npost, "Number of post-smoothing steps")
    .def_readwrite("max_levels", &AMGCLOptions::max_levels, "Maximum number of AMG levels")
    .def_readwrite("coarse_enough", &AMGCLOptions::coarse_enough,
                   "Coarse level size threshold")
    .def_readwrite("direct_coarse", &AMGCLOptions::direct_coarse,
                   "Use direct solver on coarsest level")
    .def_readwrite("printinfo", &AMGCLOptions::printinfo,
                   "Print AMG hierarchy info during setup")
    .def("__repr__", [](const AMGCLOptions & o) {
      return "AMGCLOptions(coarsening='" + o.coarsening + "', relaxation='" + o.relaxation +
             "', npre=" + std::to_string(o.npre) + ", npost=" + std::to_string(o.npost) +
             ", max_levels=" + std::to_string(o.max_levels) +
             ", coarse_enough=" + std::to_string(o.coarse_enough) + ")";
    });

  // High-level: takes a BilinearForm, auto-updates on reassembly
  py::class_<AMGCLPreconditioner, shared_ptr<AMGCLPreconditioner>, ngcomp::Preconditioner>
    (m, "AMGCLPreconditioner")
    .def(py::init([](shared_ptr<ngcomp::BilinearForm> bfa,
                     const AMGCLOptions & opts) {
           auto pre = make_shared<AMGCLPreconditioner>(bfa, opts);
           // If the BilinearForm already has a matrix, build immediately
           if (bfa->GetMatrixPtr())
             pre->Update();
           return pre;
         }),
         py::arg("bf"),
         py::arg("options") = AMGCLOptions(),
         R"(Create an AMGCL algebraic multigrid preconditioner.

Parameters
----------
bf : ngsolve.BilinearForm
    The bilinear form to precondition. The preconditioner automatically
    rebuilds when bf.Assemble() is called.
options : AMGCLOptions
    Configuration options for the AMGCL preconditioner.
)")
    .def("Update", &AMGCLPreconditioner::Update,
         "Rebuild the AMG hierarchy (called automatically by Assemble)")
    .def_property("options",
                  &AMGCLPreconditioner::GetOptions,
                  &AMGCLPreconditioner::SetOptions,
                  "Get/set AMGCL options (takes effect on next Update/Assemble)");
}
