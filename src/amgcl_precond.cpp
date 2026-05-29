#include "amgcl_precond.hpp"
#include "ngsolve_backend.hpp"

// AMGCL headers
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/chebyshev.hpp>

namespace ngsamgcl {

#if USE_NGSOLVE_BACKEND
  using Backend = amgcl::backend::ngsolve_backend;
  using VecType = NgsVector;
  inline void vec_zero(VecType & v) { v.GetBaseVector() = 0.0; }
#else
  using Backend = amgcl::backend::builtin<double>;
  using VecType = std::vector<double>;
  inline void vec_zero(VecType & v) { std::fill(v.begin(), v.end(), 0.0); }
#endif

  // Base class for type-erased AMG
  struct AMGBase {
    virtual ~AMGBase() = default;
    virtual void apply(const VecType & rhs, VecType & x) const = 0;
  };

  // Template implementation for specific coarsening/relaxation combination
  template<template<class> class Coarsening, template<class> class Relaxation>
  struct AMGImpl : public AMGBase {
    using AMG = amgcl::amg<Backend, Coarsening, Relaxation>;
    std::unique_ptr<AMG> amg;

    AMGImpl(const NgsMatrix & A, const AMGCLOptions & opts)
    {
      typename AMG::params prm;
      prm.npre = opts.npre;
      prm.npost = opts.npost;
      prm.max_levels = opts.max_levels;
      prm.coarse_enough = opts.coarse_enough;
      prm.direct_coarse = opts.direct_coarse;

      // Suspend TaskManager during setup so OpenMP gets all cores
      ngcore::SuspendTaskManager suspend;
      amg = std::make_unique<AMG>(A, prm);

      if (opts.printinfo)
        std::cout << *amg << std::endl;
    }

    void apply(const VecType & rhs, VecType & x) const override {
      amg->cycle(rhs, x);
    }
  };

  // ============================================================
  // AMGCLMatrix implementation (low-level BaseMatrix wrapper)
  // ============================================================

  struct AMGCLMatrix::Impl {
    std::unique_ptr<AMGBase> amg;
    // Cached work vectors to avoid allocation per Mult() call
    mutable VecType rhs;
    mutable VecType sol;
  };

  AMGCLMatrix::AMGCLMatrix(shared_ptr<BaseSparseMatrix> mat,
                           shared_ptr<BitArray> freedofs,
                           const AMGCLOptions & opts)
    : sparse_mat(mat), freedofs(freedofs), opts(opts)
  {
    height_ = mat->Height();
    width_ = mat->Width();

    impl = std::make_unique<Impl>();

    size_t ndof = mat->Height();

    // Build mapping: full DOF index -> reduced DOF index
    dof_map.assign(ndof, -1);
    n_free = 0;
    for (size_t i = 0; i < ndof; i++) {
      if (!freedofs || freedofs->Test(i)) {
        dof_map[i] = n_free++;
      }
    }

    // Dynamic cast to SparseMatrix<double>
    auto spm = dynamic_pointer_cast<SparseMatrix<double>>(mat);
    if (!spm) {
      throw Exception("AMGCLMatrix: matrix must be SparseMatrix<double>");
    }

    // Build the reduced (free-DOFs-only) matrix directly as NGSolve SparseMatrix
    Array<int> els_per_row(n_free);
    for (size_t i = 0; i < ndof; i++) {
      if (dof_map[i] < 0) continue;
      int cnt = 0;
      auto cols = spm->GetRowIndices(i);
      for (int j = 0; j < cols.Size(); j++)
        if (cols[j] < (int)ndof && dof_map[cols[j]] >= 0)
          cnt++;
      els_per_row[dof_map[i]] = cnt;
    }

    auto reduced_mat = make_shared<SparseMatrix<double>>(els_per_row, (int)n_free);

    for (size_t i = 0; i < ndof; i++) {
      if (dof_map[i] < 0) continue;
      int row_reduced = dof_map[i];
      auto src_cols = spm->GetRowIndices(i);
      auto src_vals = spm->GetRowValues(i);
      auto dst_cols = reduced_mat->GetRowIndices(row_reduced);
      auto dst_vals = reduced_mat->GetRowValues(row_reduced);
      int pos = 0;
      for (int j = 0; j < src_cols.Size(); j++) {
        int c = src_cols[j];
        if (c < (int)ndof && dof_map[c] >= 0) {
          dst_cols[pos] = dof_map[c];
          dst_vals[pos] = src_vals[j];
          pos++;
        }
      }
    }

    // Wrap as NgsMatrix for AMGCL
    NgsMatrix amgcl_mat;
    amgcl_mat.ngs_mat = reduced_mat;

    // Create the AMGCL AMG — pass NgsMatrix directly, no CRS tuple needed
    if (opts.coarsening == "smoothed_aggregation" && opts.relaxation == "spai0") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "smoothed_aggregation" && opts.relaxation == "gauss_seidel") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "smoothed_aggregation" && opts.relaxation == "ilu0") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::ilu0>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "smoothed_aggregation" && opts.relaxation == "damped_jacobi") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::damped_jacobi>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "ruge_stuben" && opts.relaxation == "spai0") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::spai0>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "ruge_stuben" && opts.relaxation == "gauss_seidel") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::gauss_seidel>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "ruge_stuben" && opts.relaxation == "ilu0") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::ilu0>>(amgcl_mat, opts);
    }
    else if (opts.coarsening == "ruge_stuben" && opts.relaxation == "damped_jacobi") {
      impl->amg = std::make_unique<AMGImpl<
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::damped_jacobi>>(amgcl_mat, opts);
    }
    else {
      throw Exception("AMGCLMatrix: unsupported coarsening/relaxation combination: "
                      + opts.coarsening + "/" + opts.relaxation);
    }

    // Pre-allocate work vectors
    impl->rhs = VecType(n_free, 0.0);
    impl->sol = VecType(n_free, 0.0);
  }

  AMGCLMatrix::~AMGCLMatrix() = default;

  void AMGCLMatrix::Mult(const BaseVector & x, BaseVector & y) const {
    auto fx = x.FV<double>();
    auto fy = y.FV<double>();

    size_t ndof = height_;

    // Use cached work vectors (avoid allocation per call)
    auto & rhs = impl->rhs;
    auto & sol = impl->sol;

    // Gather free DOFs (parallel)
    ngcore::ParallelFor(ndof, [&](size_t i) {
      if (dof_map[i] >= 0)
        rhs[dof_map[i]] = fx(i);
    });

    // Zero solution
    vec_zero(sol);

    // Apply AMG cycle
    impl->amg->apply(rhs, sol);

    // Scatter result back (parallel)
    fy = 0.0;
    ngcore::ParallelFor(ndof, [&](size_t i) {
      if (dof_map[i] >= 0)
        fy(i) = sol[dof_map[i]];
    });
  }

  AutoVector AMGCLMatrix::CreateRowVector() const {
    return sparse_mat->CreateColVector();
  }

  AutoVector AMGCLMatrix::CreateColVector() const {
    return sparse_mat->CreateRowVector();
  }

  // ============================================================
  // AMGCLPreconditioner implementation (Preconditioner subclass)
  // ============================================================

  AMGCLPreconditioner::AMGCLPreconditioner(shared_ptr<BilinearForm> bfa,
                                           const Flags & aflags,
                                           const string aname)
    : Preconditioner(bfa, aflags, aname), bfa(bfa)
  {
    // Parse options from flags
    if (aflags.StringFlagDefined("coarsening"))
      opts.coarsening = aflags.GetStringFlag("coarsening", "smoothed_aggregation");
    if (aflags.StringFlagDefined("relaxation"))
      opts.relaxation = aflags.GetStringFlag("relaxation", "spai0");
    if (aflags.NumFlagDefined("npre"))
      opts.npre = int(aflags.GetNumFlag("npre", 1));
    if (aflags.NumFlagDefined("npost"))
      opts.npost = int(aflags.GetNumFlag("npost", 1));
    if (aflags.NumFlagDefined("max_levels"))
      opts.max_levels = int(aflags.GetNumFlag("max_levels", 50));
    if (aflags.NumFlagDefined("coarse_enough"))
      opts.coarse_enough = int(aflags.GetNumFlag("coarse_enough", 2000));
  }

  AMGCLPreconditioner::AMGCLPreconditioner(shared_ptr<BilinearForm> bfa,
                                           const AMGCLOptions & opts,
                                           const Flags & aflags,
                                           const string aname)
    : Preconditioner(bfa, aflags, aname), bfa(bfa), opts(opts)
  { }

  void AMGCLPreconditioner::FinalizeLevel(const BaseMatrix * mat) {
    // Called by BilinearForm::Assemble() after the matrix is ready
    if (!mat) return;

    auto spmat = dynamic_cast<const BaseSparseMatrix*>(mat);
    if (!spmat) {
      // try getting from the bilinear form
      auto sp = dynamic_pointer_cast<BaseSparseMatrix>(bfa->GetMatrixPtr());
      if (!sp)
        throw Exception("AMGCLPreconditioner::FinalizeLevel: matrix is not a sparse matrix");
      auto freedofs = bfa->GetFESpace()->GetFreeDofs();
      cout << IM(3) << "AMGCLPreconditioner::FinalizeLevel - building AMG hierarchy" << endl;
      amg_matrix = make_shared<AMGCLMatrix>(sp, freedofs, opts);
    } else {
      auto sp = dynamic_pointer_cast<BaseSparseMatrix>(bfa->GetMatrixPtr());
      if (!sp)
        throw Exception("AMGCLPreconditioner::FinalizeLevel: matrix is not a sparse matrix");
      auto freedofs = bfa->GetFESpace()->GetFreeDofs();
      cout << IM(3) << "AMGCLPreconditioner::FinalizeLevel - building AMG hierarchy" << endl;
      amg_matrix = make_shared<AMGCLMatrix>(sp, freedofs, opts);
    }

    timestamp = NGS_Object::GetNextTimeStamp();
  }

  void AMGCLPreconditioner::Update() {
    // Called explicitly or by SolveBVP - rebuild if BilinearForm has been reassembled
    if (bfa->GetMatrixPtr() == nullptr) return;

    if (GetTimeStamp() < bfa->GetTimeStamp())
      FinalizeLevel(&bfa->GetMatrix());
  }

  const BaseMatrix & AMGCLPreconditioner::GetMatrix() const {
    if (!amg_matrix)
      ThrowPreconditionerNotReady();
    return *amg_matrix;
  }

  shared_ptr<BaseMatrix> AMGCLPreconditioner::GetMatrixPtr() {
    if (!amg_matrix)
      ThrowPreconditionerNotReady();
    return amg_matrix;
  }

  // Register with NGSolve's preconditioner factory
  static RegisterPreconditioner<AMGCLPreconditioner> init_amgcl_precond("amgcl");

} // namespace ngsamgcl
