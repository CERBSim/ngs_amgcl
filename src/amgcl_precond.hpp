#ifndef AMGCL_PRECOND_HPP
#define AMGCL_PRECOND_HPP

#include <comp.hpp>
#include <la.hpp>

#include <memory>
#include <vector>
#include <string>

namespace ngsamgcl {

  using namespace ngla;
  using namespace ngcomp;

  // Configuration for AMGCL preconditioner
  struct AMGCLOptions {
    // Coarsening type: "smoothed_aggregation", "ruge_stuben"
    std::string coarsening = "smoothed_aggregation";
    // Relaxation type: "spai0", "gauss_seidel", "ilu0", "damped_jacobi"
    std::string relaxation = "spai0";
    // Coarsening aggregation over (eps_strong)
    double coarsening_aggr_eps_strong = 0.08;
    // Number of pre/post relaxation sweeps
    int npre = 1;
    int npost = 1;
    // Maximum number of AMG levels
    int max_levels = 50;
    // Coarse level size threshold
    int coarse_enough = 2000;
    // Direct solver on coarsest level
    bool direct_coarse = true;
    // Coarse solver type: "sparsecholesky", "pardiso", "umfpack", "skyline_lu"
    std::string coarse_solver = "sparsecholesky";
    // Print AMG hierarchy info during setup
    bool printinfo = false;
  };

  //----------------------------------------------------------------------
  // Low-level AMG matrix wrapper (BaseMatrix)
  // Wraps a single AMGCL AMG hierarchy built from a given sparse matrix.
  //----------------------------------------------------------------------
  class AMGCLMatrix : public BaseMatrix {
  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    shared_ptr<BaseSparseMatrix> sparse_mat;
    shared_ptr<BitArray> freedofs;
    AMGCLOptions opts;
    int height_, width_;
    // Cached DOF mapping
    std::vector<int> dof_map;
    size_t n_free;

  public:
    AMGCLMatrix(shared_ptr<BaseSparseMatrix> mat,
                shared_ptr<BitArray> freedofs,
                const AMGCLOptions & opts = AMGCLOptions());
    ~AMGCLMatrix();

    int VHeight() const override { return height_; }
    int VWidth() const override { return width_; }

    void Mult(const BaseVector & x, BaseVector & y) const override;

    AutoVector CreateRowVector() const override;
    AutoVector CreateColVector() const override;
  };

  //----------------------------------------------------------------------
  // High-level Preconditioner (registers with BilinearForm)
  // Automatically rebuilds when the BilinearForm reassembles.
  //----------------------------------------------------------------------
  class AMGCLPreconditioner : public Preconditioner {
  private:
    shared_ptr<BilinearForm> bfa;
    shared_ptr<AMGCLMatrix> amg_matrix;
    AMGCLOptions opts;

  public:
    AMGCLPreconditioner(shared_ptr<BilinearForm> bfa, const Flags & aflags,
                        const string aname = "amgcl");

    // Direct construction with options (for Python convenience)
    AMGCLPreconditioner(shared_ptr<BilinearForm> bfa, const AMGCLOptions & opts,
                        const Flags & aflags = Flags(),
                        const string aname = "amgcl");

    virtual ~AMGCLPreconditioner() = default;

    /// Called by BilinearForm after assembly completes
    void FinalizeLevel(const BaseMatrix * mat) override;

    /// Called explicitly or by SolveBVP; rebuilds if timestamps indicate change
    void Update() override;

    const BaseMatrix & GetMatrix() const override;
    shared_ptr<BaseMatrix> GetMatrixPtr() override;

    const BaseMatrix & GetAMatrix() const override {
      return bfa->GetMatrix();
    }

    const char * ClassName() const override {
      return "AMGCL Preconditioner";
    }

    void SetOptions(const AMGCLOptions & new_opts) { opts = new_opts; }
    const AMGCLOptions & GetOptions() const { return opts; }
  };

} // namespace ngsamgcl

#endif // AMGCL_PRECOND_HPP
