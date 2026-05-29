#ifndef NGSOLVE_AMGCL_BACKEND_HPP
#define NGSOLVE_AMGCL_BACKEND_HPP

/**
 * Custom AMGCL backend using NGSolve's SparseMatrix and vector operations.
 *
 * - SpMV uses NGSolve's SparseMatrix::Mult/MultAdd (TaskManager-parallel)
 * - Inner products use ngla::InnerProduct (TaskManager-parallel)
 * - Vector ops use FlatVector operations (TaskManager-parallel)
 * - Matrix type wraps NGSolve's SparseMatrix<double> (converted from AMGCL's CRS)
 *
 * This keeps all parallel work within NGSolve's TaskManager thread pool.
 */

#include <vector>
#include <numeric>
#include <memory>
#include <type_traits>

#include <la.hpp>
#include <comp.hpp>
#include <vvector.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/backend/builtin.hpp>

namespace ngsamgcl {

using namespace ngla;

// ============================================================
// Matrix wrapper: directly wraps NGSolve's SparseMatrix<double>
// which is already in CSR format. No separate storage needed.
// ============================================================
struct NgsMatrix {
  typedef double value_type;
  typedef double val_type;
  typedef int col_type;
  typedef int ptr_type;

  shared_ptr<SparseMatrix<double>> ngs_mat;

  NgsMatrix() = default;

  // Construct from AMGCL's CRS format (during setup phase)
  template <class CRS>
  NgsMatrix(const CRS & A) {
    size_t nrows = amgcl::backend::rows(A);
    size_t ncols = amgcl::backend::cols(A);

    // Count nonzeros per row
    Array<int> els_per_row(nrows);
    for (size_t i = 0; i < nrows; ++i) {
      int cnt = 0;
      for (auto a = amgcl::backend::row_begin(A, i); a; ++a) ++cnt;
      els_per_row[i] = cnt;
    }

    ngs_mat = make_shared<SparseMatrix<double>>(els_per_row, (int)ncols);

    // Fill column indices and values directly into the SparseMatrix
    for (size_t i = 0; i < nrows; ++i) {
      auto cols = ngs_mat->GetRowIndices(i);
      auto vals = ngs_mat->GetRowValues(i);
      int pos = 0;
      for (auto a = amgcl::backend::row_begin(A, i); a; ++a) {
        cols[pos] = static_cast<int>(a.col());
        vals[pos] = a.value();
        ++pos;
      }
    }
  }

  size_t rows() const { return ngs_mat->Height(); }
  size_t cols() const { return ngs_mat->Width(); }
  size_t nonzeros() const {
    size_t nnz = 0;
    for (size_t i = 0; i < (size_t)ngs_mat->Height(); ++i)
      nnz += ngs_mat->GetRowIndices(i).Size();
    return nnz;
  }

  size_t bytes() const {
    size_t nnz = nonzeros();
    size_t n = rows();
    return (n+1)*sizeof(int) + nnz*(sizeof(int)+sizeof(double));
  }

  // Row iterator reading directly from the SparseMatrix's storage
  struct row_iter {
    const int * col_p;
    const double * val_p;
    const int * end_p;

    row_iter(FlatArray<int> cols, FlatVector<double> vals)
      : col_p(cols.Data()), val_p(vals.Data()),
        end_p(cols.Data() + cols.Size()) {}

    operator bool() const { return col_p < end_p; }
    row_iter & operator++() { ++col_p; ++val_p; return *this; }

    int col() const { return *col_p; }
    double value() const { return *val_p; }
  };

  row_iter row_begin(size_t i) const {
    return row_iter(ngs_mat->GetRowIndices(i), ngs_mat->GetRowValues(i));
  }
};

// ============================================================
// Direct solver wrapper: uses NGSolve's inverse (SparseCholesky,
// PARDISO, UMFPACK, etc.) on the coarsest level.
// ============================================================
struct NgsDirectSolver {
  shared_ptr<BaseMatrix> inv;

  // AMGCL queries this to decide default coarse_enough
  static size_t coarse_enough() { return 3000; }

  NgsDirectSolver(const NgsMatrix & A, const std::string & type = "sparsecholesky") {
    auto freedofs = make_shared<BitArray>(A.ngs_mat->Height());
    freedofs->Set();
    inv = A.ngs_mat->InverseMatrix(freedofs);
  }

  // AMGCL calls operator()(rhs, x) to solve A*x = rhs
  template <class Vec1, class Vec2>
  void operator()(const Vec1 & rhs, Vec2 & x) const {
    // NgsVector wraps VVector<double> which IS a BaseVector — pass directly
    inv->Mult(rhs.GetBaseVector(), x.GetBaseVector());
  }
};

// ============================================================
// Vector type: wraps NGSolve's VVector<double> (a BaseVector)
// Provides operator[] for AMGCL's element access needs.
// All parallel ops go through BaseVector interface directly.
// ============================================================
struct NgsVector {
  typedef double value_type;
  shared_ptr<VVector<double>> vec;

  NgsVector() = default;
  explicit NgsVector(size_t n, double val = 0.0)
    : vec(make_shared<VVector<double>>(n)) {
    *vec = val;
  }
  NgsVector(const std::vector<double> & v)
    : vec(make_shared<VVector<double>>(v.size())) {
    auto fv = vec->FVDouble();
    for (size_t i = 0; i < v.size(); ++i)
      fv[i] = v[i];
  }

  size_t size() const { return vec ? vec->Size() : 0; }

  double & operator[](size_t i) { return vec->FVDouble()[i]; }
  const double & operator[](size_t i) const { return vec->FVDouble()[i]; }

  double * begin() { return vec->FVDouble().Data(); }
  const double * begin() const { return vec->FVDouble().Data(); }

  size_t bytes() const { return size() * sizeof(double); }

  BaseVector & GetBaseVector() { return *vec; }
  const BaseVector & GetBaseVector() const { return *vec; }
};

} // namespace ngsamgcl


// ============================================================
// AMGCL backend specializations
// ============================================================
namespace amgcl {
namespace backend {

// value_type trait
template <>
struct value_type<ngsamgcl::NgsMatrix, void> {
  typedef double type;
};

template <>
struct value_type<ngsamgcl::NgsVector, void> {
  typedef double type;
};

// rows/cols/nonzeros
template <>
struct rows_impl<ngsamgcl::NgsMatrix, void> {
  static size_t get(const ngsamgcl::NgsMatrix & A) { return A.rows(); }
};

template <>
struct cols_impl<ngsamgcl::NgsMatrix, void> {
  static size_t get(const ngsamgcl::NgsMatrix & A) { return A.cols(); }
};

template <>
struct nonzeros_impl<ngsamgcl::NgsMatrix, void> {
  static size_t get(const ngsamgcl::NgsMatrix & A) { return A.nonzeros(); }
};

// Row iterator
template <>
struct row_iterator<ngsamgcl::NgsMatrix, void> {
  typedef ngsamgcl::NgsMatrix::row_iter type;
};

template <>
struct row_begin_impl<ngsamgcl::NgsMatrix, void> {
  static ngsamgcl::NgsMatrix::row_iter get(const ngsamgcl::NgsMatrix & A, size_t row) {
    return A.row_begin(row);
  }
};

// --- spmv: y = alpha * A * x + beta * y ---
// Uses NGSolve's parallel SparseMatrix::MultAdd
template <class Alpha, class Beta>
struct spmv_impl<Alpha, ngsamgcl::NgsMatrix, ngsamgcl::NgsVector, Beta, ngsamgcl::NgsVector, void>
{
  static void apply(Alpha alpha, const ngsamgcl::NgsMatrix & A,
                    const ngsamgcl::NgsVector & x, Beta beta, ngsamgcl::NgsVector & y)
  {
    if (math::is_zero(beta)) {
      y.GetBaseVector() = 0.0;
      A.ngs_mat->MultAdd(alpha, x.GetBaseVector(), y.GetBaseVector());
    } else {
      y.GetBaseVector() *= beta;
      A.ngs_mat->MultAdd(alpha, x.GetBaseVector(), y.GetBaseVector());
    }
  }
};

// --- residual: r = rhs - A*x ---
template <>
struct residual_impl<ngsamgcl::NgsMatrix, ngsamgcl::NgsVector, ngsamgcl::NgsVector, ngsamgcl::NgsVector, void>
{
  static void apply(const ngsamgcl::NgsVector & rhs, const ngsamgcl::NgsMatrix & A,
                    const ngsamgcl::NgsVector & x, ngsamgcl::NgsVector & r)
  {
    r.GetBaseVector() = rhs.GetBaseVector();
    A.ngs_mat->MultAdd(-1.0, x.GetBaseVector(), r.GetBaseVector());
  }
};

// --- clear: x = 0 ---
template <>
struct clear_impl<ngsamgcl::NgsVector, void>
{
  static void apply(ngsamgcl::NgsVector & x) {
    x.GetBaseVector() = 0.0;
  }
};

// --- copy: y = x ---
template <>
struct copy_impl<ngsamgcl::NgsVector, ngsamgcl::NgsVector, void>
{
  static void apply(const ngsamgcl::NgsVector & x, ngsamgcl::NgsVector & y) {
    y.GetBaseVector() = x.GetBaseVector();
  }
};

// --- inner_product: return x·y ---
template <>
struct inner_product_impl<ngsamgcl::NgsVector, ngsamgcl::NgsVector, void>
{
  typedef double return_type;

  static return_type get(const ngsamgcl::NgsVector & x, const ngsamgcl::NgsVector & y) {
    return ngla::InnerProduct<double>(x.GetBaseVector(), y.GetBaseVector());
  }
};

// --- axpby: y = a*x + b*y ---
template <class A, class B>
struct axpby_impl<A, ngsamgcl::NgsVector, B, ngsamgcl::NgsVector, void>
{
  static void apply(A a, const ngsamgcl::NgsVector & x, B b, ngsamgcl::NgsVector & y) {
    if (math::is_zero(b)) {
      y.GetBaseVector() = 0.0;
    } else {
      y.GetBaseVector() *= b;
    }
    y.GetBaseVector().Add(a, x.GetBaseVector());
  }
};

// --- axpbypcz: z = a*x + b*y + c*z ---
template <class A, class B, class C>
struct axpbypcz_impl<A, ngsamgcl::NgsVector, B, ngsamgcl::NgsVector, C, ngsamgcl::NgsVector, void>
{
  static void apply(A a, const ngsamgcl::NgsVector & x,
                    B b, const ngsamgcl::NgsVector & y,
                    C c, ngsamgcl::NgsVector & z) {
    if (math::is_zero(c)) {
      z.GetBaseVector() = 0.0;
    } else {
      z.GetBaseVector() *= c;
    }
    z.GetBaseVector().Add(a, x.GetBaseVector());
    z.GetBaseVector().Add(b, y.GetBaseVector());
  }
};

// --- vmul: z = alpha * x * y + beta * z (element-wise / Hadamard product) ---
template <class Alpha, class Beta>
struct vmul_impl<Alpha, ngsamgcl::NgsVector, ngsamgcl::NgsVector, Beta, ngsamgcl::NgsVector, void>
{
  static void apply(Alpha a, const ngsamgcl::NgsVector & x,
                    const ngsamgcl::NgsVector & y,
                    Beta b, ngsamgcl::NgsVector & z) {
    auto fvx = x.GetBaseVector().FVDouble();
    auto fvy = y.GetBaseVector().FVDouble();
    auto fvz = z.GetBaseVector().FVDouble();
    const auto n = fvz.Size();
    if (!math::is_zero(b)) {
      ngcore::ParallelForRange(n, [&](auto range) {
        for (auto i : range)
          fvz[i] = a * fvx[i] * fvy[i] + b * fvz[i];
      });
    } else {
      ngcore::ParallelForRange(n, [&](auto range) {
        for (auto i : range)
          fvz[i] = a * fvx[i] * fvy[i];
      });
    }
  }
};

// ============================================================
// Backend struct
// ============================================================
struct ngsolve_backend {
  typedef double      value_type;
  typedef int         index_type;
  typedef int         col_type;
  typedef int         ptr_type;
  typedef double      rhs_type;

  struct provides_row_iterator : std::true_type {};

  typedef ngsamgcl::NgsMatrix                matrix;
  typedef ngsamgcl::NgsVector                vector;
  typedef ngsamgcl::NgsVector                matrix_diagonal;
  typedef ngsamgcl::NgsDirectSolver          direct_solver;

  typedef amgcl::detail::empty_params params;

  static std::string name() { return "ngsolve"; }

  // Convert AMGCL's CRS to our NgsMatrix (builds NGSolve SparseMatrix)
  static std::shared_ptr<matrix>
  copy_matrix(std::shared_ptr<matrix> A, const params&) {
    return A;
  }

  // Create NgsMatrix from builtin CRS
  template <class T, class C, class P>
  static std::shared_ptr<matrix>
  copy_matrix(std::shared_ptr<crs<T, C, P>> A, const params&) {
    return std::make_shared<matrix>(*A);
  }

  static std::shared_ptr<vector>
  copy_vector(const std::vector<double> & x, const params&) {
    return std::make_shared<vector>(x);
  }

  template <class T>
  static std::shared_ptr<vector>
  copy_vector(std::shared_ptr<T> x, const params&) {
    // Generic: copy from any indexable vector type
    auto n = x->size();
    auto v = std::make_shared<vector>(n);
    for (size_t i = 0; i < n; ++i)
      (*v)[i] = (*x)[i];
    return v;
  }

  static std::shared_ptr<vector>
  create_vector(size_t size, const params&) {
    return std::make_shared<vector>(size);
  }

  struct gather {
    std::vector<ptrdiff_t> I;
    gather(size_t, const std::vector<ptrdiff_t> & I, const params&) : I(I) {}

    template <class InVec, class OutVec>
    void operator()(const InVec & vec, OutVec & vals) const {
      for (size_t i = 0; i < I.size(); ++i)
        vals[i] = vec[I[i]];
    }
  };

  struct scatter {
    std::vector<ptrdiff_t> I;
    scatter(size_t, const std::vector<ptrdiff_t> & I, const params&) : I(I) {}

    template <class InVec, class OutVec>
    void operator()(const InVec & vals, OutVec & vec) const {
      for (size_t i = 0; i < I.size(); ++i)
        vec[I[i]] = vals[i];
    }
  };

  static std::shared_ptr<direct_solver>
  create_solver(std::shared_ptr<matrix> A, const params&) {
    return std::make_shared<direct_solver>(*A);
  }

  // Overload for when AMGCL passes the internal CRS directly
  template <class T, class C, class P>
  static std::shared_ptr<direct_solver>
  create_solver(std::shared_ptr<crs<T, C, P>> A, const params&) {
    // Convert CRS to NgsMatrix first, then create solver
    ngsamgcl::NgsMatrix ngs_a(*A);
    return std::make_shared<direct_solver>(ngs_a);
  }
};

} // namespace backend
} // namespace amgcl

#endif // NGSOLVE_AMGCL_BACKEND_HPP
