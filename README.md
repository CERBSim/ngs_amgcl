# NGSolve-AMGCL

An NGSolve addon providing algebraic multigrid preconditioning via [AMGCL](https://github.com/ddemidov/amgcl).

## Features

- AMG preconditioner for NGSolve sparse matrices
- Shared memory parallelism (OpenMP)
- Multiple coarsening strategies: smoothed aggregation, Ruge-Stuben
- Multiple relaxation schemes: SPAI-0, Gauss-Seidel, ILU0, damped Jacobi
- Seamless integration with NGSolve iterative solvers

## Installation

### Quick install

```bash
pip install git+https://github.com/your-org/ngsolve-amgcl.git
```

### Step-by-step (for self-compiled NGSolve)

```bash
git clone --recursive https://github.com/your-org/ngsolve-amgcl.git
cd ngsolve-amgcl
pip install --no-build-isolation .
```

### Manual CMake build

```bash
git clone --recursive https://github.com/your-org/ngsolve-amgcl.git
cd ngsolve-amgcl
mkdir build && cd build
cmake ..
make -j4 install
```

## Usage

```python
from ngsolve import *
import ngsolve_amgcl

# ... set up mesh, FESpace, BilinearForm a ...

# Configure AMGCL options
opts = ngsolve_amgcl.AMGCLOptions()
opts.coarsening = "smoothed_aggregation"
opts.relaxation = "spai0"

# Create preconditioner
pre = ngsolve_amgcl.AMGCLPreconditioner(a.mat, fes.FreeDofs(), opts)

# Use in a CG solver
from ngsolve.krylovspace import CGSolver
inv = CGSolver(mat=a.mat, pre=pre, printrates=True, tol=1e-8)
gfu.vec.data = inv * f.vec
```

## Supported Options

### Coarsening strategies
- `"smoothed_aggregation"` (default)
- `"ruge_stuben"`

### Relaxation schemes
- `"spai0"` (default) - Sparse Approximate Inverse
- `"gauss_seidel"` - Gauss-Seidel
- `"ilu0"` - Incomplete LU factorization
- `"damped_jacobi"` - Damped Jacobi

## License

MIT
