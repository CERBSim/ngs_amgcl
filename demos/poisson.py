"""
Poisson equation solved with NGSolve using AMGCL as preconditioner.

This demo solves:
  -Delta u = f  on the unit square
         u = 0  on the boundary

using a CG solver with AMGCL AMG preconditioning.
The preconditioner takes a BilinearForm and automatically rebuilds
when the form is reassembled.
"""

from ngsolve import *
from ngsolve.la import EigenValues_Preconditioner
from netgen.geom2d import unit_square
import ngsolve_amgcl

# Generate mesh
mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))

# Define finite element space
fes = H1(mesh, order=1, dirichlet=".*")
print(f"Number of DOFs: {fes.ndof}")
print(f"Number of free DOFs: {fes.FreeDofs().NumSet()}")

# Define bilinear and linear forms
u, v = fes.TnT()
a = BilinearForm(fes)
a += grad(u) * grad(v) * dx

f = LinearForm(fes)
f += 1 * v * dx

# Create AMGCL preconditioner BEFORE assembly
# (it registers with the BilinearForm for auto-update)
opts = ngsolve_amgcl.AMGCLOptions()
opts.coarsening = "smoothed_aggregation"
opts.relaxation = "spai0"
opts.coarse_enough = 100
print(f"\nAMGCL options: {opts}")

pre = ngsolve_amgcl.AMGCLPreconditioner(a, options=opts)

# Assemble triggers preconditioner build automatically
print("\nAssembling (preconditioner builds automatically)...")
a.Assemble()
f.Assemble()

# Create solution vector
gfu = GridFunction(fes)

# Solve with CG
print("\nSolving with CG...")
from ngsolve.krylovspace import CGSolver
inv = CGSolver(mat=a.mat, pre=pre, printrates=True, maxiter=200, tol=1e-8)
gfu.vec.data = inv * f.vec

print(f"\nCG converged in {inv.iterations} iterations")

# Optional: compute eigenvalue estimates of the preconditioned system
try:
    lams = EigenValues_Preconditioner(mat=a.mat, pre=pre, tol=1e-3)
    print(f"Condition number estimate: {max(lams)/min(lams):.2f}")
except:
    pass

# ---------------------------------------------------------------
# Demonstrate automatic update on reassembly
# ---------------------------------------------------------------
print("\n" + "="*60)
print("Demonstrating auto-update after reassembly...")
print("="*60)

# Change the bilinear form to include a reaction term
a2 = BilinearForm(fes)
a2 += grad(u) * grad(v) * dx + 10 * u * v * dx

# Preconditioner created before assembly -> auto-updates on Assemble()
pre2 = ngsolve_amgcl.AMGCLPreconditioner(a2, options=opts)
a2.Assemble()

inv2 = CGSolver(mat=a2.mat, pre=pre2, printrates=True, maxiter=200, tol=1e-8)
gfu.vec.data = inv2 * f.vec
print(f"\nCG converged in {inv2.iterations} iterations (with reaction term)")

# Reassemble the same form (simulating e.g. time-stepping or nonlinear iteration)
# The preconditioner automatically rebuilds
print("\nReassembling same form (auto-update)...")
a2.Assemble()

inv3 = CGSolver(mat=a2.mat, pre=pre2, printrates=True, maxiter=200, tol=1e-8)
gfu.vec.data = inv3 * f.vec
print(f"\nCG converged in {inv3.iterations} iterations (after reassembly, no manual Update needed)")
