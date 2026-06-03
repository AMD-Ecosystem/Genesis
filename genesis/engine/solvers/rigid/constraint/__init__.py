"""
Constraint solver submodule for rigid body simulation.

Contains constraint solving, island detection, and backward pass.
"""

from .solver import ConstraintSolver
from .solver_island import ConstraintSolverIsland

# first declare func_solve_body:
from . import solver

# now register decomposed with func_solve_body:
from . import solver_breakdown

# AMDGPU-specific variants (B3 split, B4 lifted-loop) of func_solve_body.
# Temporarily disabled after the upstream-v1.0.0 merge: upstream added a
# `dofs_info` argument to func_solve_body, so the fork's 6-arg variants fail
# perf_dispatch registration (PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH).
# Re-enable once the AMD variants in solver_amdgpu.py are ported to the new
# func_solve_body signature. Opt back in with GS_ENABLE_AMDGPU_SOLVER_VARIANTS=1.
import os as _os

if _os.environ.get("GS_ENABLE_AMDGPU_SOLVER_VARIANTS", "0") == "1":
    from . import solver_amdgpu  # noqa: F401
