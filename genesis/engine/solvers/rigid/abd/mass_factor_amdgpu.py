"""AMDGPU-only register-resident mass factor, fused with the smooth-acceleration solve.

The scalar arm of func_factor_mass (one thread per (tree, env), taken where the envs alone saturate the GPU) factors
each mass block by a global read-modify-write per inner elimination step, and func_compute_qacc then reloads the factor
one dependent element at a time on one thread per mass block, launched over every (dof, env) pair. On AMDGPU, where the
largest mass block is a small compile-time constant, each block is instead loaded once into registers (every index
static), factored there, written once, and, on the smooth forward path, its smooth acceleration is solved right away
against the factor still in registers.

Every value is computed by the same expressions in the same order as the generic path: the elimination order of
func_factor_mass_tree, and the substitutions of func_solve_mass_block (each accumulator receives its terms in the order
the mass-parent walks give them). Rows past the block pad the triangle with the identity, and the entries of a
substitution off the mass-parent chain are exact zeros of the factor, so the extra static steps only subtract exact
zeros. The substitution terms are applied through a select, which keeps each sum in order under reassociation.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidInfo

from .misc import func_is_awake_tree

# Largest mass block (in dofs) whose lower triangle stays register-resident on one lane.
REGISTER_FACTOR_MAX_DOFS = 24


def register_factor_cap(rigid_config, mass_mat_L):
    """Static row count of the register-resident block factor, or 0 to keep the generic scalar path."""
    if rigid_config.backend != gs.amdgpu or rigid_config.requires_grad:
        return 0
    if rigid_config.enable_tiled_cholesky_mass_matrix or rigid_config.enable_register_tiled_mass:
        return 0
    if not rigid_config.mass_matrix_fits_shared:
        return 0
    shape = getattr(mass_mat_L, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        return 0
    n_dofs = shape[0]
    if not isinstance(n_dofs, int) or n_dofs <= 0:
        return 0
    # Every mass block fits the shared tile size (mass_matrix_fits_shared), and holds at most every dof.
    cap = min(n_dofs, int(rigid_config.tiled_n_dofs_per_block))
    if cap <= 0 or cap > REGISTER_FACTOR_MAX_DOFS:
        return 0
    return cap


def fused_solve_cap(rigid_config, mass_mat_L, is_backward):
    """Static row count of the fused factor + smooth-acceleration solve, or 0 for the generic passes."""
    if is_backward or rigid_config.use_hibernation or rigid_config.enable_fused_smooth_acc_solve:
        return 0
    return register_factor_cap(rigid_config, mass_mat_L)


@qd.func
def func_factor_mass_block_regs_amdgpu(
    i_b: int,
    block_start: int,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    solve_acc: qd.template(),
    CAP: qd.template(),
):
    """Factor one mass block of one env register-resident, then optionally solve its smooth acceleration."""
    n_block = rigid_info.dofs_mass_block_end[block_start] - block_start

    # Every load ahead of every store.
    u = qd.Matrix.zero(gs.qd_float, CAP, CAP)
    for i_ in qd.static(range(CAP)):
        if i_ < n_block:
            i_d = block_start + i_
            for j_ in qd.static(range(i_ + 1)):
                u[i_, j_] = rigid_info.mass_mat[i_d, block_start + j_, i_b]
            if qd.static(implicit_damping):
                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                u[i_, i_] = u[i_, i_] + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                    if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                        u[i_, i_] = u[i_, i_] - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
        else:
            u[i_, i_] = 1.0

    out = qd.Vector.zero(gs.qd_float, CAP)
    if qd.static(solve_acc):
        for i_ in qd.static(range(CAP)):
            if i_ < n_block:
                out[i_] = dyn_state.dofs.force[block_start + i_, i_b]

    # In-place LDL^T, eliminating dofs from last to first (func_factor_mass_tree).
    d_inv = qd.Vector.zero(gs.qd_float, CAP)
    for i_rev in qd.static(range(CAP)):
        i_ = qd.static(CAP - 1 - i_rev)
        D_inv = 1.0 / u[i_, i_]
        d_inv[i_] = D_inv
        for j_rev in qd.static(range(i_)):
            j_ = qd.static(i_ - 1 - j_rev)
            a = u[i_, j_] * D_inv
            for k_ in qd.static(range(j_ + 1)):
                u[j_, k_] = u[j_, k_] - a * u[i_, k_]
            u[i_, j_] = a

    for i_ in qd.static(range(CAP)):
        if i_ < n_block:
            i_d = block_start + i_
            rigid_info.mass_mat_D_inv[i_d, i_b] = d_inv[i_]
            for j_ in qd.static(range(i_)):
                rigid_info.mass_mat_L[i_d, block_start + j_, i_b] = u[i_, j_]
            rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0

    if qd.static(solve_acc):
        # Step 1: L^T w = force. Row i (from the last) scatters its final value to the rows above it, so each row
        # receives its terms from the rows below it, from the last to the first.
        for i_rev in qd.static(range(CAP)):
            i_ = qd.static(CAP - 1 - i_rev)
            w = out[i_]
            is_row = i_ < n_block
            for j_rev in qd.static(range(i_)):
                j_ = qd.static(i_ - 1 - j_rev)
                out[j_] = qd.select(is_row, out[j_] - u[i_, j_] * w, out[j_])
        # Step 2: z = D^-1 w
        for i_ in qd.static(range(CAP)):
            out[i_] = out[i_] * d_inv[i_]
        # Step 3: L acc = z, each row walking its ancestors from the nearest one.
        for i_ in qd.static(range(CAP)):
            curr_out = out[i_]
            is_row = i_ < n_block
            for j_rev in qd.static(range(i_)):
                j_ = qd.static(i_ - 1 - j_rev)
                curr_out = qd.select(is_row, curr_out - u[i_, j_] * out[j_], curr_out)
            out[i_] = curr_out
        for i_ in qd.static(range(CAP)):
            if i_ < n_block:
                dyn_state.dofs.acc_smooth[block_start + i_, i_b] = out[i_]
                dyn_state.dofs.acc[block_start + i_, i_b] = out[i_]


@qd.func
def func_factor_mass_tree_regs_amdgpu(
    i_t: int,
    i_b: int,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    solve_acc: qd.template(),
    CAP: qd.template(),
):
    """func_factor_mass_tree, register-resident per mass block (AMDGPU only, see register_factor_cap)."""
    # The implicit damping factor keeps the generic path (see func_factor_mass), so only the smooth factor comes here.
    qd.static_assert(not implicit_damping)
    if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
        tree_dof_start = rigid_info.trees_dof_start[i_t]
        tree_dof_end = tree_dof_start + rigid_info.trees_n_dofs[i_t]
        for block_start in range(tree_dof_start, tree_dof_end):
            if rigid_info.dofs_mass_block_start[block_start] == block_start:
                func_factor_mass_block_regs_amdgpu(
                    i_b,
                    block_start,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                    implicit_damping,
                    solve_acc,
                    CAP,
                )
