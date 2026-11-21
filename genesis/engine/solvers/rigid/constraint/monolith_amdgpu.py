"""AMD-only register-resident pieces of the per-env monolith Newton iteration (see solver.func_solve_iter).

The generic path walks two per-iteration chains through global memory, one env per thread:
- the gradient solve (func_update_gradient_batch -> func_cholesky_solve_batch): every row stores its Mgrad entry and
  every later row reads it back, 2 n_dofs dependent global round trips per iteration;
- the incremental factor (func_build_changed_constraint_list + func_rank1_flip_dense_block): every flipped row stages
  its working vector in nt_vec and every pivot of the rank-1 update reads it back and rewrites it.
On gfx9-class AMDGPU a load also waits for every store issued before it (one vmcnt), so each of those round trips is a
full memory latency, which the monolith at one wave per SIMD cannot hide. On a single-island scene whose one island is
the dense block at offset 0, with a static dof count, the loops are unrolled here over the dof count: the working
vector, the gradient and the running solution stay in registers; the factor is read (and, for the rank-1 update,
written) with constant offsets. The operations and their order are the generic path's; each substitution entry is held
apart (func_hold) so fast-math reassociation cannot regroup the sums.

The flag is decided once, at kernel compile time, from the static rigid config and the static shape of nt_H: every other
backend, layout and island structure keeps the generic calls.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.data import DynState, RigidInfo

from .data import ConstraintState

# Largest dof count whose solve is unrolled into registers.
MONO_SOLVE_MAX_DOFS = 24
# Largest dof count whose M @ search is unrolled: the generic path's mass-block loop runs scalar below the vectorizer's
# threshold (observed 18 to 28 entries on gfx950 depending on the kernel), so the unrolled sequential sum is its sum.
MV_SCALAR_MAX_DOFS = 17


def mono_solve_cap(rigid_config, nt_H):
    """Static dof count of the register-resident gradient solve / rank-1 update, or 0 to keep the generic path's."""
    if rigid_config.backend != gs.amdgpu or rigid_config.requires_grad:
        return 0
    if rigid_config.solver_type != gs.constraint_solver.Newton:
        return 0
    if not rigid_config.is_single_island or rigid_config.sparse_solve:
        return 0
    if rigid_config.enable_signorini_contact:
        return 0
    shape = getattr(nt_H, "shape", None)
    if shape is None or len(shape) != 3:
        return 0
    n_dofs = shape[1]
    if not isinstance(n_dofs, int) or n_dofs <= 0 or n_dofs > MONO_SOLVE_MAX_DOFS or shape[2] != n_dofs:
        return 0
    if gs.qd_float != qd.f32:
        return 0
    return n_dofs


def mono_rank1_cap(rigid_config, nt_H):
    """Static dof count of the register-resident incremental factor update.

    0 where the elliptic cone refactors on GPU.
    """
    if rigid_config.enable_elliptic_friction:
        return 0
    return mono_solve_cap(rigid_config, nt_H)


@qd.func
def func_hold(x, zero: int):
    """x unchanged (zero is 0), through an integer op the compiler cannot see through: it ends a floating-point chain
    so fast-math reassociation cannot regroup the substitution sums across entries."""
    return qd.bit_cast(qd.bit_cast(x, qd.i32) ^ zero, qd.f32)


@qd.func
def func_opaque_zero(i_b: int, constraint_state: ConstraintState):
    """0, computed from a count loaded from memory that no loop or branch bounds (the equality-row count is never
    negative), so the compiler can neither fold it nor any value held (func_hold) or address offset by it."""
    return qd.cast(constraint_state.n_constraints_equality[i_b] < 0, qd.i32)


@qd.func
def func_is_dense_single_block(i_b: int, constraint_state: ConstraintState):
    """Whether the env's gradient solve is the dense block at offset 0 of its one island."""
    return constraint_state.island.n_islands[i_b] == 1 and constraint_state.island.dof_range_start[0, i_b] == 0


@qd.func
def func_gradient_solve_regs(
    i_b: int,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_config: qd.template(),
    N: qd.template(),
):
    """func_update_gradient_batch of an env whose one island is the dense block at offset 0: the gradient, then the
    dense-block branch of func_cholesky_solve_batch, unrolled over the static dof count, the gradient and the running
    solution in registers.

    Every dof index is offset by the block's start dof_range_start (checked 0 by the caller), read back here so the
    compiler cannot treat the env's addresses as loop-invariant across the iterations and hoist them all."""
    # Forward-substitution rows with at least this many entries are summed as the generic path's compiled loop sums them
    # on AMDGPU: its inner loop (curr_out -= L[i, j] * out[j], j ascending, contiguous L) is vectorized by two lanes
    # from that trip count on (v_pk_fma_f32 over two partial sums, joined by one add, then the scalar tail); shorter
    # rows run scalar.
    FWD_PAIRED_MIN_ENTRIES = qd.static(14)
    if constraint_state.island.improved[0, i_b]:
        # dof_range_start is 0 here (checked by the caller), which the compiler knows: the holds take an opaque 0
        d0 = constraint_state.island.dof_range_start[0, i_b] + func_opaque_zero(i_b, constraint_state)
        grad = qd.Vector.zero(gs.qd_float, N)
        mgrad = qd.Vector.zero(gs.qd_float, N)
        g_ma = qd.Vector.zero(gs.qd_float, N)
        g_qf = qd.Vector.zero(gs.qd_float, N)
        g_qc = qd.Vector.zero(gs.qd_float, N)
        for i_d in qd.static(range(N)):
            g_ma[i_d] = constraint_state.Ma[d0 + i_d, i_b]
            g_qf[i_d] = dyn_state.dofs.qf_smooth[d0 + i_d, i_b]
            g_qc[i_d] = constraint_state.qfrc_constraint[d0 + i_d, i_b]
        for i_d in qd.static(range(N)):
            grad[i_d] = g_ma[i_d] - g_qf[i_d] - g_qc[i_d]
            constraint_state.grad[d0 + i_d, i_b] = grad[i_d]
        # forward substitution
        for i_d in qd.static(range(N)):
            curr_out = grad[i_d]
            if qd.static(rigid_config.enable_jacobi_equilibration):
                curr_out = curr_out * constraint_state.nt_jacobi[d0 + i_d, i_b]
            if qd.static(i_d >= FWD_PAIRED_MIN_ENTRIES):
                # The generic path's row as the compiler vectorizes it: two interleaved partial sums (the even entries
                # onto the running value, the odd ones onto zero) joined, then the odd tail entry.
                acc_odd = gs.qd_float(0.0)
                for p in qd.static(range(i_d // 2)):
                    curr_out = func_hold(curr_out, d0)
                    curr_out = curr_out - constraint_state.nt_H[i_b, d0 + i_d, d0 + 2 * p] * mgrad[2 * p]
                    acc_odd = func_hold(acc_odd, d0)
                    acc_odd = acc_odd - constraint_state.nt_H[i_b, d0 + i_d, d0 + 2 * p + 1] * mgrad[2 * p + 1]
                curr_out = func_hold(curr_out, d0) + func_hold(acc_odd, d0)
                if qd.static(i_d % 2 == 1):
                    curr_out = func_hold(curr_out, d0)
                    curr_out = curr_out - constraint_state.nt_H[i_b, d0 + i_d, d0 + i_d - 1] * mgrad[i_d - 1]
            else:
                for j_d in qd.static(range(i_d)):
                    curr_out = func_hold(curr_out, d0)
                    curr_out = curr_out - constraint_state.nt_H[i_b, d0 + i_d, d0 + j_d] * mgrad[j_d]
            mgrad[i_d] = func_hold(curr_out / constraint_state.nt_H[i_b, d0 + i_d, d0 + i_d], d0)
        # backward substitution
        for i_rev in qd.static(range(N)):
            i_d = qd.static(N - 1 - i_rev)
            curr_out = mgrad[i_d]
            for j_d in qd.static(range(i_d + 1, N)):
                curr_out = func_hold(curr_out, d0)
                curr_out = curr_out - constraint_state.nt_H[i_b, d0 + j_d, d0 + i_d] * mgrad[j_d]
            mgrad[i_d] = func_hold(curr_out / constraint_state.nt_H[i_b, d0 + i_d, d0 + i_d], d0)
        if qd.static(rigid_config.enable_jacobi_equilibration):
            for i_d in qd.static(range(N)):
                mgrad[i_d] = mgrad[i_d] * constraint_state.nt_jacobi[d0 + i_d, i_b]
        for i_d in qd.static(range(N)):
            constraint_state.Mgrad[d0 + i_d, i_b] = mgrad[i_d]


@qd.func
def func_rank1_flip_dense_regs(
    i_b: int,
    i_c: int,
    zero: int,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    N: qd.template(),
) -> bool:
    """solver.func_rank1_flip_dense_block over the whole env (block [0, N)), the working vector in registers instead of
    nt_vec: same operations, same order. Returns True on a non-positive downdate pivot."""
    sign = 1.0 if constraint_state.active[i_c, i_b] else -1.0
    efc_D_sqrt = qd.sqrt(constraint_state.efc_D[i_c, i_b])
    vec = qd.Vector.zero(gs.qd_float, N)
    for i_d in qd.static(range(N)):
        v = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt
        vec[i_d] = v
    # Still 0, but now computed from this row's loads: every factor address below depends on the row, so the compiler
    # cannot hoist the N^2 / 2 addresses of the env's factor out of the loop over the flipped rows (which spills).
    zero = qd.bit_cast(vec[0], qd.i32) & zero
    if qd.static(rigid_config.enable_jacobi_equilibration):
        for i_d in qd.static(range(N)):
            vec[i_d] = vec[i_d] * constraint_state.nt_jacobi[zero + i_d, i_b]
    EPS = rigid_info.EPS[None]
    is_degenerated = False
    for i_d in qd.static(range(N)):
        if not is_degenerated:
            Lkk = constraint_state.nt_H[i_b, zero + i_d, zero + i_d]
            vk = vec[i_d]
            if qd.abs(vk) > EPS * Lkk:
                tmp = Lkk**2 + sign * vk**2
                if tmp < EPS * Lkk**2:
                    is_degenerated = True
                else:
                    r = qd.sqrt(tmp)
                    c = r / Lkk
                    s = vk / Lkk
                    constraint_state.nt_H[i_b, zero + i_d, zero + i_d] = r
                    # As the generic path's loops compile on AMDGPU: s * sign formed once, each entry fused onto it and
                    # divided by c (the product by 1 / c folds into that quotient), the working vector then updated from
                    # the entry as stored (held, so the product by s is not folded into the quotient).
                    s_sign = s * sign
                    # The column is loaded whole before its first entry is stored (a load issued after a store waits
                    # for it on AMDGPU, and the compiler cannot prove the entries apart)
                    L_col = qd.Vector.zero(gs.qd_float, N)
                    for j_d in qd.static(range(i_d + 1, N)):
                        L_col[j_d] = constraint_state.nt_H[i_b, zero + j_d, zero + i_d]
                    for j_d in qd.static(range(i_d + 1, N)):
                        L_new = (L_col[j_d] + s_sign * vec[j_d]) / c
                        constraint_state.nt_H[i_b, zero + j_d, zero + i_d] = L_new
                        L_new = func_hold(L_new, zero)
                        vec[j_d] = vec[j_d] * c - s * L_new
    return is_degenerated


@qd.func
def func_factor_incremental_dense_regs(
    i_b: int,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    N: qd.template(),
) -> bool:
    """func_build_changed_constraint_list + func_hessian_and_cholesky_factor_incremental_dense_batch of an env holding
    one island: the rows whose active state flipped, in increasing order, each folded into the dense factor
    (func_rank1_flip_dense_regs). The changed-row list is scratch read by nothing else, so it is not staged."""
    n_con = constraint_state.n_constraints[i_b]
    zero = func_opaque_zero(i_b, constraint_state)
    is_degenerated = False
    for i_c in range(n_con):
        if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
            if func_rank1_flip_dense_regs(i_b, i_c, zero, constraint_state, rigid_info, rigid_config, N):
                is_degenerated = True
    return is_degenerated


def iter_rebase(rigid_config):
    """Whether each Newton iteration of the monolith addresses its env through func_iter_env (AMDGPU only)."""
    return rigid_config.backend == gs.amdgpu and not rigid_config.requires_grad


@qd.func
def func_iter_env(i_b: int, constraint_state: ConstraintState):
    """i_b, offset by a 0 read from memory inside the iteration.

    Under the batch-last layout every (array, dof) pair of the env is its own 64-bit address (the dof stride is the
    batch size, past any immediate offset), and the compiler hoists all of them out of the iteration loop of the
    monolith, which keeps several hundred registers live across it (one wave per SIMD, and spills once the unrolled
    passes add theirs). Offset by a value it cannot see through, the addresses are recomputed where they are used."""
    return i_b + qd.cast(constraint_state.n_constraints[i_b] < 0, qd.i32)


@qd.func
def func_after(x, zero: int):
    """zero (0), computed from x through an integer op the compiler cannot see through."""
    return qd.bit_cast(x, qd.i32) & zero


@qd.func
def func_mass_blocks_full(i_b: int, rigid_info: RigidInfo, N: qd.template()):
    """Whether every dof's mass block spans the env's N dofs (one kinematic tree holding every dof)."""
    is_full = True
    for i_d in qd.static(range(N)):
        is_full = is_full and rigid_info.dofs_mass_block_start[i_d] == 0 and rigid_info.dofs_mass_block_end[i_d] == N
    return is_full


def mono_mv_cap(rigid_config, nt_H):
    """Static dof count of the register-resident M @ search / J @ search of the line search, or 0."""
    n = mono_update_cap(rigid_config, nt_H)
    return n if n <= MV_SCALAR_MAX_DOFS else 0


def mono_update_cap(rigid_config, nt_H):
    """Static dof count of the register-resident M @ search / J @ search of the line search (func_mv_jv_dense_regs), or
    0 to keep the generic path's (AMDGPU only, pyramidal cone, dense single island, canonical layout)."""
    n = mono_solve_cap(rigid_config, nt_H)
    if n == 0 or rigid_config.enable_elliptic_friction:
        return 0
    if rigid_config.constraint_layout_batch_first or rigid_config.enable_cooperative_constraint_kernels:
        return 0
    return n


@qd.func
def func_mv_jv_dense_regs(i_b: int, constraint_state: ConstraintState, rigid_info: RigidInfo, N: qd.template()):
    """linesearch.func_mv_jv_dense with the search direction loaded once into registers: mv of every dof summed over
    its mass block in increasing column order from zero (the block bounds as predicates over the unrolled columns),
    all mv stored after all are formed, and jv R rows at a time, each summed in increasing dof order
    from zero. Same terms, same order as the generic path."""
    # Rows of J @ search whose loads issue together.
    R = qd.static(2)
    len_constraints = qd.static(constraint_state.Jaref.shape[0])
    s = qd.Vector.zero(gs.qd_float, N)
    for i_d in qd.static(range(N)):
        s[i_d] = constraint_state.search[i_d, i_b]
    zero = func_opaque_zero(i_b, constraint_state)
    # Every mass block spans the env (checked by the caller, func_mass_blocks_full): each row sums every column in
    # increasing order from zero, as the generic path's block loop does (its trip count below the vectorizer's
    # threshold).
    mv = qd.Vector.zero(gs.qd_float, N)
    for i_d1 in qd.static(range(N)):
        acc = gs.qd_float(0.0)
        for i_d2 in qd.static(range(N)):
            acc = func_hold(acc, zero)
            acc = acc + rigid_info.mass_mat[i_d1, i_d2, i_b] * s[i_d2]
        mv[i_d1] = acc
    for i_d in qd.static(range(N)):
        constraint_state.mv[i_d, i_b] = mv[i_d]
    n_con = constraint_state.n_constraints[i_b]
    for i_chunk in range((n_con + R - 1) // R):
        rows_jac = qd.Matrix.zero(gs.qd_float, R, N)
        for k in qd.static(range(R)):
            i_c = qd.min(i_chunk * R + k, len_constraints - 1)
            for i_d in qd.static(range(N)):
                rows_jac[k, i_d] = constraint_state.jac[i_c, i_d, i_b]
        for k in qd.static(range(R)):
            i_c = i_chunk * R + k
            if i_c < n_con:
                jv = gs.qd_float(0.0)
                for i_d in qd.static(range(N)):
                    jv = func_hold(jv, zero)
                    jv = jv + rows_jac[k, i_d] * s[i_d]
                constraint_state.jv[i_c, i_b] = jv
