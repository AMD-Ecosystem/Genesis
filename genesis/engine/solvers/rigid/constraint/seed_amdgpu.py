"""AMD-only env-minor Hessian assembly and LDS-staged per-env solve seed of a single-island scene above the cooperative
bound (has_scalar_seed_factor).

Without this, the solve init runs two kernels for this seed: func_island_hessian_assemble_all assembles each env's
Hessian block M + J^T D J into nt_H with one 128-lane block per env (most lanes idle on a block of a few dozen dofs),
then the per-env seed (func_solve_seed_batch) factors nt_H in place one env per lane. nt_H is env-leading, (_B, n, n),
so the 32 lanes of the seed walk 32 blocks one n^2 span apart: every dependent read of the O(n^3) factor is an
uncoalesced global round trip.

Here:
- the assembly (func_assemble_env) runs env-minor, 4 lanes per env and 64 consecutive envs per wave: jac,
  efc_D, active and mass_mat are batch-last, so the envs of a row are consecutive words. Every entry of the lower
  triangle is owned by one lane and summed in registers with the exact terms and order of the generic path (the mass
  entry, then per tile of 32 constraint rows the sum, from zero, of the products of the tile's active rows in
  constraint order, added when the tile has an active row; see func_island_hessian_assemble_block);
- the seed runs 32 consecutive envs per wave64 block, two lanes per env: their blocks are one contiguous span of nt_H,
  whose strictly lower triangles the lanes copy into shared memory with coalesced reads; the lanes factor in shared
  memory (func_cholesky_factor_direct_batch, same operations and order, each lane updating the rows it owns, one
  block barrier per column), copy the factor back with coalesced writes, and the gradient solve of each env reads its
  factor from shared memory, the gradient and Mgrad in registers. The diagonal stays in nt_H, which keeps the tile of
  32 strictly lower triangles within an eighth of a CU's shared memory up to 18 dofs.

The flag is decided once, at kernel compile time, from the static rigid config and the static shape of nt_H: every
other backend, layout and island structure keeps the generic path.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.data import DynState, RigidInfo

from .data import ConstraintState

# Largest dof count of the seed (32 strictly lower triangles of n (n - 1) / 2 floats per block).
SEED_MAX_DOFS = 24
# Smallest dof count of the seed: the env-minor assembly cuts the lower triangle into 4 shares (one lane each, see
# func_solve_init), each spanning two rows at least.
SEED_MIN_DOFS = 8


def _static_n_dofs(nt_H):
    shape = getattr(nt_H, "shape", None)
    try:
        shape = tuple(shape)
    except TypeError:
        return 0
    if len(shape) != 3:
        return 0
    n = shape[1]
    if not isinstance(n, int) or isinstance(n, bool) or shape[2] != n:
        return 0
    return n


def seed_fused_cap(rigid_config, nt_H):
    """Static dof count of the env-minor assembly + LDS seed, or 0 to keep the generic assembly + per-env seed."""
    if rigid_config.backend != gs.amdgpu or rigid_config.requires_grad:
        return 0
    if rigid_config.solver_type != gs.constraint_solver.Newton:
        return 0
    if not (rigid_config.enable_tiled_island_seed and rigid_config.has_scalar_seed_factor):
        return 0
    if not rigid_config.has_per_env_solve_seed:
        return 0
    if not rigid_config.is_single_island or rigid_config.sparse_solve:
        return 0
    if rigid_config.enable_cooperative_constraint_kernels or rigid_config.constraint_layout_batch_first:
        return 0
    if rigid_config.para_level < gs.PARA_LEVEL.ALL or rigid_config.enable_mujoco_compatibility:
        return 0
    if rigid_config.enable_elliptic_friction or rigid_config.enable_signorini_contact:
        return 0
    if rigid_config.enable_jacobi_equilibration or rigid_config.enable_cone_free_hessian_reuse:
        return 0
    if rigid_config.has_island_above_tile_cap:
        return 0
    n_dofs = _static_n_dofs(nt_H)
    if n_dofs < SEED_MIN_DOFS or n_dofs > SEED_MAX_DOFS:
        return 0
    return n_dofs


def tri_stride(n):
    """Shared-memory words per env: the packed strictly lower triangle, padded to an odd count (bank-conflict free)."""
    t = n * (n - 1) // 2
    return t if t % 2 == 1 else t + 1


def _owned_lower(n, sub, k):
    """Static list of the (i, j) lower-triangle entries (diagonal included, j <= i) of share sub of k: the triangle cut
    into k even contiguous shares in row order, so a share reads only the Jacobian columns of its rows' dofs."""
    entries = [(i, j) for i in range(n) for j in range(i + 1)]
    lo = (len(entries) * sub) // k
    hi = (len(entries) * (sub + 1)) // k
    return entries[lo:hi]


def _owned_dofs(n, sub, k):
    """Dofs whose Jacobian column the entries of share sub read."""
    used = set()
    for i, j in _owned_lower(n, sub, k):
        used.add(i)
        used.add(j)
    return sorted(used)


@qd.func
def func_tri_row(t: int) -> int:
    """Row i - 1 of entry t of a packed strictly lower triangle (t = i (i - 1) / 2 + j, 0 <= j < i)."""
    i = qd.cast((qd.sqrt(qd.cast(8 * t + 1, qd.f32)) - 1.0) * 0.5, qd.i32)
    if ((i + 1) * (i + 2)) // 2 <= t:
        i = i + 1
    if (i * (i + 1)) // 2 > t:
        i = i - 1
    return i


# ------------------------------------------------------------------------------------------------- env-minor assembly


@qd.func
def func_assemble_part(
    i_b: int,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    N: qd.template(),
    SUB: qd.template(),
    K: qd.template(),
):
    """func_island_hessian_assemble_block of the env's one island (single island, dense, staged, no Jacobi scaling)
    for the lower-triangle entries lane SUB of K owns, in registers: each entry starts at the mass entry, then per tile
    of RT constraint rows the sum, from zero, of the products of the tile's active rows in constraint order is
    added to it when the tile has an active row."""
    ENT = qd.static(_owned_lower(N, SUB, K))
    NE = qd.static(len(_owned_lower(N, SUB, K)))
    DOFS = qd.static(_owned_dofs(N, SUB, K))
    # Constraint rows per accumulation tile of the generic assembly (ROW_TILE of func_island_hessian_assemble_all).
    RT = qd.static(32)
    # Constraint rows whose terms the assembly loads together.
    RU = qd.static(2)
    con_base = constraint_state.island.constraint_slices.start[0, i_b]
    con_n = constraint_state.island.constraint_slices.n[0, i_b]
    val = qd.Vector.zero(gs.qd_float, NE)
    for m in qd.static(range(NE)):
        val[m] = rigid_info.mass_mat[qd.static(ENT[m][0]), qd.static(ENT[m][1]), i_b]
    for i_chunk in range((con_n + RT - 1) // RT):
        lo = i_chunk * RT
        hi = qd.min(lo + RT, con_n)
        acc = qd.Vector.zero(gs.qd_float, NE)
        n_active = 0
        for i_grp in range((hi - lo + RU - 1) // RU):
            # The terms of RU rows are loaded ahead of their activity tests (a row past the tile clamped into it and
            # dropped), so their round trips overlap; the rows are then summed in constraint order.
            r0 = lo + i_grp * RU
            act = qd.Vector.zero(gs.qd_int, RU)
            dd = qd.Vector.zero(gs.qd_float, RU)
            jr = qd.Matrix.zero(gs.qd_float, RU, N)
            for u in qd.static(range(RU)):
                i_c = con_base + qd.min(r0 + u, hi - 1)
                act[u] = qd.cast(constraint_state.active[i_c, i_b], qd.i32)
                dd[u] = constraint_state.efc_D[i_c, i_b]
                for i_d in qd.static(DOFS):
                    jr[u, i_d] = constraint_state.jac[i_c, i_d, i_b]
            for u in qd.static(range(RU)):
                if r0 + u < hi and act[u] != 0:
                    weight = qd.sqrt(dd[u])
                    jw = qd.Vector.zero(gs.qd_float, N)
                    for i_d in qd.static(DOFS):
                        jw[i_d] = jr[u, i_d] * weight
                    for m in qd.static(range(NE)):
                        acc[m] = acc[m] + jw[qd.static(ENT[m][0])] * jw[qd.static(ENT[m][1])]
                    n_active = n_active + 1
        if n_active > 0:
            for m in qd.static(range(NE)):
                val[m] = val[m] + acc[m]
    for m in qd.static(range(NE)):
        constraint_state.nt_H[i_b, qd.static(ENT[m][0]), qd.static(ENT[m][1])] = val[m]


@qd.func
def func_assemble_env(
    i_b: int,
    sub: int,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    N: qd.template(),
    K: qd.template(),
):
    """The generic Hessian assembly of env i_b (func_island_hessian_assemble_all, single island), share sub of K."""
    # The env's flags are loaded together, ahead of the tests they feed.
    n_con = constraint_state.n_constraints[i_b]
    env_improved = constraint_state.improved[i_b]
    island_improved = constraint_state.island.improved[0, i_b]
    if n_con > 0 and env_improved:
        if island_improved:
            for s in qd.static(range(K)):
                if sub == s:
                    func_assemble_part(i_b, constraint_state, rigid_info, N, s, K)


# ------------------------------------------------------------------------------------------------------- LDS seed


@qd.func
def func_stage_block(
    _B: int,
    b0: int,
    tid: int,
    sh,
    constraint_state: ConstraintState,
    N: qd.template(),
    to_shared: qd.template(),
    E: qd.template(),
    BL: qd.template(),
):
    """Copy the strictly lower triangles of the (at most E) consecutive envs from b0 between nt_H and sh, env e of
    the block owning sh[e * S + i (i - 1) / 2 + j] (j < i), the BL lanes walking the packed triangles of the block in
    order: the shared-memory side is contiguous and the nt_H side runs along the rows of one contiguous span of nt_H,
    coalesced. The reads of a pass are issued together, unconditionally (an index past the block is clamped into it
    and its value dropped), so their latencies overlap."""
    T = qd.static((N * (N - 1)) // 2)
    S = qd.static(tri_stride(N))
    # Entries each lane moves per staging pass.
    U = qd.static(16)
    n_env = qd.min(E, _B - b0)
    total = n_env * T
    q0 = tid
    while q0 < total:
        v = qd.Vector.zero(gs.qd_float, U)
        for u in qd.static(range(U)):
            q = qd.min(q0 + u * BL, total - 1)
            e = q // T
            t = q - e * T
            if qd.static(to_shared):
                r = func_tri_row(t)
                v[u] = constraint_state.nt_H[b0 + e, r + 1, t - (r * (r + 1)) // 2]
            else:
                v[u] = sh[e * S + t]
        for u in qd.static(range(U)):
            q = q0 + u * BL
            if q < total:
                e = q // T
                t = q - e * T
                if qd.static(to_shared):
                    sh[e * S + t] = v[u]
                else:
                    r = func_tri_row(t)
                    constraint_state.nt_H[b0 + e, r + 1, t - (r * (r + 1)) // 2] = v[u]
        q0 = q0 + U * BL


@qd.func
def func_prefetch_starts(i_b: int, active: int, constraint_state: ConstraintState, N: qd.template()):
    """The envelope starts of the env's one island (dof_env_start_local), loaded by every lane with the env index
    clamped into the batch, on the guess that the island's dofs start at 0 (one round trip instead of two dependent
    ones), ahead of the block's staging. A guess that misses is reloaded at the real offset."""
    starts = qd.Vector.zero(gs.qd_int, N)
    i_bc = qd.min(i_b, constraint_state.n_constraints.shape[0] - 1)
    dof_base = constraint_state.island.dof_slices.start[0, i_bc]
    for i_d in qd.static(range(N)):
        starts[i_d] = constraint_state.island.dof_env_start_local[i_d, i_bc]
    if active:
        if dof_base != 0:
            for i_d in qd.static(range(N)):
                starts[i_d] = constraint_state.island.dof_env_start_local[dof_base + i_d, i_b]
    return starts


@qd.func
def func_prefetch_diag(i_b: int, constraint_state: ConstraintState, N: qd.template()):
    """The diagonal of the env's Hessian block, loaded with the env index clamped into the batch."""
    hds = qd.Vector.zero(gs.qd_float, N)
    i_bc = qd.min(i_b, constraint_state.n_constraints.shape[0] - 1)
    for i_d in qd.static(range(N)):
        hds[i_d] = constraint_state.nt_H[i_bc, i_d, i_d]
    return hds


@qd.func
def func_factor_split(
    i_b: int,
    sh,
    base: int,
    sub: int,
    active: int,
    starts,
    hds,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    N: qd.template(),
    K: qd.template(),
):
    """func_cholesky_factor_direct_batch of the env's one island (single island: i_dg = i_d) shared by the K lanes of
    env i_b, the strictly lower triangle at sh[base + i (i - 1) / 2 + j] and the diagonal in nt_H: every lane computes
    every diagonal itself (the same operations on the same values) and updates the rows j_d % K == sub it owns, every
    dot product in the generic order and term set. Row i_d is final after step i_d - 1, so one block barrier per step
    (outside the per-env branch: every lane of the block runs all N steps) publishes the column updates. The loops
    are unrolled over the static dof count, a term below a row's envelope start selected away, so every shared-memory
    address is a constant offset from base."""
    EPS = rigid_info.EPS[None]
    M = qd.static((N + K - 1) // K)
    own = qd.Vector.zero(gs.qd_int, M)
    dg = qd.Vector.zero(gs.qd_float, N)
    if active:
        for m in qd.static(range(M)):
            for s in qd.static(range(K)):
                if qd.static(K * m + s < N):
                    if sub == s:
                        own[m] = starts[K * m + s]
    for i_d in qd.static(range(N)):
        row_i = qd.static((i_d * (i_d - 1)) // 2)
        if active:
            i_start = starts[i_d]
            hess_diag = hds[i_d]
            tmp = hess_diag
            for j_d in qd.static(range(i_d)):
                tmp = qd.select(j_d >= i_start, tmp - sh[base + row_i + j_d] ** 2, tmp)
            diag = qd.sqrt(qd.max(tmp, EPS * qd.max(hess_diag, EPS)))
            dg[i_d] = diag
            inv = 1.0 / diag
            for m in qd.static(range(M)):
                if qd.static(K * m + K - 1 > i_d):
                    j_d = K * m + sub
                    if j_d > i_d and j_d < N:
                        row_j = (j_d * (j_d - 1)) // 2
                        j_start = own[m]
                        if j_start <= i_d:
                            k_lo = qd.max(i_start, j_start)
                            dot = gs.qd_float(0.0)
                            for k_d in qd.static(range(i_d)):
                                dot = qd.select(
                                    k_d >= k_lo, dot + (sh[base + row_j + k_d] * sh[base + row_i + k_d]), dot
                                )
                            sh[base + row_j + i_d] = (sh[base + row_j + i_d] - dot) * inv
        qd.simt.block.sync()
    # The factored diagonal goes back to nt_H once, after the steps.
    if active and sub == 0:
        for i_d in qd.static(range(N)):
            constraint_state.nt_H[i_b, i_d, i_d] = dg[i_d]
    return dg


@qd.func
def func_is_dense_single_block(i_b: int, constraint_state: ConstraintState):
    """Whether the env's gradient solve is the dense block at offset 0 of its one island (func_update_gradient_batch
    then reduces to func_gradient_solve_lds)."""
    return constraint_state.island.n_islands[i_b] == 1 and constraint_state.island.dof_range_start[0, i_b] == 0


@qd.func
def func_gradient_solve_lds(
    i_b: int,
    sh,
    base: int,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    N: qd.template(),
    dg,
):
    """func_update_gradient_batch of an env whose one island is the dense block at offset 0 (single island, no Jacobi
    scaling), the factor read from its packed strictly lower triangle at sh[base + i (i - 1) / 2 + j] and its diagonal
    dg (the values written to nt_H): same operations, same order, the gradient and Mgrad kept in registers and written
    once. Also writes the seed's search direction -Mgrad. Returns whether the island was solved."""
    wrote_search = False
    ma = qd.Vector.zero(gs.qd_float, N)
    qs = qd.Vector.zero(gs.qd_float, N)
    qc = qd.Vector.zero(gs.qd_float, N)
    for i_d in qd.static(range(N)):
        ma[i_d] = constraint_state.Ma[i_d, i_b]
        qs[i_d] = dyn_state.dofs.qf_smooth[i_d, i_b]
        qc[i_d] = constraint_state.qfrc_constraint[i_d, i_b]
    if constraint_state.island.improved[0, i_b]:
        grad = qd.Vector.zero(gs.qd_float, N)
        mgrad = qd.Vector.zero(gs.qd_float, N)
        for i_d in qd.static(range(N)):
            grad[i_d] = ma[i_d] - qs[i_d] - qc[i_d]
            constraint_state.grad[i_d, i_b] = grad[i_d]
        for i_d in qd.static(range(N)):
            row_i = qd.static((i_d * (i_d - 1)) // 2)
            curr_out = grad[i_d]
            for j_d in qd.static(range(i_d)):
                curr_out = curr_out - sh[base + row_i + j_d] * mgrad[j_d]
            mgrad[i_d] = curr_out / dg[i_d]
        for i_rev in qd.static(range(N)):
            i_d = qd.static(N - 1 - i_rev)
            curr_out = mgrad[i_d]
            for j_d in qd.static(range(i_d + 1, N)):
                curr_out = curr_out - sh[base + qd.static((j_d * (j_d - 1)) // 2) + i_d] * mgrad[j_d]
            mgrad[i_d] = curr_out / dg[i_d]
        for i_d in qd.static(range(N)):
            constraint_state.Mgrad[i_d, i_b] = mgrad[i_d]
        for i_d in qd.static(range(N)):
            constraint_state.search[i_d, i_b] = -mgrad[i_d]
        wrote_search = True
    return wrote_search
