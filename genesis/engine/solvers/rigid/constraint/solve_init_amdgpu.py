"""AMD-only fused, env-minor constraint passes of the solve init above the cooperative bound.

Without this, the final Jaref initialization and constraint update of func_solve_init run one thread per env
(_initialize_Jaref_per_env, then func_update_constraint_batch): at the PEAK env counts that is a single wave64 per SIMD
walking long dependent per-env loops, latency bound. Here one dispatch covers both, 64 consecutive envs per wave (the
batch-last constraint arrays keep a row's envs in consecutive words) and K lanes per env:
- each lane owns the rows i_c = sub, sub + K, ...: Jaref (same terms, same order) and then the active flag / force of
  the row (_func_update_efc_force_body, which only touches its own row on the pyramidal cone);
- after a block barrier, each lane owns the dofs i_d = sub, sub + K, ... of the dense J^T f gather (same row order per
  dof), or lane 0 runs the sparse scatter of a multi-island env;
- after a second barrier, lane 0 sums the cost in exactly the generic order.
The flag is decided at compile time from the static rigid config; every other configuration keeps the generic path.
"""

import genesis as gs
from genesis.utils.misc import get_gpu_core_count

# Lanes per env (the dispatch, func_init_jaref_update_constraint_amdgpu, runs 64 envs per wave64, one wave per lane
# share).
K_LANES = 8
# Lanes in flight the dispatch aims for: WAVES_PER_SIMD wave64 per SIMD of every compute unit. K_LANES halves while the
# envs alone (_B * K) would exceed it, down to MIN_K_LANES.
WAVES_PER_SIMD = 8
MIN_K_LANES = 2


def k_lanes(jac):
    """Lanes per env of the fused pass, from the static env count of the batch-last jac (n_rows, n_dofs, _B)."""
    k = K_LANES
    try:
        n_envs = tuple(jac.shape)[2]
    except Exception:
        return k
    if not isinstance(n_envs, int) or isinstance(n_envs, bool):
        return k
    # get_gpu_core_count is 64 lanes per CU on AMDGPU: WAVES_PER_SIMD wave64 on each of the 4 SIMDs of every CU
    target = get_gpu_core_count() * 4 * WAVES_PER_SIMD
    while k > MIN_K_LANES and n_envs * k > target:
        k //= 2
    return k


def jaref_update_fused(rigid_config):
    if rigid_config.backend != gs.amdgpu or rigid_config.requires_grad:
        return False
    if rigid_config.parallel_init or rigid_config.enable_cooperative_constraint_kernels:
        return False
    if rigid_config.constraint_layout_batch_first or rigid_config.sparse_solve:
        return False
    if rigid_config.para_level < gs.PARA_LEVEL.ALL:
        return False
    if rigid_config.enable_elliptic_friction or rigid_config.enable_signorini_contact:
        return False
    return True
