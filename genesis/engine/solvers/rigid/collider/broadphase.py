"""
Broad-phase collision detection functions.

This module contains AABB operations, sweep-and-prune algorithms,
and collision pair validation for the rigid body collider.
"""

import os

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .utils import (
    func_is_geom_aabbs_overlap,
)


# Wave-cooperative LDS broad-phase variant for AMDGPU. The original
# `func_broad_phase_lds` puts 16 envs in a 64-thread workgroup but only
# the first lane in each 4-lane group does work, leaving 75 % of VALU
# lanes idle on wave64. The `_wc` variant keeps the same launch shape
# but uses the 4 lanes per env to process AABB-sweep candidates in
# stride-4. Set `GS_BROAD_PHASE_DISABLE_WC=1` to fall back to the
# original LDS path (e.g. for A/B comparison or if a regression is
# observed). This is read at module import time (i.e. once per process
# launch); the kernel uses `qd.static(...)` to specialise on it at JIT
# compile time.
_BROAD_PHASE_USE_WC = os.environ.get("GS_BROAD_PHASE_DISABLE_WC", "0") != "1"


@qd.func
def func_find_intersect_midpoint(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    # return the center of the intersecting AABB of AABBs of two geoms
    intersect_lower = qd.max(geoms_state.aabb_min[i_ga, i_b], geoms_state.aabb_min[i_gb, i_b])
    intersect_upper = qd.min(geoms_state.aabb_max[i_ga, i_b], geoms_state.aabb_max[i_gb, i_b])
    return 0.5 * (intersect_lower + intersect_upper)


@qd.func
def func_check_collision_valid(
    i_ga,
    i_gb,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
    collider_info: array_class.ColliderInfo,
):
    is_valid = collider_info.collision_pair_idx[i_ga, i_gb] != -1

    if is_valid:
        i_la = geoms_info.link_idx[i_ga]
        i_lb = geoms_info.link_idx[i_gb]

        # Filter out collision pairs that are involved in dynamically registered weld equality constraints
        for i_eq in range(rigid_global_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            if equalities_info.eq_type[i_eq, i_b] == gs.EQUALITY_TYPE.WELD:
                i_leqa = equalities_info.eq_obj1id[i_eq, i_b]
                i_leqb = equalities_info.eq_obj2id[i_eq, i_b]
                if (i_leqa == i_la and i_leqb == i_lb) or (i_leqa == i_lb and i_leqb == i_la):
                    is_valid = False

        # hibernated <-> fixed links
        if qd.static(static_rigid_sim_config.use_hibernation):
            I_la = [i_la, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_la
            I_lb = [i_lb, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_lb

            if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
                links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
            ):
                is_valid = False

    return is_valid


@qd.func
def func_collision_clear(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    collider_state: array_class.ColliderState,
    static_rigid_sim_config: qd.template(),
):
    _B = collider_state.n_contacts.shape[0]

    qd.loop_config(name="collision_clear", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        if qd.static(static_rigid_sim_config.use_hibernation):
            collider_state.n_contacts_hibernated[i_b] = 0

            # Advect hibernated contacts
            for i_c in range(collider_state.n_contacts[i_b]):
                i_la = collider_state.contact_data.link_a[i_c, i_b]
                i_lb = collider_state.contact_data.link_b[i_c, i_b]
                I_la = [i_la, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_la
                I_lb = [i_lb, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_lb

                # Pair of hibernated-fixed links -> hibernated contact
                # TODO: we should also include hibernated-hibernated links and wake up the whole contact island
                # once a new collision is detected
                if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
                    links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
                ):
                    i_c_hibernated = collider_state.n_contacts_hibernated[i_b]
                    if i_c != i_c_hibernated:
                        # Copying all fields of class ContactData individually
                        # (fields mode doesn't support struct-level copy operations):
                        # fmt: off
                        collider_state.contact_data.geom_a[i_c_hibernated, i_b] = collider_state.contact_data.geom_a[i_c, i_b]
                        collider_state.contact_data.geom_b[i_c_hibernated, i_b] = collider_state.contact_data.geom_b[i_c, i_b]
                        collider_state.contact_data.penetration[i_c_hibernated, i_b] = collider_state.contact_data.penetration[i_c, i_b]
                        collider_state.contact_data.normal[i_c_hibernated, i_b] = collider_state.contact_data.normal[i_c, i_b]
                        collider_state.contact_data.pos[i_c_hibernated, i_b] = collider_state.contact_data.pos[i_c, i_b]
                        collider_state.contact_data.friction[i_c_hibernated, i_b] = collider_state.contact_data.friction[i_c, i_b]
                        collider_state.contact_data.sol_params[i_c_hibernated, i_b] = collider_state.contact_data.sol_params[i_c, i_b]
                        collider_state.contact_data.force[i_c_hibernated, i_b] = collider_state.contact_data.force[i_c, i_b]
                        collider_state.contact_data.link_a[i_c_hibernated, i_b] = collider_state.contact_data.link_a[i_c, i_b]
                        collider_state.contact_data.link_b[i_c_hibernated, i_b] = collider_state.contact_data.link_b[i_c, i_b]
                        # fmt: on
                    collider_state.n_contacts_hibernated[i_b] = i_c_hibernated + 1

        # Clear contacts: when hibernation is enabled, only clear non-hibernated contacts.
        # The hibernated contacts (positions 0 to n_contacts_hibernated-1) were just advected and should be preserved.
        for i_c in range(collider_state.n_contacts[i_b]):
            should_clear = True
            if qd.static(static_rigid_sim_config.use_hibernation):
                # Only clear if this is not a hibernated contact
                should_clear = i_c >= collider_state.n_contacts_hibernated[i_b]
            if should_clear:
                collider_state.contact_data.link_a[i_c, i_b] = -1
                collider_state.contact_data.link_b[i_c, i_b] = -1
                collider_state.contact_data.geom_a[i_c, i_b] = -1
                collider_state.contact_data.geom_b[i_c, i_b] = -1
                collider_state.contact_data.penetration[i_c, i_b] = 0.0
                collider_state.contact_data.pos[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
                collider_state.contact_data.normal[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
                collider_state.contact_data.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)

        if qd.static(static_rigid_sim_config.use_hibernation):
            collider_state.n_contacts[i_b] = collider_state.n_contacts_hibernated[i_b]
        else:
            collider_state.n_contacts[i_b] = 0

MAX_GEOMS_IN_LDS = 60

@qd.kernel(fastcache=gs.use_fastcache)
def func_broad_phase_lds(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    equalities_info: array_class.EqualitiesInfo,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    """
    Sweep and Prune (SAP) for broad-phase collision detection.

    This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for
    potential collision pairs based on the AABB overlap.

    The optimized LDS path primarily targets use_hibernation=False.
    The hibernation path keeps the original active_buffer_awake/hib logic.
    """
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    # Clear collider state
    func_collision_clear(links_state, links_info, collider_state, static_rigid_sim_config)

    MAX_GEOMS_NUM = qd.static(MAX_GEOMS_IN_LDS)
    MAX_SORT_ELEM_NUM = qd.static(MAX_GEOMS_NUM * 2)

    BLOCK_DIM = qd.static(64)
    ENVS_PER_BLOCK = qd.static(16)

    # Only one lane out of THREADS_PER_ENV currently processes one env.
    # THREADS_PER_ENV is used to map 16 envs to one 64-thread workgroup and
    # reserve one LDS slot per env.
    THREADS_PER_ENV = qd.static(BLOCK_DIM // ENVS_PER_BLOCK)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=BLOCK_DIM)
    for i_thread in range(_B * THREADS_PER_ENV):
        i_b = i_thread // THREADS_PER_ENV
        if i_thread - i_b * THREADS_PER_ENV != 0:
            continue

        lds_sort_value = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_SORT_ELEM_NUM), gs.qd_float)

        # Packed format: lds_sort_i_g_packed = (i_g << 1) | is_max_bit
        lds_sort_packed = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_SORT_ELEM_NUM), gs.qd_int)

        # Don't need to copy `collider_state.active_buffer` into `lds_active` before using it.
        # Because the sweep below starts with `n_active = 0` and rebuilds the set from scratch.
        lds_active = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_GEOMS_NUM), gs.qd_int)
        
        i_b_lds = i_b % ENVS_PER_BLOCK

        axis = 0

        # Calculate the number of active geoms for this environment
        # (for heterogeneous entities, different envs may have different geoms)
        env_n_geoms = 0
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            env_n_geoms = env_n_geoms + links_info.geom_end[I_l] - links_info.geom_start[I_l]

        # copy updated geom aabbs to buffer for sorting
        if collider_state.first_time[i_b]:
            i_buffer = 0
            for i_l in range(n_links):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                    lds_sort_value[i_b_lds, 2 * i_buffer] = geoms_state.aabb_min[i_g, i_b][axis]
                    lds_sort_packed[i_b_lds, 2 * i_buffer] = i_g << 1 # is_max = 0

                    lds_sort_value[i_b_lds, 2 * i_buffer + 1] = geoms_state.aabb_max[i_g, i_b][axis]
                    lds_sort_packed[i_b_lds, 2 * i_buffer + 1] = (i_g << 1) | 1 # is_max = 1

                    geoms_state.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                    geoms_state.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                    i_buffer = i_buffer + 1

            collider_state.first_time[i_b] = False

        else:
            if qd.static(not static_rigid_sim_config.use_hibernation):
                for i in range(env_n_geoms * 2):
                    is_max = collider_state.sort_buffer.is_max[i, i_b]
                    i_g = collider_state.sort_buffer.i_g[i, i_b]
                    if is_max:
                        lds_sort_value[i_b_lds, i] = geoms_state.aabb_max[i_g, i_b][axis]
                    else:
                        lds_sort_value[i_b_lds, i] = geoms_state.aabb_min[i_g, i_b][axis]

                    lds_sort_packed[i_b_lds, i] = (i_g << 1) | qd.cast(is_max, gs.qd_int)
            else:
                for i in range(env_n_geoms * 2):
                    is_max = collider_state.sort_buffer.is_max[i, i_b]
                    i_g = collider_state.sort_buffer.i_g[i, i_b]
                    value = collider_state.sort_buffer.value[i, i_b]
                    lds_sort_packed[i_b_lds, i] = (i_g << 1) | qd.cast(is_max, gs.qd_int)
                    lds_sort_value[i_b_lds, i] = value


        # insertion sort, which has complexity near O(n) for nearly sorted array
        for i in range(1, 2 * env_n_geoms):
            key_value = lds_sort_value[i_b_lds, i]
            key_packed_ig_ismax = lds_sort_packed[i_b_lds, i]

            j = i - 1
            while j >= 0 and key_value < lds_sort_value[i_b_lds, j]:
                packed_ig_ismax = lds_sort_packed[i_b_lds, j]
                lds_sort_value[i_b_lds, j + 1] = lds_sort_value[i_b_lds, j]
                lds_sort_packed[i_b_lds, j + 1] = packed_ig_ismax

                if qd.static(static_rigid_sim_config.use_hibernation):
                    shifted_i_g = packed_ig_ismax >> 1
                    if packed_ig_ismax & 1:
                        geoms_state.max_buffer_idx[shifted_i_g, i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[shifted_i_g, i_b] = j + 1

                j -= 1
            lds_sort_value[i_b_lds, j + 1] = key_value
            lds_sort_packed[i_b_lds, j + 1] = key_packed_ig_ismax

            if qd.static(static_rigid_sim_config.use_hibernation):
                key_i_g = key_packed_ig_ismax >> 1
                if key_packed_ig_ismax & 1:
                    geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        
        n_broad = 0
        if qd.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0

            for i in range(2 * env_n_geoms):
                packed_ig_ismax = lds_sort_packed[i_b_lds, i]
                is_max = packed_ig_ismax & 1
                i_g = packed_ig_ismax >> 1
                

                if not is_max:
                    min_b0, min_b1, min_b2 = geoms_state.aabb_min[i_g, i_b]
                    max_b0, max_b1, max_b2 = geoms_state.aabb_max[i_g, i_b]

                    for j in range(n_active):
                        i_ga = lds_active[i_b_lds, j]

                        i_ga_c = i_ga
                        i_gb_c = i_g
                        if i_ga > i_g:
                            i_ga_c = i_g
                            i_gb_c = i_ga

                        if collider_info.collision_pair_idx[i_ga_c, i_gb_c] == -1:
                            continue

                        min_a0, min_a1, min_a2 = geoms_state.aabb_min[i_ga, i_b]
                        max_a0, max_a1, max_a2 = geoms_state.aabb_max[i_ga, i_b]
                        

                        if (min_a0 > max_b0 or min_a1 > max_b1 or min_a2 > max_b2 or
                            max_a0 < min_b0 or max_a1 < min_b1 or max_a2 < min_b2):
                            continue

                        if not func_check_collision_valid(
                            i_ga_c,
                            i_gb_c,
                            i_b,
                            links_state,
                            links_info,
                            geoms_info,
                            rigid_global_info,
                            static_rigid_sim_config,
                            constraint_state,
                            equalities_info,
                            collider_info,
                        ):
                            continue

                        if n_broad < collider_info.max_collision_pairs_broad[None]:
                            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga_c
                            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb_c
                            n_broad += 1
                        else:
                            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS

                    lds_active[i_b_lds, n_active] = i_g
                    geoms_state.active_buffer_idx[i_g, i_b] = n_active
                    n_active += 1

                else:
                    j_remove = geoms_state.active_buffer_idx[i_g, i_b]
                    if j_remove < n_active - 1:
                        # Swap with last element
                        i_g_last = lds_active[i_b_lds, n_active - 1]
                        lds_active[i_b_lds, j_remove] = i_g_last
                        geoms_state.active_buffer_idx[i_g_last, i_b] = j_remove
                    n_active -= 1

            collider_state.n_broad_pairs[i_b] = n_broad
        else:
            if rigid_global_info.n_awake_dofs[i_b] > 0:
                n_active_awake = 0
                n_active_hib = 0
                for i in range(2 * env_n_geoms):
                    packed_ig_ismax = lds_sort_packed[i_b_lds, i]
                    i_gb_origin = packed_ig_ismax >> 1
                    is_max = packed_ig_ismax & 1
                    is_incoming_geom_hibernated = geoms_state.hibernated[i_gb_origin, i_b]

                    if not is_max:
                        # both awake and hibernated geom check with active awake geoms
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = i_gb_origin
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not func_check_collision_valid(
                                i_ga,
                                i_gb,
                                i_b,
                                links_state,
                                links_info,
                                geoms_info,
                                rigid_global_info,
                                static_rigid_sim_config,
                                constraint_state,
                                equalities_info,
                                collider_info,
                            ):
                                continue

                            if not func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                # Clear collision normal cache if not in contact
                                if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                continue

                            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                            n_broad = n_broad + 1

                        # if incoming geom is awake, also need to check with hibernated geoms
                        if not is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                i_ga = collider_state.active_buffer_hib[j, i_b]
                                i_gb = i_gb_origin
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not func_check_collision_valid(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    links_state,
                                    links_info,
                                    geoms_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    constraint_state,
                                    equalities_info,
                                    collider_info,
                                ):
                                    continue

                                if not func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                    # Clear collision normal cache if not in contact
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                    continue

                                collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                                collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                                n_broad = n_broad + 1

                        if is_incoming_geom_hibernated:
                            collider_state.active_buffer_hib[n_active_hib, i_b] = i_gb_origin
                            n_active_hib = n_active_hib + 1
                        else:
                            collider_state.active_buffer_awake[n_active_awake, i_b] = i_gb_origin
                            n_active_awake = n_active_awake + 1
                    else:
                        i_g_to_remove = i_gb_origin
                        if is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                if collider_state.active_buffer_hib[j, i_b] == i_g_to_remove:
                                    if j < n_active_hib - 1:
                                        for k in range(j, n_active_hib - 1):
                                            collider_state.active_buffer_hib[k, i_b] = collider_state.active_buffer_hib[
                                                k + 1, i_b
                                            ]
                                    n_active_hib = n_active_hib - 1
                                    break
                        else:
                            for j in range(n_active_awake):
                                if collider_state.active_buffer_awake[j, i_b] == i_g_to_remove:
                                    if j < n_active_awake - 1:
                                        for k in range(j, n_active_awake - 1):
                                            collider_state.active_buffer_awake[k, i_b] = (
                                                collider_state.active_buffer_awake[k + 1, i_b]
                                            )
                                    n_active_awake = n_active_awake - 1
                                    break

        for i in range(env_n_geoms):

            if qd.static(static_rigid_sim_config.use_hibernation):
                collider_state.sort_buffer.value[2 * i, i_b] = lds_sort_value[i_b_lds, 2 * i]
                collider_state.sort_buffer.value[2 * i + 1, i_b] = lds_sort_value[i_b_lds, 2 * i + 1]

            packed_ig_ismax = lds_sort_packed[i_b_lds, 2 * i]
            collider_state.sort_buffer.i_g[2 * i, i_b] = packed_ig_ismax >> 1
            collider_state.sort_buffer.is_max[2 * i, i_b] = qd.cast(packed_ig_ismax & 1, gs.qd_bool)

            packed_ig_ismax = lds_sort_packed[i_b_lds, 2 * i + 1]
            collider_state.sort_buffer.i_g[2 * i + 1, i_b] = packed_ig_ismax >> 1
            collider_state.sort_buffer.is_max[2 * i + 1, i_b] = qd.cast(packed_ig_ismax & 1, gs.qd_bool)

            if qd.static(not static_rigid_sim_config.use_hibernation):
                collider_state.active_buffer[i, i_b] = lds_active[i_b_lds, i]

        collider_state.n_broad_pairs[i_b] = n_broad


# ---------------------------------------------------------------------------
# Wave-cooperative broad-phase variant for AMDGPU wave64
# ---------------------------------------------------------------------------
#
# `func_broad_phase_lds` (above) maps 16 envs to a 64-thread workgroup with
# `THREADS_PER_ENV = 4`, but inside the kernel only `lane_in_env == 0`
# actually does work (lines 185-186 of the LDS variant). On AMDGPU wave64
# this leaves 75 % of VALU lanes idle for the entire kernel runtime.
#
# This variant keeps the same launch shape (16 envs/workgroup,
# `THREADS_PER_ENV = 4`, BLOCK_DIM = 64) so the per-CU LDS budget and the
# total wavefront count are unchanged, but uses the 4 lanes per env to
# parallelise the hot inner work:
#
#   * Sort + initial buffer fill: leader-only (`lane_in_env == 0`), with
#     a workgroup sync afterwards so the other 3 lanes see the sorted
#     LDS state. Insertion sort is sequential by construction (each step
#     reads from the slot one back to decide whether to shift); making
#     it cooperative would require a different sort entirely.
#
#   * Sweep candidate-test (the `for j in range(n_active)` loop on lines
#     280-323 of `func_broad_phase_lds`): the 4 lanes process candidates
#     in stride-4 (lane k checks j = k, k+4, k+8, ...). This is where the
#     work is concentrated -- each candidate triggers up to 6 HBM AABB
#     loads, a `collision_pair_idx` lookup, and (on hit) a
#     `func_check_collision_valid` call that walks
#     `qd_n_equalities[i_b]`. With G1's ~46 geoms and ~30 simultaneously
#     active ones per env, this is ~30 candidates per event * 2 *
#     env_n_geoms events per env, so the divergence cost dwarfs the
#     other phases.
#
#   * Per-lane hit results land in a small LDS staging buffer
#     (`lds_lane_hits` / `lds_lane_hit_count`). The leader lane then
#     drains the 4 buckets in interleaved order (lane 0's k-th hit,
#     then lane 1's k-th hit, ..., then lane 0's (k+1)-th, ...). When
#     hit/miss patterns differ across lanes the relative order of
#     broad pairs *within a single event* may diverge from the
#     sequential LDS variant. That's safe: the narrowphase processes
#     each broad pair independently (each thread handles a chunk of
#     the pair list and writes contacts via atomic increment), so
#     intra-event order does not affect downstream correctness, only
#     the cache-access pattern. If a test ever relies on bit-exact
#     `broad_collision_pairs` between the LDS and WC variants, set
#     `GS_BROAD_PHASE_DISABLE_WC=1` to fall back to the sequential
#     LDS path.
#
#   * `n_broad` stays a leader-only register (only the leader writes to
#     `broad_collision_pairs[n_broad, i_b]` and the final
#     `n_broad_pairs[i_b]`). `n_active` and `lds_active` are kept in
#     sync across all 4 lanes by having every lane execute the
#     control-flow updates with identical inputs (LDS broadcasts).
#
#   * Hibernation path is left scalar (gated on `lane_in_env == 0`) --
#     this benchmark uses `use_hibernation = False`, and the
#     hibernation branch has more complex active-set bookkeeping that
#     isn't on the perf-critical path. Treat as a follow-up.
#
# LDS budget:
#   * Existing: `lds_sort_value` + `lds_sort_packed` + `lds_active` =
#     ~14.4 KB at MAX_GEOMS_NUM=46, ENVS_PER_BLOCK=16.
#   * New: `lds_lane_hits` (16 envs * 4 lanes * 12 hits * 4 B i32 =
#     3.0 KB) + `lds_lane_hit_count` (16 envs * 4 lanes * 4 B = 0.25
#     KB) = ~3.25 KB.
#   * Total per workgroup: ~17.7 KB, well within MI300X's 64 KB / CU
#     LDS budget. Occupancy stays ~3 workgroups per CU.
@qd.func
def func_broad_phase_lds_wc(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    equalities_info: array_class.EqualitiesInfo,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    # Clear collider state
    func_collision_clear(links_state, links_info, collider_state, static_rigid_sim_config)

    MAX_GEOMS_NUM = qd.static(MAX_GEOMS_IN_LDS)
    MAX_SORT_ELEM_NUM = qd.static(MAX_GEOMS_NUM * 2)

    BLOCK_DIM = qd.static(64)
    ENVS_PER_BLOCK = qd.static(16)
    THREADS_PER_ENV = qd.static(BLOCK_DIM // ENVS_PER_BLOCK)
    # Per-lane hit-list capacity: with stride-THREADS_PER_ENV scan over at
    # most MAX_GEOMS_NUM candidates, each lane sees at most ceil(M/T) of
    # them, all of which could be hits in the worst case.
    MAX_HITS_PER_LANE = qd.static((MAX_GEOMS_NUM + THREADS_PER_ENV - 1) // THREADS_PER_ENV)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=BLOCK_DIM)
    for i_thread in range(_B * THREADS_PER_ENV):
        i_b = i_thread // THREADS_PER_ENV
        # Drop the legacy `if lane != 0: continue` -- all 4 lanes per env
        # execute the body. Per-env scratch is keyed by `i_b_lds`.
        lane_in_env = i_thread - i_b * THREADS_PER_ENV
        is_leader = lane_in_env == 0

        lds_sort_value = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_SORT_ELEM_NUM), gs.qd_float)
        # Packed format: lds_sort_i_g_packed = (i_g << 1) | is_max_bit
        lds_sort_packed = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_SORT_ELEM_NUM), gs.qd_int)
        lds_active = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_GEOMS_NUM), gs.qd_int)
        # Per-(env, lane) staging for the cooperative candidate scan.
        # Hits are packed as `(i_ga_c << 16) | i_gb_c` -- both fit in 16
        # bits because MAX_GEOMS_NUM = 46.
        lds_lane_hits = qd.simt.block.SharedArray(
            (ENVS_PER_BLOCK, THREADS_PER_ENV, MAX_HITS_PER_LANE), gs.qd_int
        )
        lds_lane_hit_count = qd.simt.block.SharedArray((ENVS_PER_BLOCK, THREADS_PER_ENV), gs.qd_int)

        i_b_lds = i_b % ENVS_PER_BLOCK
        axis = 0

        # Per-env active-geom count: identical formula on every lane, so
        # all 4 lanes converge to the same value as a register.
        env_n_geoms = 0
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            env_n_geoms = env_n_geoms + links_info.geom_end[I_l] - links_info.geom_start[I_l]

        # ---------------- Initial buffer fill (leader only) ---------------
        # Insertion sort is fundamentally sequential and the buffer fill
        # writes to `lds_sort_*` and `geoms_state.{min,max}_buffer_idx`.
        # Restricting these to lane 0 avoids 4-way races on the same LDS
        # slots. The trailing block sync makes the sorted state visible
        # to lanes 1..3 before the sweep loop starts.
        if is_leader:
            if collider_state.first_time[i_b]:
                i_buffer = 0
                for i_l in range(n_links):
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                        lds_sort_value[i_b_lds, 2 * i_buffer] = geoms_state.aabb_min[i_g, i_b][axis]
                        lds_sort_packed[i_b_lds, 2 * i_buffer] = i_g << 1  # is_max = 0

                        lds_sort_value[i_b_lds, 2 * i_buffer + 1] = geoms_state.aabb_max[i_g, i_b][axis]
                        lds_sort_packed[i_b_lds, 2 * i_buffer + 1] = (i_g << 1) | 1  # is_max = 1

                        geoms_state.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                        geoms_state.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                        i_buffer = i_buffer + 1

                collider_state.first_time[i_b] = False
            else:
                if qd.static(not static_rigid_sim_config.use_hibernation):
                    for i in range(env_n_geoms * 2):
                        is_max = collider_state.sort_buffer.is_max[i, i_b]
                        i_g = collider_state.sort_buffer.i_g[i, i_b]
                        if is_max:
                            lds_sort_value[i_b_lds, i] = geoms_state.aabb_max[i_g, i_b][axis]
                        else:
                            lds_sort_value[i_b_lds, i] = geoms_state.aabb_min[i_g, i_b][axis]
                        lds_sort_packed[i_b_lds, i] = (i_g << 1) | qd.cast(is_max, gs.qd_int)
                else:
                    for i in range(env_n_geoms * 2):
                        is_max = collider_state.sort_buffer.is_max[i, i_b]
                        i_g = collider_state.sort_buffer.i_g[i, i_b]
                        value = collider_state.sort_buffer.value[i, i_b]
                        lds_sort_packed[i_b_lds, i] = (i_g << 1) | qd.cast(is_max, gs.qd_int)
                        lds_sort_value[i_b_lds, i] = value

            # ---------------- Insertion sort (leader only) -------------
            # Near-O(n) for nearly-sorted data, which is the steady-state
            # case (small per-step AABB perturbations preserve order).
            for i in range(1, 2 * env_n_geoms):
                key_value = lds_sort_value[i_b_lds, i]
                key_packed_ig_ismax = lds_sort_packed[i_b_lds, i]

                j = i - 1
                while j >= 0 and key_value < lds_sort_value[i_b_lds, j]:
                    packed_ig_ismax = lds_sort_packed[i_b_lds, j]
                    lds_sort_value[i_b_lds, j + 1] = lds_sort_value[i_b_lds, j]
                    lds_sort_packed[i_b_lds, j + 1] = packed_ig_ismax

                    if qd.static(static_rigid_sim_config.use_hibernation):
                        shifted_i_g = packed_ig_ismax >> 1
                        if packed_ig_ismax & 1:
                            geoms_state.max_buffer_idx[shifted_i_g, i_b] = j + 1
                        else:
                            geoms_state.min_buffer_idx[shifted_i_g, i_b] = j + 1

                    j -= 1
                lds_sort_value[i_b_lds, j + 1] = key_value
                lds_sort_packed[i_b_lds, j + 1] = key_packed_ig_ismax

                if qd.static(static_rigid_sim_config.use_hibernation):
                    key_i_g = key_packed_ig_ismax >> 1
                    if key_packed_ig_ismax & 1:
                        geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        # All 4 lanes per env wait for the leader's sort+fill to finish
        # before reading lds_sort_*. The sync is workgroup-wide so it
        # also synchronises across the 16 envs sharing this workgroup.
        qd.simt.block.sync()

        # ---------------- Sweep loop -------------------------------------
        # `n_broad` is a leader-only register (only the leader writes the
        # output array; non-leader lanes' n_broad is unused).
        # `n_active` is a per-lane register that all 4 lanes update with
        # identical inputs, so it stays in sync.
        n_broad = 0
        if qd.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0

            for i in range(2 * env_n_geoms):
                # Broadcast LDS read (all 4 lanes hit the same bank).
                packed_ig_ismax = lds_sort_packed[i_b_lds, i]
                is_max = packed_ig_ismax & 1
                i_g = packed_ig_ismax >> 1

                # CRITICAL: the workgroup contains 16 envs running in
                # lockstep through this `for i in range(2 *
                # env_n_geoms):` loop. Different envs may have a `not
                # is_max` event at the same `i` while others have an
                # `is_max` event. Since `qd.simt.block.sync()` is a
                # workgroup-wide barrier, *every* thread in the
                # workgroup must reach the SAME sync at the SAME
                # iteration to avoid deadlock. We therefore place the
                # two per-event syncs OUTSIDE any `if not is_max:`
                # branch, so threads in both branches converge on them.
                # `is_max` threads experience them as no-op barriers.
                #
                # All envs sharing this workgroup also have the same
                # `env_n_geoms * 2` loop bound when `batch_links_info`
                # is False. With `batch_links_info=True` this can vary
                # per env, but the existing LDS variant already relies
                # on per-env loop counts being equal in this dispatch
                # path (lds_sort_* sized to MAX_SORT_ELEM_NUM); in
                # practice the simulator keeps geom counts uniform
                # across envs in a workgroup.

                # ----- Phase A: lane-stride candidate scan + leader
                # pre-write of lds_active.
                if not is_max:
                    # Pre-write the new active-set entry now so the
                    # Phase A -> Phase B sync below also flushes this
                    # leader write. The slot index `n_active` is one
                    # past the last current entry, so Phase A's
                    # `j < n_active` reads never see this write
                    # (removes the read/write hazard within the same
                    # event without needing an extra sync).
                    if is_leader:
                        lds_active[i_b_lds, n_active] = i_g
                        geoms_state.active_buffer_idx[i_g, i_b] = n_active

                    min_b0, min_b1, min_b2 = geoms_state.aabb_min[i_g, i_b]
                    max_b0, max_b1, max_b2 = geoms_state.aabb_max[i_g, i_b]

                    # Stride-THREADS_PER_ENV scan: lane k visits
                    # j = k, k+T, k+2T, ... while j < n_active.
                    # Quadrants' `range_for` AST transformer only
                    # accepts 1- or 2-arg `range(...)`, so the
                    # 3-arg-range / `continue`-on-miss pattern of the
                    # LDS variant is rewritten as an explicit `while`
                    # loop with a `keep` flag (the loop's manual
                    # `j += T` increment makes `continue` unsafe -- it
                    # would skip the increment and infinite-loop).
                    my_count = 0
                    j = lane_in_env
                    while j < n_active:
                        i_ga = lds_active[i_b_lds, j]

                        i_ga_c = i_ga
                        i_gb_c = i_g
                        if i_ga > i_g:
                            i_ga_c = i_g
                            i_gb_c = i_ga

                        keep = collider_info.collision_pair_idx[i_ga_c, i_gb_c] != -1

                        if keep:
                            min_a0, min_a1, min_a2 = geoms_state.aabb_min[i_ga, i_b]
                            max_a0, max_a1, max_a2 = geoms_state.aabb_max[i_ga, i_b]

                            if (min_a0 > max_b0 or min_a1 > max_b1 or min_a2 > max_b2 or
                                max_a0 < min_b0 or max_a1 < min_b1 or max_a2 < min_b2):
                                keep = False

                        if keep:
                            if not func_check_collision_valid(
                                i_ga_c,
                                i_gb_c,
                                i_b,
                                links_state,
                                links_info,
                                geoms_info,
                                rigid_global_info,
                                static_rigid_sim_config,
                                constraint_state,
                                equalities_info,
                                collider_info,
                            ):
                                keep = False

                        if keep:
                            # Pack into one i32 -- both indices fit in
                            # 16 bits since MAX_GEOMS_NUM = 46.
                            lds_lane_hits[i_b_lds, lane_in_env, my_count] = (i_ga_c << 16) | i_gb_c
                            my_count = my_count + 1

                        j = j + THREADS_PER_ENV

                    lds_lane_hit_count[i_b_lds, lane_in_env] = my_count

                # Workgroup-wide sync between Phase A and Phase B.
                # Reached by ALL 64 threads (both `is_max` and
                # `not is_max` branches converge here).
                qd.simt.block.sync()

                # ----- Phase B: drain hit list (not_is_max) OR scalar
                # active-set removal (is_max). Both leader-only.
                if not is_max:
                    # Drain the 4 lane buckets in interleaved order
                    # (lane0[0], lane1[0], lane2[0], lane3[0],
                    # lane0[1], lane1[1], ...). When all candidates hit
                    # this exactly reproduces the sequential scan order
                    # j = 0, 1, 2, ..., n_active-1; when hit/miss
                    # patterns differ across lanes the relative order
                    # of broad pairs *within a single event* may
                    # diverge from the LDS variant. That's safe: the
                    # narrowphase processes each broad pair
                    # independently (each thread handles a chunk of
                    # the pair list and writes contacts via atomic
                    # increment of `n_contact`), so the intra-event
                    # order does not affect correctness, only the
                    # downstream cache-access pattern.
                    if is_leader:
                        cnt0 = lds_lane_hit_count[i_b_lds, 0]
                        cnt1 = lds_lane_hit_count[i_b_lds, 1]
                        cnt2 = lds_lane_hit_count[i_b_lds, 2]
                        cnt3 = lds_lane_hit_count[i_b_lds, 3]
                        max_cnt = qd.max(qd.max(cnt0, cnt1), qd.max(cnt2, cnt3))
                        max_pairs = collider_info.max_collision_pairs_broad[None]
                        for k in range(max_cnt):
                            if k < cnt0:
                                pair = lds_lane_hits[i_b_lds, 0, k]
                                if n_broad < max_pairs:
                                    collider_state.broad_collision_pairs[n_broad, i_b][0] = pair >> 16
                                    collider_state.broad_collision_pairs[n_broad, i_b][1] = pair & 0xFFFF
                                    n_broad = n_broad + 1
                                else:
                                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                            if k < cnt1:
                                pair = lds_lane_hits[i_b_lds, 1, k]
                                if n_broad < max_pairs:
                                    collider_state.broad_collision_pairs[n_broad, i_b][0] = pair >> 16
                                    collider_state.broad_collision_pairs[n_broad, i_b][1] = pair & 0xFFFF
                                    n_broad = n_broad + 1
                                else:
                                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                            if k < cnt2:
                                pair = lds_lane_hits[i_b_lds, 2, k]
                                if n_broad < max_pairs:
                                    collider_state.broad_collision_pairs[n_broad, i_b][0] = pair >> 16
                                    collider_state.broad_collision_pairs[n_broad, i_b][1] = pair & 0xFFFF
                                    n_broad = n_broad + 1
                                else:
                                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                            if k < cnt3:
                                pair = lds_lane_hits[i_b_lds, 3, k]
                                if n_broad < max_pairs:
                                    collider_state.broad_collision_pairs[n_broad, i_b][0] = pair >> 16
                                    collider_state.broad_collision_pairs[n_broad, i_b][1] = pair & 0xFFFF
                                    n_broad = n_broad + 1
                                else:
                                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                    # All 4 lanes increment n_active in lockstep so
                    # the loop bound stays consistent across lanes.
                    # The leader's lds_active pre-write above is
                    # already visible to lanes 1..3 (flushed by the
                    # Phase A -> Phase B sync), so the next event's
                    # Phase A reads the up-to-date active set.
                    n_active = n_active + 1

                else:
                    # Removal of geom from active set. Scalar work
                    # plus a couple LDS writes -- leader-only.
                    if is_leader:
                        j_remove = geoms_state.active_buffer_idx[i_g, i_b]
                        if j_remove < n_active - 1:
                            # Swap with last element
                            i_g_last = lds_active[i_b_lds, n_active - 1]
                            lds_active[i_b_lds, j_remove] = i_g_last
                            geoms_state.active_buffer_idx[i_g_last, i_b] = j_remove
                    n_active = n_active - 1

                # Workgroup-wide sync at end of event so the leader's
                # lds_active swap (in `is_max` branch) is visible to
                # lanes 1..3 on the next event's Phase A. For
                # `not is_max` events the Phase A -> B sync already
                # flushed the leader's pre-write, so this end-of-event
                # sync is redundant for them, but the unified barrier
                # is required to keep the workgroup converged across
                # branch-divergent envs in the same workgroup.
                qd.simt.block.sync()

            if is_leader:
                collider_state.n_broad_pairs[i_b] = n_broad
        else:
            # Hibernation path -- intentionally NOT cooperative. Wave64
            # bookkeeping is more complex here (separate awake/hib active
            # lists, in-place removals, n_awake_dofs gate), and this
            # benchmark uses use_hibernation=False. Keep the original
            # leader-only structure to avoid touching a less-tested code
            # path. Ports to a cooperative design would mirror the
            # !use_hibernation branch above.
            if is_leader:
                if rigid_global_info.n_awake_dofs[i_b] > 0:
                    n_active_awake = 0
                    n_active_hib = 0
                    for i in range(2 * env_n_geoms):
                        packed_ig_ismax = lds_sort_packed[i_b_lds, i]
                        i_gb_origin = packed_ig_ismax >> 1
                        is_max = packed_ig_ismax & 1
                        is_incoming_geom_hibernated = geoms_state.hibernated[i_gb_origin, i_b]

                        if not is_max:
                            # both awake and hibernated geom check with active awake geoms
                            for j in range(n_active_awake):
                                i_ga = collider_state.active_buffer_awake[j, i_b]
                                i_gb = i_gb_origin
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not func_check_collision_valid(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    links_state,
                                    links_info,
                                    geoms_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    constraint_state,
                                    equalities_info,
                                    collider_info,
                                ):
                                    continue

                                if not func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                    if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                        i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                        collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(
                                            gs.qd_float, 3
                                        )
                                    continue

                                collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                                collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                                n_broad = n_broad + 1

                            if not is_incoming_geom_hibernated:
                                for j in range(n_active_hib):
                                    i_ga = collider_state.active_buffer_hib[j, i_b]
                                    i_gb = i_gb_origin
                                    if i_ga > i_gb:
                                        i_ga, i_gb = i_gb, i_ga

                                    if not func_check_collision_valid(
                                        i_ga,
                                        i_gb,
                                        i_b,
                                        links_state,
                                        links_info,
                                        geoms_info,
                                        rigid_global_info,
                                        static_rigid_sim_config,
                                        constraint_state,
                                        equalities_info,
                                        collider_info,
                                    ):
                                        continue

                                    if not func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                        i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                        collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(
                                            gs.qd_float, 3
                                        )
                                        continue

                                    collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                                    collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                                    n_broad = n_broad + 1

                            if is_incoming_geom_hibernated:
                                collider_state.active_buffer_hib[n_active_hib, i_b] = i_gb_origin
                                n_active_hib = n_active_hib + 1
                            else:
                                collider_state.active_buffer_awake[n_active_awake, i_b] = i_gb_origin
                                n_active_awake = n_active_awake + 1
                        else:
                            i_g_to_remove = i_gb_origin
                            if is_incoming_geom_hibernated:
                                for j in range(n_active_hib):
                                    if collider_state.active_buffer_hib[j, i_b] == i_g_to_remove:
                                        if j < n_active_hib - 1:
                                            for k in range(j, n_active_hib - 1):
                                                collider_state.active_buffer_hib[k, i_b] = (
                                                    collider_state.active_buffer_hib[k + 1, i_b]
                                                )
                                        n_active_hib = n_active_hib - 1
                                        break
                            else:
                                for j in range(n_active_awake):
                                    if collider_state.active_buffer_awake[j, i_b] == i_g_to_remove:
                                        if j < n_active_awake - 1:
                                            for k in range(j, n_active_awake - 1):
                                                collider_state.active_buffer_awake[k, i_b] = (
                                                    collider_state.active_buffer_awake[k + 1, i_b]
                                                )
                                        n_active_awake = n_active_awake - 1
                                        break

        # ---------------- Write-back -------------------------------------
        # Persist the sorted buffer (and active set in the
        # !use_hibernation case) back to global memory for the next step.
        # Stride-THREADS_PER_ENV gives a free 4x speedup on this phase.
        # The original `func_broad_phase_lds` writes
        # `collider_state.n_broad_pairs[i_b] = n_broad`
        # unconditionally at the very end of the function, AFTER the
        # hibernation/non-hibernation branches. We mirror that so that
        # the hibernation path also persists its own n_broad
        # (computed leader-only above). Only the leader writes; non-
        # leader lanes' n_broad is unused.
        if is_leader:
            collider_state.n_broad_pairs[i_b] = n_broad

        # All 4 lanes participate in the write-back: lane k handles
        # i % THREADS_PER_ENV == k. This requires the LDS state to be
        # stable for all lanes -- the trailing block.sync() in the
        # !is_max branch (after the leader's lds_active write) is what
        # synchronises Phase B's writes; for the sweep tail there are no
        # more LDS writes after the loop exits, so reads here are safe.
        i = lane_in_env
        while i < env_n_geoms:
            if qd.static(static_rigid_sim_config.use_hibernation):
                collider_state.sort_buffer.value[2 * i, i_b] = lds_sort_value[i_b_lds, 2 * i]
                collider_state.sort_buffer.value[2 * i + 1, i_b] = lds_sort_value[i_b_lds, 2 * i + 1]

            packed_ig_ismax = lds_sort_packed[i_b_lds, 2 * i]
            collider_state.sort_buffer.i_g[2 * i, i_b] = packed_ig_ismax >> 1
            collider_state.sort_buffer.is_max[2 * i, i_b] = qd.cast(packed_ig_ismax & 1, gs.qd_bool)

            packed_ig_ismax = lds_sort_packed[i_b_lds, 2 * i + 1]
            collider_state.sort_buffer.i_g[2 * i + 1, i_b] = packed_ig_ismax >> 1
            collider_state.sort_buffer.is_max[2 * i + 1, i_b] = qd.cast(packed_ig_ismax & 1, gs.qd_bool)

            if qd.static(not static_rigid_sim_config.use_hibernation):
                collider_state.active_buffer[i, i_b] = lds_active[i_b_lds, i]

            i = i + THREADS_PER_ENV


@qd.func
def func_broad_phase_global_mem(
@qd.kernel(fastcache=True)
def _func_broad_phase_sap(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    equalities_info: array_class.EqualitiesInfo,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    """
    Sweep and Prune (SAP) for broad-phase collision detection.

    This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for
    potential collision pairs based on the AABB overlap.
    """
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    # Clear collider state
    func_collision_clear(links_state, links_info, collider_state, static_rigid_sim_config)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        axis = 0

        # Calculate the number of active geoms for this environment
        # (for heterogeneous entities, different envs may have different geoms)
        env_n_geoms = 0
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            env_n_geoms = env_n_geoms + links_info.geom_end[I_l] - links_info.geom_start[I_l]

        # copy updated geom aabbs to buffer for sorting
        if collider_state.first_time[i_b]:
            i_buffer = 0
            for i_l in range(n_links):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                    collider_state.sort_buffer.value[2 * i_buffer, i_b] = geoms_state.aabb_min[i_g, i_b][axis]
                    collider_state.sort_buffer.i_g[2 * i_buffer, i_b] = i_g
                    collider_state.sort_buffer.is_max[2 * i_buffer, i_b] = False

                    collider_state.sort_buffer.value[2 * i_buffer + 1, i_b] = geoms_state.aabb_max[i_g, i_b][axis]
                    collider_state.sort_buffer.i_g[2 * i_buffer + 1, i_b] = i_g
                    collider_state.sort_buffer.is_max[2 * i_buffer + 1, i_b] = True

                    geoms_state.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                    geoms_state.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                    i_buffer = i_buffer + 1

            collider_state.first_time[i_b] = False

        else:
            # warm start. If `use_hibernation=True`, it's already updated in rigid_solver.
            if qd.static(not static_rigid_sim_config.use_hibernation):
                for i in range(env_n_geoms * 2):
                    if collider_state.sort_buffer.is_max[i, i_b]:
                        collider_state.sort_buffer.value[i, i_b] = geoms_state.aabb_max[
                            collider_state.sort_buffer.i_g[i, i_b], i_b
                        ][axis]
                    else:
                        collider_state.sort_buffer.value[i, i_b] = geoms_state.aabb_min[
                            collider_state.sort_buffer.i_g[i, i_b], i_b
                        ][axis]

        # insertion sort, which has complexity near O(n) for nearly sorted array
        for i in range(1, 2 * env_n_geoms):
            key_value = collider_state.sort_buffer.value[i, i_b]
            key_is_max = collider_state.sort_buffer.is_max[i, i_b]
            key_i_g = collider_state.sort_buffer.i_g[i, i_b]

            j = i - 1
            while j >= 0 and key_value < collider_state.sort_buffer.value[j, i_b]:
                collider_state.sort_buffer.value[j + 1, i_b] = collider_state.sort_buffer.value[j, i_b]
                collider_state.sort_buffer.is_max[j + 1, i_b] = collider_state.sort_buffer.is_max[j, i_b]
                collider_state.sort_buffer.i_g[j + 1, i_b] = collider_state.sort_buffer.i_g[j, i_b]

                if qd.static(static_rigid_sim_config.use_hibernation):
                    if collider_state.sort_buffer.is_max[j, i_b]:
                        geoms_state.max_buffer_idx[collider_state.sort_buffer.i_g[j, i_b], i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[collider_state.sort_buffer.i_g[j, i_b], i_b] = j + 1

                j -= 1
            collider_state.sort_buffer.value[j + 1, i_b] = key_value
            collider_state.sort_buffer.is_max[j + 1, i_b] = key_is_max
            collider_state.sort_buffer.i_g[j + 1, i_b] = key_i_g

            if qd.static(static_rigid_sim_config.use_hibernation):
                if key_is_max:
                    geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        # sweep over the sorted AABBs to find potential collision pairs
        n_broad = 0
        if qd.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0

            for i in range(2 * env_n_geoms):
                i_g = collider_state.sort_buffer.i_g[i, i_b]
                is_max = collider_state.sort_buffer.is_max[i, i_b]

                if not is_max:
                    min_b0 = geoms_state.aabb_min[i_g, i_b][0]
                    min_b1 = geoms_state.aabb_min[i_g, i_b][1]
                    min_b2 = geoms_state.aabb_min[i_g, i_b][2]
                    max_b0 = geoms_state.aabb_max[i_g, i_b][0]
                    max_b1 = geoms_state.aabb_max[i_g, i_b][1]
                    max_b2 = geoms_state.aabb_max[i_g, i_b][2]

                    for j in range(n_active):
                        i_ga = collider_state.active_buffer[j, i_b]

                        i_ga_c = i_ga
                        i_gb_c = i_g
                        if i_ga > i_g:
                            i_ga_c = i_g
                            i_gb_c = i_ga

                        if collider_info.collision_pair_idx[i_ga_c, i_gb_c] == -1:
                            continue

                        max_a_axis = geoms_state.aabb_max[i_ga, i_b][axis]
                        if max_a_axis < min_b0:  # axis=0, so min_b0
                            continue

                        min_a0 = geoms_state.aabb_min[i_ga, i_b][0]
                        max_a0 = geoms_state.aabb_max[i_ga, i_b][0]
                        min_a1 = geoms_state.aabb_min[i_ga, i_b][1]
                        max_a1 = geoms_state.aabb_max[i_ga, i_b][1]
                        min_a2 = geoms_state.aabb_min[i_ga, i_b][2]
                        max_a2 = geoms_state.aabb_max[i_ga, i_b][2]

                        if not (min_a0 <= max_b0 and max_a0 >= min_b0 and
                                min_a1 <= max_b1 and max_a1 >= min_b1 and
                                min_a2 <= max_b2 and max_a2 >= min_b2):
                            continue

                        if not func_check_collision_valid(
                            i_ga_c,
                            i_gb_c,
                            i_b,
                            links_state,
                            links_info,
                            geoms_info,
                            rigid_global_info,
                            static_rigid_sim_config,
                            constraint_state,
                            equalities_info,
                            collider_info,
                        ):
                            continue

                        if n_broad < collider_info.max_collision_pairs_broad[None]:
                            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga_c
                            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb_c
                            n_broad += 1
                        else:
                            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS

                    collider_state.active_buffer[n_active, i_b] = i_g
                    geoms_state.active_buffer_idx[i_g, i_b] = n_active
                    n_active += 1

                else:
                    j_remove = geoms_state.active_buffer_idx[i_g, i_b]
                    if j_remove < n_active - 1:
                        # Swap with last element
                        i_g_last = collider_state.active_buffer[n_active - 1, i_b]
                        collider_state.active_buffer[j_remove, i_b] = i_g_last
                        geoms_state.active_buffer_idx[i_g_last, i_b] = j_remove
                    n_active -= 1

            collider_state.n_broad_pairs[i_b] = n_broad
        else:
            if rigid_global_info.n_awake_dofs[i_b] > 0:
                n_active_awake = 0
                n_active_hib = 0
                for i in range(2 * env_n_geoms):
                    is_incoming_geom_hibernated = geoms_state.hibernated[collider_state.sort_buffer.i_g[i, i_b], i_b]

                    if not collider_state.sort_buffer.is_max[i, i_b]:
                        # both awake and hibernated geom check with active awake geoms
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = collider_state.sort_buffer.i_g[i, i_b]
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not func_check_collision_valid(
                                i_ga,
                                i_gb,
                                i_b,
                                links_state,
                                links_info,
                                geoms_info,
                                rigid_global_info,
                                static_rigid_sim_config,
                                constraint_state,
                                equalities_info,
                                collider_info,
                            ):
                                continue

                            if not func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                # Clear collision normal cache if not in contact
                                if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                continue

                            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                            n_broad = n_broad + 1

                        # if incoming geom is awake, also need to check with hibernated geoms
                        if not is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                i_ga = collider_state.active_buffer_hib[j, i_b]
                                i_gb = collider_state.sort_buffer.i_g[i, i_b]
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not func_check_collision_valid(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    links_state,
                                    links_info,
                                    geoms_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    constraint_state,
                                    equalities_info,
                                    collider_info,
                                ):
                                    continue

                                if not func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                    # Clear collision normal cache if not in contact
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                    continue

                                collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                                collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                                n_broad = n_broad + 1

                        if is_incoming_geom_hibernated:
                            collider_state.active_buffer_hib[n_active_hib, i_b] = collider_state.sort_buffer.i_g[i, i_b]
                            n_active_hib = n_active_hib + 1
                        else:
                            collider_state.active_buffer_awake[n_active_awake, i_b] = collider_state.sort_buffer.i_g[
                                i, i_b
                            ]
                            n_active_awake = n_active_awake + 1
                    else:
                        i_g_to_remove = collider_state.sort_buffer.i_g[i, i_b]
                        if is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                if collider_state.active_buffer_hib[j, i_b] == i_g_to_remove:
                                    if j < n_active_hib - 1:
                                        for k in range(j, n_active_hib - 1):
                                            collider_state.active_buffer_hib[k, i_b] = collider_state.active_buffer_hib[
                                                k + 1, i_b
                                            ]
                                    n_active_hib = n_active_hib - 1
                                    break
                        else:
                            for j in range(n_active_awake):
                                if collider_state.active_buffer_awake[j, i_b] == i_g_to_remove:
                                    if j < n_active_awake - 1:
                                        for k in range(j, n_active_awake - 1):
                                            collider_state.active_buffer_awake[k, i_b] = (
                                                collider_state.active_buffer_awake[k + 1, i_b]
                                            )
                                    n_active_awake = n_active_awake - 1
                                    break
        collider_state.n_broad_pairs[i_b] = n_broad


@qd.kernel(fastcache=True)
def _func_broad_phase_all_vs_all(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    equalities_info: array_class.EqualitiesInfo,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    """
    All-vs-all broad-phase collision detection.

    Iterates over pre-filtered valid geom pairs in parallel across pairs and batches, checking 3D AABB overlap.
    Passing pairs are appended to the output buffer via atomic add.
    """
    if qd.static(static_rigid_sim_config.n_geoms <= MAX_GEOMS_IN_LDS and static_rigid_sim_config.backend != gs.cpu):
        # On AMDGPU wave64, the original `func_broad_phase_lds` leaves
        # 75 % of lanes idle (only `lane_in_env == 0` of each 4-lane env
        # group does work). Use the wave-cooperative variant
        # `func_broad_phase_lds_wc` to put those lanes to work on
        # stride-4 candidate scanning. The variant is gated behind
        # `_BROAD_PHASE_USE_WC` so it can be disabled with
        # `GS_BROAD_PHASE_DISABLE_WC=1`.
        #
        # CUDA wave32 has only 2 lanes per env in the existing layout
        # (THREADS_PER_ENV * ENVS_PER_BLOCK = 64 = 2 waves of 32, so
        # each env's 4 lanes straddle a wave boundary on NVIDIA). The
        # win there is much smaller and the LDS-staging cost may
        # dominate -- keep CUDA on the original sequential LDS path
        # until benchmarked.
        #
        # The WC variant inserts workgroup-wide block.sync() barriers
        # inside the per-event sweep loop, so all 64 threads in the
        # workgroup must take the same trip count through that loop
        # AND must all be live (no "phantom" threads from an
        # underfilled tail workgroup). The variant therefore requires:
        #
        #   1. `not batch_links_info`: with `batch_links_info=False`
        #      (the default and the benchmark's setting) `env_n_geoms`
        #      is identical for every env in the workgroup so the
        #      sweep-loop barriers always converge. With
        #      `batch_links_info=True` env_n_geoms can vary per env
        #      and the barriers would risk deadlock.
        #   2. `n_envs % 16 == 0`: the workgroup packs 16 envs each.
        #      An underfilled tail workgroup would have some 4-lane
        #      env groups that exit the outer thread loop early, but
        #      the live env groups still hit the in-kernel barriers
        #      -- the barrier semantics on AMDGPU require ALL launched
        #      threads in the workgroup to participate, so an
        #      underfilled workgroup risks a hang.
        #
        # Either condition failing falls back to the original LDS
        # variant.
        if qd.static(
            static_rigid_sim_config.backend == gs.amdgpu
            and _BROAD_PHASE_USE_WC
            and not static_rigid_sim_config.batch_links_info
            and static_rigid_sim_config.n_envs % 16 == 0
        ):
            func_broad_phase_lds_wc(
                links_state,
                links_info,
                geoms_state,
                geoms_info,
                rigid_global_info,
                static_rigid_sim_config,
                constraint_state,
                collider_state,
                equalities_info,
                collider_info,
                errno,
            )
        else:
            func_broad_phase_lds(
                links_state,
                links_info,
                geoms_state,
                geoms_info,
                rigid_global_info,
                static_rigid_sim_config,
                constraint_state,
                collider_state,
                equalities_info,
                collider_info,
                errno,
            )
    else:
        _func_broad_phase_sap(
            links_state,
            links_info,
            geoms_state,
            geoms_info,
            rigid_global_info,
            static_rigid_sim_config,
            constraint_state,
            collider_state,
            equalities_info,
            collider_info,
            errno,
        )
