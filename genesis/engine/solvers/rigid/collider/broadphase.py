"""
Broad-phase collision detection functions.

This module contains AABB operations, sweep-and-prune algorithms,
and collision pair validation for the rigid body collider.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .utils import (
    func_is_geom_aabbs_overlap,
)


# === sort_buffer.i_g_packed encoding helpers ================================
# i_g_packed is a u32 (see StructSortBuffer in genesis/utils/array_class.py)
# holding two fields per SAP event in a single word:
#
#     bit  0     : is_max   (0 = min/start event, 1 = max/end event)
#     bits 1..31 : i_g      (geom index, supports up to 2**31 - 1)
#
# These three helpers are the single source of truth for the encoding. All
# reads/writes of sort_buffer.i_g_packed in this file MUST go through them
# rather than open-coding `<< 1` / `>> 1` / `& 1`. That way a future change
# to the bit layout (e.g., promoting to u64, swapping which field is in the
# low bits) is one edit, not thirty. Range overflow is guarded host-side at
# scene-build in get_sort_buffer (see SORT_BUFFER_MAX_GEOMS).


@qd.func
def func_pack_event(i_g, is_max) -> qd.u32:
    # Pack (i_g, is_max) into a single u32 word.
    # is_max is coerced to {0, 1} via u32(...) so callers may pass a Python
    # bool literal (True/False) or a runtime bool/int interchangeably.
    return (qd.u32(i_g) << 1) | qd.u32(is_max)


@qd.func
def func_unpack_i_g(packed) -> qd.i32:
    # Logical right shift on u32 (no sign extension) recovers i_g.
    return qd.i32(packed >> 1)


@qd.func
def func_unpack_is_max(packed) -> gs.qd_bool:
    # Bit 0 of the packed word holds is_max.
    return (packed & 1) != 0


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
    n_eq_static,
    n_eq_dyn,
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

        # Filter out collision pairs that are involved in dynamically registered weld equality constraints.
        # Uses hoisted n_eq_static / n_eq_dyn so the per-pair scan avoids the scattered
        # qd_n_equalities[i_b] load. When there are no dynamic equalities, the range is
        # empty and the loop is a no-op.
        for i_eq in range(n_eq_static, n_eq_dyn):
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
def func_collision_clear_per_env(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    collider_state: array_class.ColliderState,
    static_rigid_sim_config: qd.template(),
):
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
                    # Copying all fields of class StructContactData individually
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

MAX_GEOMS_IN_LDS = 46

@qd.func
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
    """Sweep and Prune (SAP) for broad-phase collision detection -- LDS path.

    Integrates broadphase-stack commits 1+2+3 on top of PR #56's LDS scaffolding:
    * Commit 1: replace func_collision_clear() kernel-call with the per-env
      helper func_collision_clear_per_env(i_b, ...) inlined here.
    * Commit 2: hoist n_eq_static / n_eq_dyn out of the per-pair check.
    * Commit 3: pack i_g + is_max into a u32 single-word; use func_pack_event /
      func_unpack_i_g / func_unpack_is_max helpers; collapse 2-store sort_buffer
      write-back to a single i_g_packed store.

    Cooperative parallel warm-start (the original commit 4) is intentionally
    omitted from this experiment because adding qd.simt.block.sync() to a loop
    that doesn't fill block_dim=64 (e.g. n_envs=1 -> 4 active lanes) triggers
    a workgroup-barrier UB / GPU memory access fault. Lane-0-only execution
    is preserved here.
    """
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    MAX_GEOMS_NUM = qd.static(MAX_GEOMS_IN_LDS)
    MAX_SORT_ELEM_NUM = qd.static(MAX_GEOMS_NUM * 2)
    BLOCK_DIM = qd.static(64)
    ENVS_PER_BLOCK = qd.static(16)
    THREADS_PER_ENV = qd.static(BLOCK_DIM // ENVS_PER_BLOCK)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=BLOCK_DIM)
    for i_thread in range(_B * THREADS_PER_ENV):
        i_b = i_thread // THREADS_PER_ENV
        if i_thread - i_b * THREADS_PER_ENV != 0:
            continue

        lds_sort_value = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_SORT_ELEM_NUM), gs.qd_float)
        # Packed format: bit 0 = is_max, bits 1..31 = i_g (matches StructSortBuffer.i_g_packed).
        lds_sort_packed = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_SORT_ELEM_NUM), qd.u32)
        # No need to copy collider_state.active_buffer; sweep starts with n_active=0.
        lds_active = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_GEOMS_NUM), gs.qd_int)

        i_b_lds = i_b % ENVS_PER_BLOCK
        axis = 0

        # Commit 1: per-env contact-clear helper, inlined into this kernel.
        func_collision_clear_per_env(i_b, links_state, links_info, collider_state, static_rigid_sim_config)

        # Commit 2: hoist equality bounds out of the per-pair check.
        n_eq_static = rigid_global_info.n_equalities[None]
        n_eq_dyn = constraint_state.qd_n_equalities[i_b]

        # Calculate the number of active geoms for this environment.
        env_n_geoms = 0
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            env_n_geoms = env_n_geoms + links_info.geom_end[I_l] - links_info.geom_start[I_l]

        # copy updated geom aabbs to LDS for sorting
        if collider_state.first_time[i_b]:
            i_buffer = 0
            for i_l in range(n_links):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                    lds_sort_value[i_b_lds, 2 * i_buffer] = geoms_state.aabb_min[i_g, i_b][axis]
                    lds_sort_packed[i_b_lds, 2 * i_buffer] = func_pack_event(i_g, False)
                    lds_sort_value[i_b_lds, 2 * i_buffer + 1] = geoms_state.aabb_max[i_g, i_b][axis]
                    lds_sort_packed[i_b_lds, 2 * i_buffer + 1] = func_pack_event(i_g, True)
                    geoms_state.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                    geoms_state.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                    i_buffer = i_buffer + 1
            collider_state.first_time[i_b] = False
        else:
            # Warm-start re-fill: read packed event from global sort_buffer, decode via
            # helpers, look up new aabb extent in LDS slot.
            for i in range(env_n_geoms * 2):
                packed = collider_state.sort_buffer.i_g_packed[i, i_b]
                is_max = func_unpack_is_max(packed)
                i_g = func_unpack_i_g(packed)
                lds_sort_packed[i_b_lds, i] = packed
                if qd.static(not static_rigid_sim_config.use_hibernation):
                    if is_max:
                        lds_sort_value[i_b_lds, i] = geoms_state.aabb_max[i_g, i_b][axis]
                    else:
                        lds_sort_value[i_b_lds, i] = geoms_state.aabb_min[i_g, i_b][axis]
                else:
                    lds_sort_value[i_b_lds, i] = collider_state.sort_buffer.value[i, i_b]

        # insertion sort, near O(n) for nearly sorted input
        for i in range(1, 2 * env_n_geoms):
            key_value = lds_sort_value[i_b_lds, i]
            key_packed = lds_sort_packed[i_b_lds, i]

            j = i - 1
            while j >= 0 and key_value < lds_sort_value[i_b_lds, j]:
                slid_packed = lds_sort_packed[i_b_lds, j]
                lds_sort_value[i_b_lds, j + 1] = lds_sort_value[i_b_lds, j]
                lds_sort_packed[i_b_lds, j + 1] = slid_packed

                if qd.static(static_rigid_sim_config.use_hibernation):
                    slid_i_g = func_unpack_i_g(slid_packed)
                    if func_unpack_is_max(slid_packed):
                        geoms_state.max_buffer_idx[slid_i_g, i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[slid_i_g, i_b] = j + 1

                j -= 1
            lds_sort_value[i_b_lds, j + 1] = key_value
            lds_sort_packed[i_b_lds, j + 1] = key_packed

            if qd.static(static_rigid_sim_config.use_hibernation):
                key_i_g = func_unpack_i_g(key_packed)
                if func_unpack_is_max(key_packed):
                    geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        n_broad = 0
        if qd.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0
            for i in range(2 * env_n_geoms):
                packed = lds_sort_packed[i_b_lds, i]
                is_max = func_unpack_is_max(packed)
                i_g = func_unpack_i_g(packed)

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
                            n_eq_static,
                            n_eq_dyn,
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
                        i_g_last = lds_active[i_b_lds, n_active - 1]
                        lds_active[i_b_lds, j_remove] = i_g_last
                        geoms_state.active_buffer_idx[i_g_last, i_b] = j_remove
                    n_active -= 1
        else:
            if rigid_global_info.n_awake_dofs[i_b] > 0:
                n_active_awake = 0
                n_active_hib = 0
                for i in range(2 * env_n_geoms):
                    packed = lds_sort_packed[i_b_lds, i]
                    i_gb_origin = func_unpack_i_g(packed)
                    is_max = func_unpack_is_max(packed)
                    is_incoming_geom_hibernated = geoms_state.hibernated[i_gb_origin, i_b]

                    if not is_max:
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = i_gb_origin
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not func_check_collision_valid(
                                i_ga,
                                i_gb,
                                i_b,
                                n_eq_static,
                                n_eq_dyn,
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
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
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
                                    n_eq_static,
                                    n_eq_dyn,
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
                                            collider_state.active_buffer_hib[k, i_b] = collider_state.active_buffer_hib[k + 1, i_b]
                                    n_active_hib = n_active_hib - 1
                                    break
                        else:
                            for j in range(n_active_awake):
                                if collider_state.active_buffer_awake[j, i_b] == i_g_to_remove:
                                    if j < n_active_awake - 1:
                                        for k in range(j, n_active_awake - 1):
                                            collider_state.active_buffer_awake[k, i_b] = collider_state.active_buffer_awake[k + 1, i_b]
                                    n_active_awake = n_active_awake - 1
                                    break

        # Write-back to global sort_buffer for next step's warm-start.
        # Single i_g_packed store per event (commit 3 dtype change collapsed
        # what used to be two stores into one).
        for i in range(env_n_geoms):
            if qd.static(static_rigid_sim_config.use_hibernation):
                collider_state.sort_buffer.value[2 * i, i_b] = lds_sort_value[i_b_lds, 2 * i]
                collider_state.sort_buffer.value[2 * i + 1, i_b] = lds_sort_value[i_b_lds, 2 * i + 1]
            collider_state.sort_buffer.i_g_packed[2 * i, i_b] = lds_sort_packed[i_b_lds, 2 * i]
            collider_state.sort_buffer.i_g_packed[2 * i + 1, i_b] = lds_sort_packed[i_b_lds, 2 * i + 1]
            if qd.static(not static_rigid_sim_config.use_hibernation):
                collider_state.active_buffer[i, i_b] = lds_active[i_b_lds, i]

        collider_state.n_broad_pairs[i_b] = n_broad



@qd.func
def func_broad_phase_global_mem(
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
    """
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_collision_clear_per_env(i_b, links_state, links_info, collider_state, static_rigid_sim_config)

        # Hoist equality bounds out of the per-pair func_check_collision_valid hot path.
        # When n_eq_dyn == n_eq_static (the common case: no dynamic weld equalities), the
        # per-pair check skips a scattered qd_n_equalities[i_b] load entirely.
        n_eq_static = rigid_global_info.n_equalities[None]
        n_eq_dyn = constraint_state.qd_n_equalities[i_b]

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
                    collider_state.sort_buffer.i_g_packed[2 * i_buffer, i_b] = func_pack_event(i_g, False)

                    collider_state.sort_buffer.value[2 * i_buffer + 1, i_b] = geoms_state.aabb_max[i_g, i_b][axis]
                    collider_state.sort_buffer.i_g_packed[2 * i_buffer + 1, i_b] = func_pack_event(i_g, True)

                    geoms_state.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                    geoms_state.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                    i_buffer = i_buffer + 1

            collider_state.first_time[i_b] = False

        else:
            # warm start. If `use_hibernation=True`, it's already updated in rigid_solver.
            if qd.static(not static_rigid_sim_config.use_hibernation):
                for i in range(env_n_geoms * 2):
                    packed = collider_state.sort_buffer.i_g_packed[i, i_b]
                    is_max = func_unpack_is_max(packed)
                    i_g = func_unpack_i_g(packed)
                    if is_max:
                        collider_state.sort_buffer.value[i, i_b] = geoms_state.aabb_max[i_g, i_b][axis]
                    else:
                        collider_state.sort_buffer.value[i, i_b] = geoms_state.aabb_min[i_g, i_b][axis]

        # insertion sort, which has complexity near O(n) for nearly sorted array.
        # T1.3: i_g and is_max share a single i32 (bit 0 = is_max, bits 1+ = i_g)
        # so the inner loop touches 2 SoA streams instead of 3.
        for i in range(1, 2 * env_n_geoms):
            key_value = collider_state.sort_buffer.value[i, i_b]
            key_packed = collider_state.sort_buffer.i_g_packed[i, i_b]

            j = i - 1
            while j >= 0 and key_value < collider_state.sort_buffer.value[j, i_b]:
                collider_state.sort_buffer.value[j + 1, i_b] = collider_state.sort_buffer.value[j, i_b]
                slid_packed = collider_state.sort_buffer.i_g_packed[j, i_b]
                collider_state.sort_buffer.i_g_packed[j + 1, i_b] = slid_packed

                if qd.static(static_rigid_sim_config.use_hibernation):
                    slid_i_g = func_unpack_i_g(slid_packed)
                    slid_is_max = func_unpack_is_max(slid_packed)
                    if slid_is_max:
                        geoms_state.max_buffer_idx[slid_i_g, i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[slid_i_g, i_b] = j + 1

                j -= 1
            collider_state.sort_buffer.value[j + 1, i_b] = key_value
            collider_state.sort_buffer.i_g_packed[j + 1, i_b] = key_packed

            if qd.static(static_rigid_sim_config.use_hibernation):
                key_i_g = func_unpack_i_g(key_packed)
                key_is_max = func_unpack_is_max(key_packed)
                if key_is_max:
                    geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        # sweep over the sorted AABBs to find potential collision pairs
        n_broad = 0
        if qd.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0

            for i in range(2 * env_n_geoms):
                packed = collider_state.sort_buffer.i_g_packed[i, i_b]
                i_g = func_unpack_i_g(packed)
                is_max = func_unpack_is_max(packed)

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
                            n_eq_static,
                            n_eq_dyn,
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
                    # Single load of the packed event, decoded once per iteration via helpers.
                    packed = collider_state.sort_buffer.i_g_packed[i, i_b]
                    incoming_i_g = func_unpack_i_g(packed)
                    incoming_is_max = func_unpack_is_max(packed)
                    is_incoming_geom_hibernated = geoms_state.hibernated[incoming_i_g, i_b]

                    if not incoming_is_max:
                        # both awake and hibernated geom check with active awake geoms
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = incoming_i_g
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not func_check_collision_valid(
                                i_ga,
                                i_gb,
                                i_b,
                                n_eq_static,
                                n_eq_dyn,
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
                                i_gb = incoming_i_g
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not func_check_collision_valid(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    n_eq_static,
                                    n_eq_dyn,
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
                            collider_state.active_buffer_hib[n_active_hib, i_b] = incoming_i_g
                            n_active_hib = n_active_hib + 1
                        else:
                            collider_state.active_buffer_awake[n_active_awake, i_b] = incoming_i_g
                            n_active_awake = n_active_awake + 1
                    else:
                        i_g_to_remove = incoming_i_g
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


@qd.kernel(fastcache=gs.use_fastcache)
def func_broad_phase(
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
    """
    if qd.static(static_rigid_sim_config.n_geoms <= MAX_GEOMS_IN_LDS and static_rigid_sim_config.backend != gs.cpu):
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
        func_broad_phase_global_mem(
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
