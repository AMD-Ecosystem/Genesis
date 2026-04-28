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

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
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
    """
    Sweep and Prune (SAP) for broad-phase collision detection.

    This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for
    potential collision pairs based on the AABB overlap.
    """
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    # Clear collider state
    func_collision_clear(links_state, links_info, collider_state, static_rigid_sim_config)

    MAX_GEOMS_NUM = qd.static(MAX_GEOMS_IN_LDS)
    MAX_SORT_ELEM_NUM = qd.static(MAX_GEOMS_NUM * 2)
    # Pad row stride by +1 so that 16 lanes (one per env in a workgroup) hit 16 distinct
    # LDS banks for column-aligned accesses instead of colliding 2-way (gcd(92, 32) = 4).
    LDS_ROW_STRIDE = qd.static(MAX_SORT_ELEM_NUM + 1)

    BLOCK_DIM = qd.static(64)
    ENVS_PER_BLOCK = qd.static(16)
    THREADS_PER_ENV = qd.static(BLOCK_DIM // ENVS_PER_BLOCK)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=BLOCK_DIM)
    for i_thread in range(_B * THREADS_PER_ENV):
        i_b = i_thread // THREADS_PER_ENV
        if i_thread - i_b * THREADS_PER_ENV != 0:
            continue

        # `lds_sort_i_g_packed` stores `(i_g << 1) | is_max` — packing the SAP endpoint flag
        # into the low bit of the geom index removes a separate LDS array (and its loads/stores)
        # from the inner sort/sweep loops. Geom ids fit easily in 31 bits.
        lds_sort_value = qd.simt.block.SharedArray((ENVS_PER_BLOCK, LDS_ROW_STRIDE), gs.qd_float)
        lds_sort_i_g_packed = qd.simt.block.SharedArray((ENVS_PER_BLOCK, LDS_ROW_STRIDE), gs.qd_int)
        lds_active = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_GEOMS_NUM), gs.qd_int)
        # Reverse map: for each geom id, its current index in `lds_active` (or stale if not active).
        # Lets us remove a geom in O(1) instead of O(n_active) linear-shift, in the hot
        # non-hibernation sweep path.
        lds_pos_in_active = qd.simt.block.SharedArray((ENVS_PER_BLOCK, MAX_GEOMS_NUM), gs.qd_int)

        i_b_lds = i_b % ENVS_PER_BLOCK

        axis = 0

        # Calculate the number of active geoms for this environment
        # (for heterogeneous entities, different envs may have different geoms)
        env_n_geoms = 0
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            env_n_geoms = env_n_geoms + links_info.geom_end[I_l] - links_info.geom_start[I_l]

        # copy updated geom aabbs to buffer for sorting.
        # Packed format: lds_sort_i_g_packed = (i_g << 1) | is_max_bit
        if collider_state.first_time[i_b]:
            i_buffer = 0
            for i_l in range(n_links):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                    lds_sort_value[i_b_lds, 2 * i_buffer] = geoms_state.aabb_min[i_g, i_b][axis]
                    lds_sort_i_g_packed[i_b_lds, 2 * i_buffer] = i_g << 1  # is_max=0

                    lds_sort_value[i_b_lds, 2 * i_buffer + 1] = geoms_state.aabb_max[i_g, i_b][axis]
                    lds_sort_i_g_packed[i_b_lds, 2 * i_buffer + 1] = (i_g << 1) | 1  # is_max=1

                    geoms_state.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                    geoms_state.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                    i_buffer = i_buffer + 1

            collider_state.first_time[i_b] = False

        else:
            # warm start. If `use_hibernation=True`, it's already updated in rigid_solver.
            if qd.static(not static_rigid_sim_config.use_hibernation):
                for i in range(env_n_geoms * 2):
                    is_max = collider_state.sort_buffer.is_max[i, i_b]
                    i_g = collider_state.sort_buffer.i_g[i, i_b]
                    if is_max:
                        lds_sort_value[i_b_lds, i] = geoms_state.aabb_max[i_g, i_b][axis]
                        lds_sort_i_g_packed[i_b_lds, i] = (i_g << 1) | 1
                    else:
                        lds_sort_value[i_b_lds, i] = geoms_state.aabb_min[i_g, i_b][axis]
                        lds_sort_i_g_packed[i_b_lds, i] = i_g << 1
            else:
                for i in range(env_n_geoms * 2):
                    is_max = collider_state.sort_buffer.is_max[i, i_b]
                    i_g = collider_state.sort_buffer.i_g[i, i_b]
                    lds_sort_value[i_b_lds, i] = collider_state.sort_buffer.value[i, i_b]
                    lds_sort_i_g_packed[i_b_lds, i] = (i_g << 1) | qd.cast(is_max, gs.qd_int)

        # Note: the previous implementation copied `collider_state.active_buffer` into `lds_active`
        # here, but it was dead — the sweep below starts with `n_active = 0` and rebuilds the set
        # from scratch. Removing the read saves env_n_geoms HBM loads per env per call.

        # insertion sort, which has complexity near O(n) for nearly sorted array.
        # Inner loop now does 2 LDS shifts/iter (value + packed i_g) instead of 3.
        for i in range(1, 2 * env_n_geoms):
            key_value = lds_sort_value[i_b_lds, i]
            key_packed = lds_sort_i_g_packed[i_b_lds, i]

            j = i - 1
            while j >= 0 and key_value < lds_sort_value[i_b_lds, j]:
                shifted_packed = lds_sort_i_g_packed[i_b_lds, j]
                lds_sort_value[i_b_lds, j + 1] = lds_sort_value[i_b_lds, j]
                lds_sort_i_g_packed[i_b_lds, j + 1] = shifted_packed

                if qd.static(static_rigid_sim_config.use_hibernation):
                    shifted_i_g = shifted_packed >> 1
                    if shifted_packed & 1:
                        geoms_state.max_buffer_idx[shifted_i_g, i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[shifted_i_g, i_b] = j + 1

                j -= 1
            lds_sort_value[i_b_lds, j + 1] = key_value
            lds_sort_i_g_packed[i_b_lds, j + 1] = key_packed

            if qd.static(static_rigid_sim_config.use_hibernation):
                key_i_g = key_packed >> 1
                if key_packed & 1:
                    geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        # sweep over the sorted AABBs to find potential collision pairs
        n_broad = 0
        if qd.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0
            for i in range(2 * env_n_geoms):
                packed_i = lds_sort_i_g_packed[i_b_lds, i]
                i_g_i = packed_i >> 1
                if (packed_i & 1) == 0:
                    # `min` event: pair this geom with every currently active one.
                    for j in range(n_active):
                        i_ga = lds_active[i_b_lds, j]
                        i_gb = i_g_i
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

                        if n_broad == collider_info.max_collision_pairs_broad[None]:
                            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                            break
                        collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                        collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                        n_broad = n_broad + 1

                    # Append new geom to active set; record its position for O(1) removal.
                    lds_active[i_b_lds, n_active] = i_g_i
                    lds_pos_in_active[i_b_lds, i_g_i] = n_active
                    n_active = n_active + 1
                else:
                    # `max` event: remove this geom from the active set in O(1) using
                    # swap-with-last (set membership is all the sweep cares about).
                    j = lds_pos_in_active[i_b_lds, i_g_i]
                    last_idx = n_active - 1
                    if j != last_idx:
                        swapped = lds_active[i_b_lds, last_idx]
                        lds_active[i_b_lds, j] = swapped
                        lds_pos_in_active[i_b_lds, swapped] = j
                    n_active = last_idx
        else:
            if rigid_global_info.n_awake_dofs[i_b] > 0:
                n_active_awake = 0
                n_active_hib = 0
                for i in range(2 * env_n_geoms):
                    packed_i = lds_sort_i_g_packed[i_b_lds, i]
                    i_g_i = packed_i >> 1
                    is_max_i = (packed_i & 1) != 0
                    is_incoming_geom_hibernated = geoms_state.hibernated[i_g_i, i_b]

                    if not is_max_i:
                        # both awake and hibernated geom check with active awake geoms
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = i_g_i
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
                                i_gb = i_g_i
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
                            collider_state.active_buffer_hib[n_active_hib, i_b] = i_g_i
                            n_active_hib = n_active_hib + 1
                        else:
                            collider_state.active_buffer_awake[n_active_awake, i_b] = i_g_i
                            n_active_awake = n_active_awake + 1
                    else:
                        i_g_to_remove = i_g_i
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

        # Decode packed (i_g, is_max) back to the persistent global sort_buffer fields.
        # The `lds_active` writeback was dead in the previous implementation (the sweep ends
        # with n_active = 0 and the next call rebuilds it from scratch), so it has been
        # removed entirely.
        for i in range(2 * env_n_geoms):
            packed = lds_sort_i_g_packed[i_b_lds, i]
            collider_state.sort_buffer.value[i, i_b] = lds_sort_value[i_b_lds, i]
            collider_state.sort_buffer.i_g[i, i_b] = packed >> 1
            collider_state.sort_buffer.is_max[i, i_b] = qd.cast(packed & 1, gs.qd_bool)

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
                    if not collider_state.sort_buffer.is_max[i, i_b]:
                        for j in range(n_active):
                            i_ga = collider_state.active_buffer[j, i_b]
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

                            if n_broad == collider_info.max_collision_pairs_broad[None]:
                                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                                break
                            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                            n_broad = n_broad + 1

                        collider_state.active_buffer[n_active, i_b] = collider_state.sort_buffer.i_g[i, i_b]
                        n_active = n_active + 1
                    else:
                        i_g_to_remove = collider_state.sort_buffer.i_g[i, i_b]
                        for j in range(n_active):
                            if collider_state.active_buffer[j, i_b] == i_g_to_remove:
                                if j < n_active - 1:
                                    for k in range(j, n_active - 1):
                                        collider_state.active_buffer[k, i_b] = collider_state.active_buffer[k + 1, i_b]
                                n_active = n_active - 1
                                break
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
