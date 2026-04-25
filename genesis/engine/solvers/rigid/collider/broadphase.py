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


# Wave-cooperative AVA constants. Each block (BLOCK_DIM=64 lanes on AMD wave64)
# handles ONE env. Phase 1 sorts the geoms by axis-0 min via wave-cooperative
# bitonic sort; Phase 2 iterates pair candidates as ``(sorted_j, sorted_i)`` in
# lex order (the same order SAP would emit), filters via SAP's axis-0 active-list
# semantics, and writes survivors via a wave-cooperative prefix scan. Result is
# deterministic and SAP-equivalent in BOTH the pair set and the emission order,
# so downstream narrowphase/solver see identical input.
_WAVE_COOP_BROADPHASE_BLOCK_DIM = 64


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

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, force_inline=(gs.backend == gs.amdgpu))
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


@qd.kernel(fastcache=gs.use_fastcache)
def func_broad_phase_wave_coop(
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
    Wave-cooperative deterministic all-vs-all broad-phase that emits pairs
    in **SAP-equivalent order**.

    One block (BLOCK_DIM=64 lanes) handles one env. The kernel runs in two
    phases:

      Phase 1 — geom sort. Each lane owns one geom (tid in [0, n_geoms)).
        Lanes cooperatively bitonic-sort geom indices by ``aabb_min[i_g, 0]``
        (the axis-0 sweep coordinate). After the sort, ``sh_sort_g[k]`` is
        the geom index in sorted position ``k`` — exactly the order SAP
        would walk events on axis 0.

      Phase 2 — SAP-order pair iteration. Iterate pair candidates as
        ``(sorted_j, sorted_i)`` with ``sorted_i < sorted_j`` in lex order
        (sorted_j outer, sorted_i inner), exactly matching the order in
        which SAP emits pairs (when ``sorted_j``'s min event fires, pair it
        against currently-active ``sorted_i`` geoms in insertion order).
        For each candidate, apply SAP's axis-0 active-list filter
        (``aabb_max[g_i, 0] >= aabb_min[g_j, 0]``) plus the standard
        ``collision_pair_idx`` / ``func_check_collision_valid`` /
        ``func_is_geom_aabbs_overlap`` filters. Survivors flow through a
        wave-cooperative exclusive prefix scan and are written into
        ``broad_collision_pairs`` in batch order — which, because we
        iterate in SAP order, IS SAP order.

    Because the emission order matches SAP exactly (the pair SET also
    matches by construction), downstream narrowphase + constraint solver
    see identical inputs and the float-summation-order divergence that
    sank the row-major variant disappears.

    Limit: assumes ``n_geoms <= BLOCK_DIM`` (=64). The dispatcher in
    ``collider.py`` checks this and falls back to SAP otherwise.
    """
    n_geoms, _B = collider_state.active_buffer.shape
    BLOCK_DIM = qd.static(_WAVE_COOP_BROADPHASE_BLOCK_DIM)

    func_collision_clear(links_state, links_info, collider_state, static_rigid_sim_config)

    max_broad = collider_info.max_collision_pairs_broad[None]
    # Number of (sorted_i, sorted_j) candidate pairs with sorted_i < sorted_j.
    n_pair_cand = n_geoms * (n_geoms - 1) // 2

    qd.loop_config(block_dim=BLOCK_DIM, force_inline=(gs.backend == gs.amdgpu))
    for flat in range(_B * BLOCK_DIM):
        tid = flat % BLOCK_DIM
        i_b = flat // BLOCK_DIM

        # ===== Phase 1: bitonic sort of geoms by aabb_min[axis 0] =====
        # Each lane initially holds one geom (tid). Lanes >= n_geoms hold a
        # SENTINEL so they sort to the end and don't participate in pairs.
        SENTINEL_KEY = qd.static(0x7FFFFFFF)
        my_g = -1
        my_k = SENTINEL_KEY
        if tid < n_geoms:
            v = geoms_state.aabb_min[tid, i_b][0]
            # Encode positive-biased float as monotonic i32 key. World-space
            # AABB mins for typical sims live well within ±1e3, so the bias
            # of 1e6 keeps things positive and the *1000 multiplier preserves
            # ~1 mm resolution which is enough to prevent stale ties.
            v_biased = v + gs.qd_float(1.0e6)
            my_k = qd.cast(v_biased * gs.qd_float(1000.0), qd.i32)
            my_g = tid

        # Bitonic sort using shfl_xor_i32 (no LDS needed for the sort itself
        # — the AMD shuffle lowering via amdgcn.ds.bpermute provides direct
        # cross-lane access).
        mask = qd.u32(0xFFFFFFFF)
        for stage in qd.static(range(6)):
            for sub in qd.static(range(stage + 1)):
                stride = qd.static(1 << (stage - sub))
                partner = tid ^ stride
                p_k = qd.simt.warp.shfl_xor_i32(mask, my_k, stride)
                p_g = qd.simt.warp.shfl_xor_i32(mask, my_g, stride)
                ascending = ((tid >> qd.static(stage + 1)) & 1) == 0
                i_am_lower = tid < partner
                want_low = (ascending and i_am_lower) or ((not ascending) and (not i_am_lower))
                should_take = (want_low and my_k > p_k) or ((not want_low) and my_k < p_k)
                if should_take:
                    my_k = p_k
                    my_g = p_g

        # Publish sorted geom indices to LDS so any lane can read any
        # sort position when iterating pair candidates.
        sh_sort_g = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_int)
        sh_sort_g[tid] = my_g
        qd.simt.block.sync()

        # ===== Phase 2: SAP-order pair iteration =====
        # Per-block running write count.
        sh_count = qd.simt.block.SharedArray((1,), gs.qd_int)
        if tid == 0:
            sh_count[0] = 0
        qd.simt.block.sync()

        n_batches = (n_pair_cand + BLOCK_DIM - 1) // BLOCK_DIM
        for batch in range(n_batches):
            cand_idx = batch * BLOCK_DIM + tid

            my_flag = 0
            my_i_ga = -1
            my_i_gb = -1

            if cand_idx < n_pair_cand:
                # Decode cand_idx -> (sorted_j, sorted_i) where 0 <= sorted_i < sorted_j.
                # Triangular indexing: cand_idx in [j*(j-1)/2, j*(j+1)/2).
                # Closed form: j = floor((1 + sqrt(1 + 8*cand_idx)) / 2),
                # plus a fix-up for off-by-one from float imprecision.
                disc_f = qd.cast(8 * cand_idx + 1, gs.qd_float)
                sj = qd.cast((1.0 + qd.sqrt(disc_f)) * 0.5, qd.i32)
                if sj * (sj - 1) // 2 > cand_idx:
                    sj = sj - 1
                si = cand_idx - sj * (sj - 1) // 2
                # Now (si, sj) are sort positions; recover original geom indices.
                g_i = sh_sort_g[si]
                g_j = sh_sort_g[sj]
                # SAP active-list filter: g_i is still "active" when g_j enters
                # iff g_i's max event hasn't fired yet, i.e. aabb_max[g_i, 0]
                # >= aabb_min[g_j, 0]. If the filter fails SAP would have
                # removed g_i from active before g_j enters, so no pair.
                max_g_i_0 = geoms_state.aabb_max[g_i, i_b][0]
                min_g_j_0 = geoms_state.aabb_min[g_j, i_b][0]
                if max_g_i_0 >= min_g_j_0:
                    # Canonicalize pair to (smaller_index, larger_index) — SAP
                    # does the same swap before lookup / emit.
                    i_ga = qd.min(g_i, g_j)
                    i_gb = qd.max(g_i, g_j)
                    if collider_info.collision_pair_idx[i_ga, i_gb] != -1:
                        if func_check_collision_valid(
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
                            if func_is_geom_aabbs_overlap(geoms_state, i_ga, i_gb, i_b):
                                my_flag = 1
                                my_i_ga = i_ga
                                my_i_gb = i_gb
                            else:
                                # Clear cached normal for non-overlapping pair (matches SAP).
                                if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)

            # Wave-cooperative inclusive prefix scan via shfl_up_i32.
            scan_val = my_flag
            n = qd.simt.warp.shfl_up_i32(mask, scan_val, 1)
            if tid >= 1:
                scan_val = scan_val + n
            n = qd.simt.warp.shfl_up_i32(mask, scan_val, 2)
            if tid >= 2:
                scan_val = scan_val + n
            n = qd.simt.warp.shfl_up_i32(mask, scan_val, 4)
            if tid >= 4:
                scan_val = scan_val + n
            n = qd.simt.warp.shfl_up_i32(mask, scan_val, 8)
            if tid >= 8:
                scan_val = scan_val + n
            n = qd.simt.warp.shfl_up_i32(mask, scan_val, 16)
            if tid >= 16:
                scan_val = scan_val + n
            n = qd.simt.warp.shfl_up_i32(mask, scan_val, 32)
            if tid >= 32:
                scan_val = scan_val + n
            my_offset = scan_val - my_flag
            batch_total = qd.simt.warp.shfl_sync_i32(mask, scan_val, BLOCK_DIM - 1)

            base = sh_count[0]
            if my_flag == 1:
                slot_out = base + my_offset
                if slot_out < max_broad:
                    collider_state.broad_collision_pairs[slot_out, i_b][0] = my_i_ga
                    collider_state.broad_collision_pairs[slot_out, i_b][1] = my_i_gb
                else:
                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS

            qd.simt.block.sync()
            if tid == 0:
                sh_count[0] = sh_count[0] + batch_total
            qd.simt.block.sync()

        # Publish the final per-env count.
        if tid == 0:
            collider_state.n_broad_pairs[i_b] = sh_count[0]
        qd.simt.block.sync()


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
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = links_info.geom_start.shape[0]

    # Clear collider state
    func_collision_clear(links_state, links_info, collider_state, static_rigid_sim_config)

    # AMD-tuned: block_dim=64 matches wave64 hardware exactly while doubling
    # block count (vs default 128) to better fill MI300X CUs at 8192 envs.
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=64, force_inline=(gs.backend == gs.amdgpu))
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
