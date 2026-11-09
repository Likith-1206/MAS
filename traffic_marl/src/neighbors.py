
"""Neighbour-information features for MAPPO.

Supports variable incoming lane counts, with a maximum of 10 lanes
per intersection.

Expected input observation format (33 values when n_act=2):
    0:10   - normalized waiting information
    10:20  - normalized vehicle counts
    20:30  - lane-presence mask (1 = lane exists, 0 = padding)
    30:30+n_act - current phase one-hot
    Last   - normalized time since phase change

Communication levels:
    Level 0: own observation only
    Level 1: own observation + neighbour mean queue
    Level 2: own observation + neighbour mean queue + neighbour phase

Neighbour slots: N, E, S, W.
Missing neighbours are represented by zeros.
"""

import numpy as np

MAX_LANES = 10
N_ACT = 2


def base_dim(n_act=N_ACT, max_lanes=MAX_LANES):
    """Size of a padded observation for one intersection."""
    return 3 * max_lanes + n_act + 1


def slot_dim(level, n_act=N_ACT):
    """Number of additional features per neighbour."""
    if level == 0:
        return 0
    return 1 + (n_act if level >= 2 else 0)


def extra_dim(level, n_act=N_ACT):
    """Total additional features from four neighbour slots."""
    return 4 * slot_dim(level, n_act)


def augment(obs, neighbors, level, n_lanes=MAX_LANES,
            n_act=N_ACT):
    """
    Add neighbour information to padded observations.

    Parameters
    ----------
    obs : dict
        {agent_id: fixed-size padded observation}
    neighbors : dict
        {agent_id: [north, east, south, west]}
    level : int
        Communication level: 0, 1, or 2.
    n_lanes : int
        Maximum number of lane slots.
    n_act : int
        Number of traffic-light phases.

    Returns
    -------
    dict
        Observations with neighbour features appended.
    """
    if level not in (0, 1, 2):
        raise ValueError("Communication level must be 0, 1, or 2.")

    if level == 0:
        return obs

    out = {}

    phase_start = 3 * n_lanes
    phase_end = phase_start + n_act
    mask_start = 2 * n_lanes
    mask_end = 3 * n_lanes

    for agent, own in obs.items():
        parts = [own]

        for neighbour in neighbors[agent]:

            # No neighbour at this grid position.
            if neighbour is None:
                parts.append(
                    np.zeros(slot_dim(level, n_act), dtype=np.float32)
                )
                continue

            neighbour_obs = obs[neighbour]

            # Extract only real incoming lanes.
            waiting = neighbour_obs[:n_lanes]
            lane_mask = neighbour_obs[mask_start:mask_end]

            valid = lane_mask > 0.5

            if np.any(valid):
                mean_queue = float(np.mean(waiting[valid]))
            else:
                mean_queue = 0.0

            features = [mean_queue]

            # Add the neighbour's current phase for level 2.
            if level >= 2:
                phase = neighbour_obs[phase_start:phase_end]
                features.extend(phase.tolist())

            parts.append(np.asarray(features, dtype=np.float32))

        out[agent] = np.concatenate(parts).astype(np.float32)

    return out
