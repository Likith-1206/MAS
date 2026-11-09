"""Non-learning baselines for the variable-phase NYC map."""

import numpy as np
import traci


class FixedTime:
    """Cycle through each intersection's valid phases."""

    name = "fixed_time"

    def __init__(self, green_s=30):
        self.green_s = green_s

    def reset(self, env):
        pass

    def act(self, env, obs):
        actions = {}

        for a in env.agents:
            n = env.n_acts[a]
            phase = (env.t // self.green_s) % n
            actions[a] = int(phase)

        return actions


class MaxPressure:
    """Choose the phase with the highest queue pressure."""

    name = "max_pressure"

    def reset(self, env):
        pass

    def act(self, env, obs):
        actions = {}

        for a in env.agents:
            pressures = []

            for phase_movements in env.phase_links[a]:
                pressure = sum(
                    traci.lane.getLastStepHaltingNumber(in_lane)
                    - traci.lane.getLastStepHaltingNumber(out_lane)
                    for in_lane, out_lane in phase_movements
                )
                pressures.append(pressure)

            current = env.cur[a]
            best = int(np.argmax(pressures))

            # Keep the current phase when it ties for maximum pressure.
            if pressures[best] <= pressures[current]:
                actions[a] = current
            else:
                actions[a] = best

        return actions