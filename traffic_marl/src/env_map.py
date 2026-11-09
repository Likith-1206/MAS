
"""Environment v2: one agent per controllable traffic light of ANY map.

Usage:
    python src/env_map.py maps/nyc16

Reward modes:
    queue:
        Original reward = -(halting cars on incoming lanes) / 10

    pressure_delta:
        Reward = (selected-phase queue cost before - after) / 10

The pressure_delta reward measures the reduction in halted vehicles
on the incoming and corresponding outgoing lanes of the selected phase.
"""

import heapq
import math
import os
import sys
import tempfile

import numpy as np

try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)
except ImportError:
    pass

import traci
import sumolib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenario_gen as G


DELTA, YELLOW = 5, 2
EPISODE_LEN = G.EPISODE_LEN

COMPASS = ["N", "E", "S", "W"]
SCENARIOS = list(G.PATTERNS)


def _bearing_key(edge):
    """Return the compass bearing of an incoming road approach."""

    shp = edge.getShape()
    end = shp[-1]
    ref = shp[0]

    for p in reversed(shp[:-1]):
        ref = p
        if math.hypot(p[0] - end[0], p[1] - end[1]) >= 15.0:
            break

    ang = math.degrees(
        math.atan2(ref[0] - end[0], ref[1] - end[1])
    ) % 360.0

    return (ang + 45.0) % 360.0, ang


class SumoMapEnv:

    def __init__(
        self,
        map_dir,
        neighbor_dist=300.0,
        demand_mult=1.0,
        reward_mode="queue"
    ):
        """Initialize the SUMO map environment.

        reward_mode:
            "queue"         - original incoming queue penalty
            "pressure_delta" - selected-phase queue reduction reward
        """

        self.map_dir = map_dir
        self.net_file = os.path.join(map_dir, "net.net.xml")
        self.demand_mult = demand_mult
        self.neighbor_limit_m = neighbor_dist

        if reward_mode not in ("queue", "pressure_delta"):
            raise ValueError(
                "reward_mode must be 'queue' or 'pressure_delta'"
            )

        self.reward_mode = reward_mode

        self.net = sumolib.net.readNet(
            self.net_file,
            withPrograms=True
        )

        self.info = G.MapInfo(map_dir)

        self.tmp = tempfile.mkdtemp()
        self.running = False
        self.route_file = None

        ctrl = [
            t["id"]
            for t in self.info.report["traffic_lights"]
            if t["controllable"]
        ]

        self.agents = sorted(ctrl)

        if not self.agents:
            raise ValueError(
                "this map has no controllable traffic light (>= 2 green phases)"
            )

        self._static()

        self.n_act = max(self.n_acts.values())
        self.max_lanes = max(
            len(v) for v in self.lanes.values()
        )

        self.max_obs_dim = max(self.obs_dims.values())

        self.padded_dim = (
            2 * self.max_lanes + self.n_act + 1
        )

    # ---------------------------------------------------------
    # Static topology
    # ---------------------------------------------------------

    def _static(self):

        self.green = {}
        self.lanes = {}
        self.lane_cap = {}
        self.phase_links = {}

        self.phase_dirs = {}
        self.n_acts = {}
        self.obs_dims = {}
        self.obs_names = {}

        self.tls_nodes = {}
        self.centroid = {}

        for a in self.agents:

            tls = self.net.getTLS(a)
            conns = tls.getConnections()

            prog = next(iter(tls.getPrograms().values()))

            self.green[a] = [
                p.state
                for p in prog.getPhases()
                if any(c in "Gg" for c in p.state)
                and not any(c in "yY" for c in p.state)
            ]

            link = {}

            for inl, outl, idx in conns:
                link[idx] = (inl, outl)

            lane_objs = {
                inl.getID(): inl
                for inl, _, _ in conns
            }

            key = {}

            for lid, ln in lane_objs.items():

                k, ang = _bearing_key(ln.getEdge())

                key[lid] = (
                    k,
                    ln.getEdge().getID(),
                    ln.getIndex(),
                    ang
                )

            order = sorted(
                lane_objs,
                key=lambda l: key[l][:3]
            )

            self.lanes[a] = order

            for l in order:

                self.lane_cap[l] = max(
                    1.0,
                    lane_objs[l].getLength() / 7.5
                )

            self.phase_links[a] = []
            self.phase_dirs[a] = []

            for st in self.green[a]:

                pl = [
                    (
                        link[i][0].getID(),
                        link[i][1].getID()
                    )
                    for i, ch in enumerate(st)
                    if ch in "Gg" and i in link
                ]

                self.phase_links[a].append(pl)

                dirs = sorted(
                    {
                        COMPASS[
                            int(key[i][3] + 45.0) % 360 // 90
                        ]
                        for i, _ in pl
                    },
                    key=COMPASS.index
                )

                self.phase_dirs[a].append(dirs)

            self.n_acts[a] = len(self.green[a])

            self.obs_dims[a] = (
                2 * len(order) + self.n_acts[a] + 1
            )

            names = []

            lane_names = [
                f"{COMPASS[int(key[l][3] + 45.0) % 360 // 90]}_{l}"
                for l in order
            ]

            names += [
                f"{n}_waiting"
                for n in lane_names
            ]

            names += [
                f"{n}_cars"
                for n in lane_names
            ]

            names += [
                f"phase{p}_green"
                for p in range(self.n_acts[a])
            ]

            names += ["time_since_change"]

            self.obs_names[a] = names

            self.tls_nodes[a] = sorted(
                {
                    ln.getEdge().getToNode().getID()
                    for ln in lane_objs.values()
                }
            )

            pts = [
                self.net.getNode(n).getCoord()
                for n in self.tls_nodes[a]
            ]

            self.centroid[a] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts)
            )

        self._neighbour_graph()

    # ---------------------------------------------------------
    # Neighbour graph
    # ---------------------------------------------------------

    def _neighbour_graph(self, max_dist=None):

        max_dist = max_dist or self.neighbor_limit_m

        owner = {
            n: a
            for a in self.agents
            for n in self.tls_nodes[a]
        }

        down = {
            a: {}
            for a in self.agents
        }

        for a in self.agents:

            heap = [
                (0.0, n)
                for n in self.tls_nodes[a]
            ]

            heapq.heapify(heap)
            seen = set()

            while heap:

                d, nid = heapq.heappop(heap)

                if nid in seen:
                    continue

                seen.add(nid)

                node = self.net.getNode(nid)

                if owner.get(nid) not in (None, a):

                    b = owner[nid]

                    down[a][b] = min(
                        d,
                        down[a].get(b, 1e9)
                    )

                    continue

                for e in sorted(
                    node.getOutgoing(),
                    key=lambda e: e.getID()
                ):

                    nd = d + e.getLength()

                    if (
                        nd <= max_dist
                        and e.getToNode().getID() not in seen
                    ):

                        heapq.heappush(
                            heap,
                            (nd, e.getToNode().getID())
                        )

        self.downstream = {
            a: sorted(down[a])
            for a in self.agents
        }

        self.upstream = {
            a: sorted(
                b for b in self.agents
                if a in down[b]
            )
            for a in self.agents
        }

        self.neighbor_dist = {}

        for a in self.agents:

            for b in (
                set(down[a])
                | {
                    x for x in self.agents
                    if a in down[x]
                }
            ):

                self.neighbor_dist[(a, b)] = min(
                    down[a].get(b, 1e9),
                    down[b].get(a, 1e9)
                )

        def bearing(a, b):

            (ax, ay) = self.centroid[a]
            (bx, by) = self.centroid[b]

            return math.degrees(
                math.atan2(bx - ax, by - ay)
            ) % 360.0

        self.neighbors = {
            a: sorted(
                {
                    b
                    for (x, b) in self.neighbor_dist
                    if x == a
                },
                key=lambda b: (
                    (bearing(a, b) + 45.0) % 360.0,
                    b
                )
            )
            for a in self.agents
        }

    # ---------------------------------------------------------
    # Actions and padded observations
    # ---------------------------------------------------------

    def action_mask(self, a):

        m = np.zeros(
            self.n_act,
            dtype=np.float32
        )

        m[:self.n_acts[a]] = 1.0

        return m

    def action_masks(self):

        return {
            a: self.action_mask(a)
            for a in self.agents
        }

    def pad_obs(self, obs):

        """Pad every agent observation to one common length."""

        out = {}

        for a, v in obs.items():

            n = len(self.lanes[a])
            k = self.n_acts[a]

            p = np.zeros(
                self.padded_dim,
                dtype=np.float32
            )

            p[:n] = v[:n]

            p[
                self.max_lanes:
                self.max_lanes + n
            ] = v[n:2 * n]

            p[
                2 * self.max_lanes:
                2 * self.max_lanes + k
            ] = v[2 * n:2 * n + k]

            p[-1] = v[-1]

            out[a] = p

        return out

    # ---------------------------------------------------------
    # Episode control
    # ---------------------------------------------------------

    def _make_routes(self, scenario, seed):

        if scenario not in G.PATTERNS:

            raise ValueError(
                f"unknown scenario {scenario!r}; "
                f"choose from {G.PATTERNS}"
            )

        try:

            scale = (
                G.scale_for(self.map_dir, scenario)
                * self.demand_mult
            )

        except FileNotFoundError:

            raise FileNotFoundError(
                f"{self.map_dir}/demand.json not found: "
                "run python src/scenario_gen.py "
                f"{self.map_dir} --calibrate-all first"
            )

        path = os.path.join(
            self.tmp,
            "r.rou.xml"
        )

        G.build_route_file(
            self.map_dir,
            scenario,
            seed,
            path,
            scale,
            self.info
        )

        return path

    def reset(
        self,
        scenario="uniform",
        seed=0,
        tripinfo_file=None,
        route_file=None,
        own_programs=False
    ):

        """Start a new simulation episode."""

        if self.running:

            traci.close()
            self.running = False

        self.route_file = (
            route_file
            or self._make_routes(scenario, seed)
        )

        args = [
            "sumo",
            "-n", self.net_file,
            "-r", self.route_file,
            "--begin", "0",
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--seed", str(seed),
            "--time-to-teleport", "-1"
        ]

        if tripinfo_file:

            args += [
                "--tripinfo-output",
                tripinfo_file
            ]

        traci.start(args)

        self.running = True
        self.own_programs = own_programs

        self.t = 0

        self.cur = {
            a: 0
            for a in self.agents
        }

        self.since = {
            a: 0
            for a in self.agents
        }

        if not own_programs:

            for a in self.agents:

                traci.trafficlight.setRedYellowGreenState(
                    a,
                    self.green[a][0]
                )

        self.dep_time = {}
        self.durations = []

        self.m = dict(
            halt_vs=0.0,
            veh_s=0.0,
            departed=0,
            arrived=0,
            qsum=0.0,
            qn=0,
            maxq=0.0
        )

        return self._obs()

    def close(self):

        if self.running:

            traci.close()
            self.running = False

    def _sim(self, secs):

        for _ in range(secs):

            traci.simulationStep()
            self.t += 1

            for v in traci.simulation.getDepartedIDList():

                self.dep_time[v] = self.t
                self.m["departed"] += 1

            for v in traci.simulation.getArrivedIDList():

                self.m["arrived"] += 1

                if v in self.dep_time:

                    self.durations.append(
                        self.t - self.dep_time.pop(v)
                    )

            self.m["veh_s"] += (
                traci.vehicle.getIDCount()
            )

    @staticmethod
    def _yellow(cur, new):

        return "".join(
            "y" if (c in "Gg" and n not in "Gg") else c
            for c, n in zip(cur, new)
        )

    # ---------------------------------------------------------
    # NEW: selected-phase queue cost for alternative reward
    # ---------------------------------------------------------

    def _phase_queue_cost(self, agent, phase):

        """Sum halted vehicles on incoming and outgoing lanes
        connected by the selected phase."""

        movements = self.phase_links[agent][phase]

        cost = 0.0

        for in_lane, out_lane in movements:

            q_in = traci.lane.getLastStepHaltingNumber(
                in_lane
            )

            q_out = traci.lane.getLastStepHaltingNumber(
                out_lane
            )

            cost += q_in + q_out

        return float(cost)

    # ---------------------------------------------------------
    # Environment step
    # ---------------------------------------------------------

    def step(self, actions):

        """Execute one 5-second decision interval."""

        selected_phases = None
        cost_before = None

        # Record the phase whose queue cost will be measured.
        if self.reward_mode == "pressure_delta":

            if self.own_programs:

                selected_phases = dict(self.cur)

            else:

                for a in self.agents:

                    if (
                        a not in actions
                        or not (
                            0 <= actions[a] < self.n_acts[a]
                        )
                    ):

                        raise ValueError(
                            f"agent {a}: action "
                            f"{actions.get(a)!r} is not in "
                            f"0..{self.n_acts[a] - 1}"
                        )

                selected_phases = {
                    a: int(actions[a])
                    for a in self.agents
                }

            # Measure queue cost BEFORE simulation.
            cost_before = {
                a: self._phase_queue_cost(
                    a,
                    selected_phases[a]
                )
                for a in self.agents
            }

        # -----------------------------------------------------
        # Original signal-control logic
        # -----------------------------------------------------

        if self.own_programs:

            sw = []

            self._sim(DELTA)

            for a in self.agents:

                st = (
                    traci.trafficlight.getRedYellowGreenState(a)
                )

                if st in self.green[a]:

                    k = self.green[a].index(st)

                    self.since[a] = (
                        self.since[a] + DELTA
                        if k == self.cur[a]
                        else DELTA
                    )

                    self.cur[a] = k

                else:

                    self.since[a] += DELTA

        else:

            for a in self.agents:

                if (
                    a not in actions
                    or not (
                        0 <= actions[a] < self.n_acts[a]
                    )
                ):

                    raise ValueError(
                        f"agent {a}: action "
                        f"{actions.get(a)!r} is not in "
                        f"0..{self.n_acts[a] - 1}"
                    )

            sw = [
                a
                for a in self.agents
                if actions[a] != self.cur[a]
            ]

            if sw:

                for a in sw:

                    traci.trafficlight.setRedYellowGreenState(
                        a,
                        self._yellow(
                            self.green[a][self.cur[a]],
                            self.green[a][actions[a]]
                        )
                    )

                self._sim(YELLOW)

                for a in sw:

                    traci.trafficlight.setRedYellowGreenState(
                        a,
                        self.green[a][actions[a]]
                    )

                    self.cur[a] = actions[a]

                self._sim(DELTA - YELLOW)

            else:

                self._sim(DELTA)

            for a in self.agents:

                self.since[a] = (
                    (DELTA - YELLOW)
                    if a in sw
                    else self.since[a] + DELTA
                )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        halt = {
            a: self._halting(a)
            for a in self.agents
        }

        tot = sum(
            h.sum()
            for h in halt.values()
        )

        self.m["halt_vs"] += tot * DELTA

        self.m["qsum"] += np.mean([
            h.sum()
            for h in halt.values()
        ])

        self.m["qn"] += 1

        self.m["maxq"] = max(
            self.m["maxq"],
            max(h.max() for h in halt.values())
        )

        # -----------------------------------------------------
        # NEW: choose the requested reward function
        # -----------------------------------------------------

        if self.reward_mode == "queue":

            # Original reward: penalize incoming halted cars.
            rewards = {
                a: -halt[a].sum() / 10.0
                for a in self.agents
            }

        else:

            # Alternative reward: reward reduction in the
            # selected phase's incoming + outgoing queue cost.
            cost_after = {
                a: self._phase_queue_cost(
                    a,
                    selected_phases[a]
                )
                for a in self.agents
            }

            rewards = {
                a: (
                    cost_before[a] - cost_after[a]
                ) / 10.0
                for a in self.agents
            }

        done = self.t >= EPISODE_LEN

        tot_cars = {}
        acc_wait = {}

        for a in self.agents:

            tot_cars[a] = int(
                sum(
                    traci.lane.getLastStepVehicleNumber(l)
                    for l in self.lanes[a]
                )
            )

            acc_wait[a] = float(
                sum(
                    traci.vehicle.getAccumulatedWaitingTime(v)
                    for l in self.lanes[a]
                    for v in traci.lane.getLastStepVehicleIDs(l)
                )
            )

        queue = {
            a: float(halt[a].sum())
            for a in self.agents
        }

        info = {
            "queue": queue,
            "waiting_cars": dict(queue),
            "total_cars": tot_cars,
            "acc_wait_s": acc_wait,
            "phase": dict(self.cur),
            "yellow": {
                a: a in sw
                for a in self.agents
            },
            "time": self.t
        }

        return self._obs(), rewards, done, info

    # ---------------------------------------------------------
    # Observations and metrics
    # ---------------------------------------------------------

    def _halting(self, a):

        return np.array(
            [
                traci.lane.getLastStepHaltingNumber(l)
                for l in self.lanes[a]
            ],
            dtype=np.float32
        )

    def _obs(self):

        obs = {}

        for a in self.agents:

            h = self._halting(a)

            c = np.array(
                [
                    traci.lane.getLastStepVehicleNumber(l)
                    for l in self.lanes[a]
                ],
                dtype=np.float32
            )

            cap = np.array(
                [
                    self.lane_cap[l]
                    for l in self.lanes[a]
                ],
                dtype=np.float32
            )

            ph = np.zeros(
                self.n_acts[a],
                dtype=np.float32
            )

            ph[self.cur[a]] = 1.0

            obs[a] = np.concatenate([
                np.clip(h / cap, 0, 1),
                np.clip(c / cap, 0, 1),
                ph,
                [min(self.since[a], 60) / 60.0]
            ]).astype(np.float32)

        return obs

    def metrics(self):

        m = self.m

        dep = max(1, m["departed"])

        return dict(
            avg_wait_s=m["halt_vs"] / dep,
            avg_queue=m["qsum"] / max(1, m["qn"]),
            max_queue=m["maxq"],
            throughput=m["arrived"],
            total_travel_h=m["veh_s"] / 3600.0,
            avg_trip_s=(
                float(np.mean(self.durations))
                if self.durations
                else float("nan")
            ),
            in_network=traci.vehicle.getIDCount(),
            departed=m["departed"]
        )

    def describe(self):

        return [
            dict(
                agent=a,
                lanes=len(self.lanes[a]),
                actions=self.n_acts[a],
                obs_dim=self.obs_dims[a],
                serves=self.phase_dirs[a],
                neighbors=self.neighbors[a]
            )
            for a in self.agents
        ]


# -------------------------------------------------------------
# Standalone environment test
# -------------------------------------------------------------

if __name__ == "__main__":

    import time

    d = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "maps/nyc16"
    )

    env = SumoMapEnv(d)

    print(
        f"map {d}: {len(env.agents)} agents, "
        f"up to {env.max_lanes} lanes and "
        f"{env.n_act} actions, "
        f"observation {min(env.obs_dims.values())}-"
        f"{env.max_obs_dim} numbers "
        f"(padded {env.padded_dim})"
    )

    for r in env.describe():

        print(
            f"  {r['agent'][:34]:>34}: "
            f"{r['lanes']:>2} lanes, "
            f"{r['actions']} actions "
            f"{r['serves']}, "
            f"neighbours {len(r['neighbors'])}"
        )

    for scn in SCENARIOS:

        t0 = time.time()

        obs = env.reset(scn, 9001)
        done = False

        rng = np.random.default_rng(0)

        while not done:

            actions = {
                a: int(rng.integers(env.n_acts[a]))
                for a in env.agents
            }

            obs, rewards, done, info = env.step(actions)

        m = env.metrics()

        print(
            f"  {scn:>10}: random agent, "
            f"one episode {time.time() - t0:4.1f} s, "
            f"avg wait {m['avg_wait_s']:.1f} s, "
            f"throughput {m['throughput']}, "
            f"in network {m['in_network']}"
        )

    env.close()
