"""MAPPO for variable-lane, multi-intersection traffic signal control.

Centralized Training, Decentralized Execution:
- One shared actor for all intersections.
- One centralized critic using the observations of all intersections.
- Variable lane counts, padded to max_lanes.
- Variable valid action counts, handled with action masks.
- PPO with GAE, clipped objective, entropy regularization and value loss.

This initial map integration supports communication level 0.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


torch.set_num_threads(1)

MAX_LANES = 10


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class MAPPO:

    def __init__(
        self,
        env,
        level=0,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        epochs=4,
        minibatch=256,
        ent_coef=0.01,
        vf_coef=0.5,
        hidden=64,
        max_grad=0.5,
        max_lanes=MAX_LANES,
    ):
        if level != 0:
            raise ValueError(
                "This map integration currently supports level=0 only."
            )

        self.ag = list(env.agents)
        self.n_ag = len(self.ag)
        self.n_act = int(env.n_act)
        self._n_acts = env.n_acts
        self.level = level
        self.max_lanes = int(max_lanes)

        if self.n_ag == 0:
            raise ValueError("No controllable traffic-signal agents found.")

        if self.n_act <= 0:
            raise ValueError("The environment has no available actions.")

        self.lane_counts = {
            a: len(env.lanes[a]) for a in self.ag
        }

        for a, count in self.lane_counts.items():
            if count <= 0:
                raise ValueError(f"Intersection {a} has no incoming lanes.")

            if count > self.max_lanes:
                raise ValueError(
                    f"Intersection {a} has {count} lanes; "
                    f"the configured maximum is {self.max_lanes}."
                )

        # Observation:
        # 10 waiting values + 10 vehicle counts + 10 lane masks
        # + n_act phase values + 1 time value.
        self.base_dim = 3 * self.max_lanes + self.n_act + 1

        # Actor uses only the local intersection observation.
        self.obs_dim = self.base_dim

        # Centralized critic sees every intersection's observation
        # and the identity of the intersection being evaluated.
        self.state_dim = self.n_ag * self.base_dim + self.n_ag

        self.actor = mlp(
            self.obs_dim, hidden, self.n_act
        )

        self.critic = mlp(
            self.state_dim, 2 * hidden, 1
        )

        self.opt_a = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor
        )

        self.opt_c = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad = max_grad
        self.hidden = hidden

        self.eye = np.eye(self.n_ag, dtype=np.float32)

        self._reset_buffer()

        print("MAPPO initialized successfully.")
        print("Number of agents:", self.n_ag)
        print("Maximum incoming lanes:", self.max_lanes)
        print("Maximum actions:", self.n_act)
        print("Padded observation dimension:", self.base_dim)
        print("Actor input dimension:", self.obs_dim)
        print("Centralized critic input dimension:", self.state_dim)

    # ---------------------------------------------------------
    # Observation preprocessing
    # ---------------------------------------------------------

    def _prepare_observations(self, obs):
        prepared = {}

        for agent in self.ag:
            if agent not in obs:
                raise ValueError(
                    f"Observation missing for intersection {agent}."
                )

            raw = np.asarray(
                obs[agent], dtype=np.float32
            ).reshape(-1)

            n = self.lane_counts[agent]

            # The environment provides a phase vector of length
            # env.n_acts[agent], which may differ between agents.
            phase_count = int(self._n_acts[agent])
            expected = 2 * n + phase_count + 1

            if len(raw) != expected:
                raise ValueError(
                    f"Invalid observation for {agent}: "
                    f"expected {expected}, received {len(raw)}."
                )

            waiting = raw[:n]
            counts = raw[n:2 * n]
            phase = raw[2 * n:2 * n + phase_count]
            time_value = raw[-1:]

            waiting_pad = np.zeros(self.max_lanes, dtype=np.float32)
            counts_pad = np.zeros(self.max_lanes, dtype=np.float32)
            lane_mask = np.zeros(self.max_lanes, dtype=np.float32)
            phase_pad = np.zeros(self.n_act, dtype=np.float32)

            waiting_pad[:n] = waiting
            counts_pad[:n] = counts
            lane_mask[:n] = 1.0
            phase_pad[:phase_count] = phase

            prepared[agent] = np.concatenate([
                waiting_pad,
                counts_pad,
                lane_mask,
                phase_pad,
                time_value,
            ]).astype(np.float32)

        return prepared

    def _actor_obs(self, prepared):
        return np.stack([
            prepared[a] for a in self.ag
        ]).astype(np.float32)

    def _critic_obs(self, prepared):
        global_state = np.concatenate([
            prepared[a] for a in self.ag
        ]).astype(np.float32)

        states = []

        for i in range(self.n_ag):
            states.append(
                np.concatenate([global_state, self.eye[i]])
            )

        return np.stack(states).astype(np.float32)

    # ---------------------------------------------------------
    # Action masks
    # ---------------------------------------------------------

    def _get_action_masks(self, env):
        masks_dict = env.action_masks()

        masks = []

        for agent in self.ag:
            if agent not in masks_dict:
                raise ValueError(
                    f"Action mask missing for intersection {agent}."
                )

            mask = np.asarray(
                masks_dict[agent], dtype=np.float32
            ).reshape(-1)

            if len(mask) != self.n_act:
                raise ValueError(
                    f"Invalid action mask size for {agent}: "
                    f"expected {self.n_act}, received {len(mask)}."
                )

            if not np.any(mask > 0):
                raise ValueError(
                    f"Intersection {agent} has no valid actions."
                )

            masks.append(mask)

        return np.stack(masks).astype(np.float32)

    @staticmethod
    def _masked_logits(logits, masks):
        return logits.masked_fill(masks <= 0, -1e9)

    # ---------------------------------------------------------
    # Experience buffer
    # ---------------------------------------------------------

    def _reset_buffer(self):
        self.B = {
            "o": [],
            "s": [],
            "a": [],
            "logp": [],
            "v": [],
            "r": [],
            "mask": [],
        }

    # ---------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------

    def act(self, obs, eps=0.0, explore=False, env=None):
        """
        Select one valid action per intersection.

        explore=True:
            Sample from the masked stochastic policy and save
            information needed for the PPO update.

        explore=False:
            Select the highest-probability valid action.
        """
        if env is None:
            raise ValueError(
                "Pass env=environment when calling MAPPO.act()."
            )

        prepared = self._prepare_observations(obs)
        actor_input = self._actor_obs(prepared)
        critic_input = self._critic_obs(prepared)

        masks = self._get_action_masks(env)

        O = torch.from_numpy(actor_input)
        S = torch.from_numpy(critic_input)
        M = torch.from_numpy(masks)

        with torch.no_grad():
            logits = self.actor(O)
            logits = self._masked_logits(logits, M)
            dist = Categorical(logits=logits)

            if explore:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.critic(S).squeeze(1)

                self._last = (
                    O.numpy().copy(),
                    S.numpy().copy(),
                    actions.numpy().copy(),
                    log_probs.numpy().copy(),
                    values.numpy().copy(),
                    M.numpy().copy(),
                )
            else:
                actions = logits.argmax(dim=1)

        return {
            agent: int(actions[i].item())
            for i, agent in enumerate(self.ag)
        }

    # ---------------------------------------------------------
    # Store transitions
    # ---------------------------------------------------------

    def store(self, rewards):
        if not hasattr(self, "_last"):
            raise RuntimeError(
                "Call act(explore=True) before store()."
            )

        o, s, a, lp, v, mask = self._last

        r = np.asarray([
            rewards[agent] for agent in self.ag
        ], dtype=np.float32)

        for key, value in zip(
            ("o", "s", "a", "logp", "v", "r", "mask"),
            (o, s, a, lp, v, r, mask),
        ):
            self.B[key].append(value)

        del self._last

    # ---------------------------------------------------------
    # GAE and PPO update
    # ---------------------------------------------------------

    def update(self, last_obs):
        T = len(self.B["r"])

        if T == 0:
            raise RuntimeError("No training transitions available.")

        rewards = np.stack(self.B["r"])
        values = np.stack(self.B["v"])

        final_prepared = self._prepare_observations(last_obs)
        final_state = torch.from_numpy(
            self._critic_obs(final_prepared)
        )

        with torch.no_grad():
            v_last = self.critic(final_state).squeeze(1).numpy()

        # GAE: bootstrap from the final observation because
        # the episode ends at the simulation time limit.
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(self.n_ag, dtype=np.float32)

        for t in reversed(range(T)):
            next_value = v_last if t == T - 1 else values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_value
                - values[t]
            )

            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae

        returns = advantages + values

        # Flatten time and agent dimensions.
        O = torch.from_numpy(
            np.concatenate(self.B["o"], axis=0)
        )

        S = torch.from_numpy(
            np.concatenate(self.B["s"], axis=0)
        )

        A = torch.from_numpy(
            np.concatenate(self.B["a"], axis=0)
        ).long()

        LP = torch.from_numpy(
            np.concatenate(self.B["logp"], axis=0)
        )

        MASK = torch.from_numpy(
            np.concatenate(self.B["mask"], axis=0)
        )

        ADV = torch.from_numpy(
            advantages.reshape(-1).astype(np.float32)
        )

        RET = torch.from_numpy(
            returns.reshape(-1).astype(np.float32)
        )

        ADV = (
            ADV - ADV.mean()
        ) / (
            ADV.std(unbiased=False) + 1e-8
        )

        n = len(A)
        stats = []

        for _ in range(self.epochs):
            permutation = torch.randperm(n)

            for start in range(0, n, self.minibatch):
                indices = permutation[
                    start:start + self.minibatch
                ]

                # Apply the SAME stored action masks during PPO.
                logits = self.actor(O[indices])
                logits = self._masked_logits(
                    logits, MASK[indices]
                )

                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(A[indices])

                ratio = torch.exp(
                    new_log_probs - LP[indices]
                )

                surrogate1 = ratio * ADV[indices]

                surrogate2 = torch.clamp(
                    ratio,
                    1 - self.clip,
                    1 + self.clip,
                ) * ADV[indices]

                surrogate = torch.min(
                    surrogate1, surrogate2
                )

                entropy = dist.entropy().mean()

                actor_loss = (
                    -surrogate.mean()
                    - self.ent_coef * entropy
                )

                self.opt_a.zero_grad()
                actor_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad
                )

                self.opt_a.step()

                # Centralized critic update.
                predicted_values = self.critic(
                    S[indices]
                ).squeeze(1)

                critic_loss = (
                    self.vf_coef
                    * nn.functional.smooth_l1_loss(
                        predicted_values, RET[indices]
                    )
                )

                self.opt_c.zero_grad()
                critic_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad
                )

                self.opt_c.step()

                stats.append((
                    actor_loss.item(),
                    critic_loss.item(),
                    entropy.item(),
                ))

        self._reset_buffer()

        mean_stats = np.mean(stats, axis=0)

        return {
            "actor_loss": float(mean_stats[0]),
            "critic_loss": float(mean_stats[1]),
            "entropy": float(mean_stats[2]),
        }

    # ---------------------------------------------------------
    # Save and load
    # ---------------------------------------------------------

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])

    # ---------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------

    def hyperparams(self):
        return {
            "algo": "MAPPO",
            "number_of_agents": self.n_ag,
            "max_incoming_lanes": self.max_lanes,
            "number_of_phases": self.n_act,
            "communication_level": self.level,
            "lr_actor": self.opt_a.param_groups[0]["lr"],
            "lr_critic": self.opt_c.param_groups[0]["lr"],
            "gamma": self.gamma,
            "gae_lambda": self.lam,
            "clip": self.clip,
            "ppo_epochs": self.epochs,
            "minibatch": self.minibatch,
            "ent_coef": self.ent_coef,
            "hidden": self.hidden,
            "base_obs_dim": self.base_dim,
            "actor_obs_dim": self.obs_dim,
            "critic_state_dim": self.state_dim,
        }