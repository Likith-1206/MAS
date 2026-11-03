"""MAPPO agent (Multi-Agent PPO, CTDE): one shared actor for all intersections + a centralised critic.

Same interface as the DQN agents in agents.py (act / state_dict / load), so eval_policy() in
run_experiments.py can evaluate it without changes.

* Actor  (decentralised): obs of ONE intersection (+ optional neighbour info, see neighbors.py) -> logits over phases.
* Critic (centralised, training only): concatenated level-0 obs of ALL intersections + one-hot agent id -> V_i.
* Update: GAE + clipped PPO objective, advantage normalisation, entropy bonus, several epochs per episode.
"""
import numpy as np
import torch, torch.nn as nn
from neighbors import augment, extra_dim
torch.set_num_threads(1)


def mlp(i, h, o, act=nn.Tanh):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


class MAPPO:
    def __init__(self, env, level=2, lr_actor=3e-4, lr_critic=1e-3, gamma=0.95, lam=0.95, clip=0.2,
                 epochs=4, minibatch=256, ent_coef=0.01, vf_coef=0.5, hidden=64, max_grad=0.5):
        self.ag, self.nbrs, self.level = env.agents, env.neighbors, level
        self.n_ag, self.n_act = len(self.ag), env.n_act
        self.obs_dim = env.obs_dim + extra_dim(level)
        self.state_dim = self.n_ag * env.obs_dim + self.n_ag          # all level-0 obs + agent one-hot
        self.base_dim = env.obs_dim
        self.actor = mlp(self.obs_dim, hidden, self.n_act)
        self.critic = mlp(self.state_dim, 2 * hidden, 1)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma, self.lam, self.clip, self.epochs, self.minibatch = gamma, lam, clip, epochs, minibatch
        self.ent_coef, self.vf_coef, self.max_grad = ent_coef, vf_coef, max_grad
        self.eye = np.eye(self.n_ag, dtype=np.float32)
        self._reset_buffer()

    # ---------- helpers ----------
    def _reset_buffer(self):
        self.B = dict(o=[], s=[], a=[], logp=[], v=[], r=[])

    def _actor_obs(self, obs):
        o = augment(obs, self.nbrs, self.level)
        return np.stack([o[a] for a in self.ag]).astype(np.float32)           # (n_ag, obs_dim)

    def _critic_obs(self, obs):
        glob = np.concatenate([obs[a] for a in self.ag]).astype(np.float32)   # level-0 obs of everyone
        return np.concatenate([np.tile(glob, (self.n_ag, 1)), self.eye], axis=1)  # (n_ag, state_dim)

    # ---------- acting ----------
    def act(self, obs, eps=0.0, explore=False):
        """eps is ignored (PPO explores through its stochastic policy). explore=False -> greedy (evaluation)."""
        O = torch.from_numpy(self._actor_obs(obs))
        with torch.no_grad():
            logits = self.actor(O)
            if not explore:
                a = logits.argmax(1)
                return {k: int(a[i]) for i, k in enumerate(self.ag)}
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            S = torch.from_numpy(self._critic_obs(obs))
            v = self.critic(S).squeeze(1)
            self._last = (O.numpy(), S.numpy(), a.numpy(), dist.log_prob(a).numpy(), v.numpy())
        return {k: int(a[i]) for i, k in enumerate(self.ag)}

    def store(self, rewards):
        """Call right after act(explore=True) + env.step with the (possibly mixed) per-agent rewards."""
        o, s, a, lp, v = self._last
        r = np.array([rewards[k] for k in self.ag], np.float32)
        for key, val in zip(("o", "s", "a", "logp", "v", "r"), (o, s, a, lp, v, r)):
            self.B[key].append(val)

    # ---------- learning ----------
    def update(self, last_obs):
        """PPO update at the end of an episode. last_obs bootstraps the value (episode end = time limit, not terminal)."""
        T = len(self.B["r"])
        r, v = np.stack(self.B["r"]), np.stack(self.B["v"])                    # (T, n_ag)
        with torch.no_grad():
            v_last = self.critic(torch.from_numpy(self._critic_obs(last_obs))).squeeze(1).numpy()
        adv, gae = np.zeros_like(r), np.zeros(self.n_ag, np.float32)
        for t in reversed(range(T)):
            v_next = v_last if t == T - 1 else v[t + 1]
            delta = r[t] + self.gamma * v_next - v[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        ret = adv + v
        O = torch.from_numpy(np.concatenate(self.B["o"])); S = torch.from_numpy(np.concatenate(self.B["s"]))
        A = torch.from_numpy(np.concatenate(self.B["a"])); LP = torch.from_numpy(np.concatenate(self.B["logp"]))
        ADV = torch.from_numpy(adv.reshape(-1).astype(np.float32)); RET = torch.from_numpy(ret.reshape(-1).astype(np.float32))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n, stats = len(A), []
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                dist = torch.distributions.Categorical(logits=self.actor(O[idx]))
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                surr = torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[idx])
                ent = dist.entropy().mean()
                loss_a = -surr.mean() - self.ent_coef * ent
                self.opt_a.zero_grad(); loss_a.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad); self.opt_a.step()
                loss_c = self.vf_coef * nn.functional.smooth_l1_loss(self.critic(S[idx]).squeeze(1), RET[idx])
                self.opt_c.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad); self.opt_c.step()
                stats.append((loss_a.item(), loss_c.item(), ent.item()))
        self._reset_buffer()
        return dict(zip(("actor_loss", "critic_loss", "entropy"), np.mean(stats, 0)))

    def state_dict(self): return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict())

    def load(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])

    def hyperparams(self):
        return dict(algo="MAPPO", lr_actor=self.opt_a.param_groups[0]["lr"], lr_critic=self.opt_c.param_groups[0]["lr"],
                    gamma=self.gamma, gae_lambda=self.lam, clip=self.clip, ppo_epochs=self.epochs, minibatch=self.minibatch,
                    ent_coef=self.ent_coef, hidden=self.actor[0].out_features, actor_obs_dim=self.obs_dim, critic_state_dim=self.state_dim)
