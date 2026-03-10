import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable


class PPO:
    def __init__(
        self,
        clip=0.2,
        gamma=0.99,
        lam=0.95,
        eps=1e-8,
        normalize_adv=True,
        vf_coef=0.5,
        ent_coef=0.0,
        clip_vloss=False,
        vclip=0.2,
        # NEW: KL coefficient (beta in RLHF PPO)
        kl_coef=0.0,
    ):
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.eps = eps
        self.normalize_adv = normalize_adv
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_vloss = clip_vloss
        self.vclip = vclip
        self.kl_coef = kl_coef

    def masked_mean(self, x, mask, dim=None):
        mask = mask.to(dtype=x.dtype)
        if dim is None:
            denom = mask.sum().clamp_min(self.eps)
            return (x * mask).sum() / denom
        denom = mask.sum(dim=dim).clamp_min(self.eps)
        return (x * mask).sum(dim=dim) / denom

    def kl_sample(self, new_logp, ref_logp, act_mask):
        """
        Returns tokenwise KL sample estimate: log pi_new(a|s) - log pi_ref(a|s)
        new_logp, ref_logp: [B,T] for sampled actions
        """
        act_mask = act_mask.to(dtype=new_logp.dtype)
        kl = (new_logp - ref_logp) * act_mask
        # also return mean KL if you want logging/diagnostics
        kl_mean = self.masked_mean(kl, act_mask)
        return kl, kl_mean

    def shaped_rewards(self, rm_reward, kl_tokenwise, act_mask):
        """
        rm_reward: [B,T]
        kl_tokenwise: [B,T]
        shaped_r = rm_reward - beta * KL
        """
        act_mask = act_mask.to(dtype=rm_reward.dtype)
        return (rm_reward - self.kl_coef * kl_tokenwise) * act_mask

    @torch.no_grad()
    def advantage_estimate(self, rewards, values, dones, mask=None):
        """
        GAE, generalized advantage estimation.
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

        if mask is None:
            mask = torch.ones_like(rewards, dtype=rewards.dtype)

        mask = mask.to(dtype=rewards.dtype)
        dones = dones.to(dtype=rewards.dtype)

        for t in reversed(range(T)):
            valid_t = mask[:, t]
            next_valid = (
                mask[:, t + 1] if t + 1 < T else torch.zeros_like(valid_t)
            )
            next_value = (
                values[:, t + 1]
                if t + 1 < T
                else torch.zeros_like(values[:, t])
            )

            nonterminal = (1.0 - dones[:, t]) * next_valid

            delta = (
                rewards[:, t]
                + self.gamma * next_value * nonterminal
                - values[:, t]
            )
            gae = delta + self.gamma * self.lam * nonterminal * gae

            advantages[:, t] = gae * valid_t
            gae = gae * valid_t

        returns = advantages + values
        return advantages, returns

    def policy_loss(self, new_logp, old_logp, advantages, act_mask):
        """
        $$
        L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[
            \min(\frac{\pi_{\theta}(o_t|q, o_{< t})}{\pi_{\theta_{old}}(o_t|q,
            o_{< t})}A_t, \text{CLIP}(\frac{\pi_{\theta}(o_t|q, o_{< t})}
            {\pi_{\theta_{old}}(o_t|q, o_{< t})}, 1-\varepsilon, 1+\varepsilon)
            A_t) \right]
        $$
        """
        act_mask = act_mask.to(dtype=new_logp.dtype)
        adv = advantages.detach()

        if self.normalize_adv:
            mean = self.masked_mean(adv, act_mask)
            var = self.masked_mean((adv - mean) ** 2, act_mask)
            adv = (adv - mean) / (var.sqrt() + self.eps)

        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
        loss = -torch.min(surr1, surr2)

        return self.masked_mean(loss, act_mask)

    def entropy_bonus(self, entropy, act_mask):
        if self.ent_coef == 0.0 or entropy is None:
            return torch.tensor(0.0, device=act_mask.device)
        return self.ent_coef * self.masked_mean(entropy, act_mask)

    def value_loss(self, new_values, returns, val_mask, old_values=None):
        val_mask = val_mask.to(dtype=new_values.dtype)
        target = returns.detach()

        if self.clip_vloss and (old_values is not None):
            v_clipped = old_values + torch.clamp(
                new_values - old_values, -self.vclip, self.vclip
            )
            vloss1 = (new_values - target) ** 2
            vloss2 = (v_clipped - target) ** 2
            vloss = torch.max(vloss1, vloss2)
        else:
            vloss = (new_values - target) ** 2

        return self.vf_coef * self.masked_mean(vloss, val_mask)

    def total_loss(self, pg_loss, v_loss, ent_bonus):
        return pg_loss + v_loss - ent_bonus


def rollout(*args, **kwargs):
    pass


def logp_of_sampled_tokens(*args, **kwargs):
    pass


def value_preds(*args, **kwargs):
    """
    Per-token value predictions for PPO critic.

    1) Build token states for the full sequence (prompt + response):
       hidden_all: [B, L, H], where L = prompt_len + response_len.
    2) Map each token state to a scalar value with a value head
       (commonly Linear(H, 1), either in `value_model` or attached to
       `policy_model`), giving:
       v_all: [B, L].
    3) Slice response token positions only, returning:
       values: [B, T], where T = response_len.

    These values estimate expected future discounted shaped reward from each
    response timestep and are trained via value loss against PPO returns.
    """
    # A minimal concrete implementation often looks like:
    # - concat prompts/responses -> run backbone -> hidden_all [B,L,H]
    # - v_all = value_head(hidden_all).squeeze(-1)              [B,L]
    # - return v_all at response indices                         [B,T]
    #
    # Note: if policy and value share a backbone, that shared backbone gets
    # gradients from both policy loss and value loss.


def iterate_minibatches(*args, **kwargs):
    pass


"""
PSEUDO-CODE FOR PPO TRAINING:
    p_model <- policy model (the one we're training)
    r_model <- reward model (frozen, provides reward signal)
    v_model <- value model (critic, can be None if integrated into policy)
    reward_model <- reward model (frozen model or callable that provides reward
                    signal)

    for i in range(num_iterations):
        prompts <- [B, prompt_len]; Sampled from prompts dataset
        resps <- [B, T]; Rollout from p_model on prompts
        old_logp <- [B, T]; Calculated from p_model with prompts and resps
        old_values <- [B, T]; Calculated from v_model with prompts and resps
        ref_logp <- [B, T]; Calculated from ref_model with prompts and resps
        reward <- [B, T] or [B]; Calculated from reward_model with prompts and
                  resps; If the reward_model provides a seq-level reward ([B]),
                  place it on the last valid action token in the [B, T] tensor
        kl_tok <- [B, T]; kl_tok = old_logp - ref_logp
        shaped_reward <- [B, T]; shaped_reward = reward - beta * kl_tok
        adv, ret <- [B, T], [B, T]; GAE with shaped_reward and old_values as
                    baseline; ret = adv + old_values, used as value targets
                    for value loss

        batch <- {prompts, resps, old_logp, old_values, ref_logp, adv, ret}

        for epoch in range(num_epochs_per_rollout):
            for minibatch in iterate_minibatches(batch, mb_size):
                new_logp <- [mb, T]; Calculated from (updated) p_model with
                            minibatch prompts
                new_values <- [mb, T]; Calculated from (updated) v_model with
                              minibatch prompts and resps
                pg_loss <- scalar; PPO clipped policy loss from new_logp,
                           old_logp, adv
                v_loss <- scalar; Value loss from new_values, ret, (optionally
                          old_values for clipping)
                ent_bonus <- scalar; Optional entropy bonus from new_logp
                loss <- scalar; total loss = pg_loss + v_loss - ent_bonus

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
"""


def ppo_training(
    ppo: PPO,
    ref_model: nn.Module,
    policy_model: nn.Module,
    reward_model: nn.Module | Callable,
    value_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    prompt_dataset,
    num_iterations=1000,
    num_epochs_per_rollout=4,
    B=64,
    T=32648,
    mb_size=16,
):
    ref_model.eval()
    policy_model.train()

    if value_model is not None:
        # If value_model is a separate network, we need to train it with the
        # optimizer and it should be in train mode.
        value_model.train()

    for _ in range(num_iterations):
        # ---------- (A) COLLECT A ROLLOUT BATCH ----------
        prompts = prompt_dataset.sample(B)

        with torch.no_grad():
            # reponses: [B, T], act_mask: [B,T], dones: [B,T]
            responses, act_mask, dones = rollout(
                policy_model, prompts, max_len=T
            )

            # Behavior policy quantities (fixed for this batch)
            old_logp = logp_of_sampled_tokens(
                policy_model, prompts, responses
            )  # [B,T]
            old_values = value_preds(
                policy_model, value_model, prompts, responses
            )  # [B,T]

            # Reference policy logp (fixed)
            ref_logp = logp_of_sampled_tokens(
                ref_model, prompts, responses
            )  # [B,T]

            # Reward model output
            rm_reward = reward_model(prompts, responses)  # [B,T] or [B]

            # ---- KL shaping + GAE ----
            # Use behavior-policy logp for KL shaping, not new_logp.
            kl_tok, kl_mean = ppo.kl_sample(old_logp, ref_logp, act_mask)

            # If reward model is sequence-level [B], put it on final token
            if rm_reward.dim() == 1:
                rm_tok = torch.zeros_like(old_logp)
                # place reward on last valid action token
                last_idx = (
                    act_mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
                )  # [B]
                rm_tok[torch.arange(B, device=rm_tok.device), last_idx] = (
                    rm_reward
                )
                rm_reward_tok = rm_tok
            # If rm_reward is already [B,T], this is a no-op.
            else:
                rm_reward_tok = rm_reward

            rewards_shaped = ppo.shaped_rewards(
                rm_reward_tok, kl_tok, act_mask
            )

            adv, ret = ppo.advantage_estimate(
                rewards=rewards_shaped,
                values=old_values,
                dones=dones,
                mask=act_mask,
            )

        # Store fixed rollout buffer
        buffer = {
            "prompts": prompts,
            "responses": responses,
            "act_mask": act_mask,
            "dones": dones,
            "old_logp": old_logp,
            "ref_logp": ref_logp,
            "old_values": old_values,
            "adv": adv,
            "ret": ret,
            # optional logging
            "kl_mean_rollout": kl_mean,
            "rm_reward_mean": ppo.masked_mean(
                rm_reward_tok, act_mask
            ).detach(),
            "shaped_reward_mean": ppo.masked_mean(
                rewards_shaped, act_mask
            ).detach(),
        }

        # ---------- (B) PPO OPTIMIZATION ON THIS FIXED BUFFER ----------
        for _ in range(num_epochs_per_rollout):
            for minibatch in iterate_minibatches(buffer, mb_size):
                # Recompute under current policy parameters
                new_logp = logp_of_sampled_tokens(
                    policy_model, minibatch["prompts"], minibatch["responses"]
                )  # [mb,T]
                new_values = value_preds(
                    policy_model,
                    value_model,
                    minibatch["prompts"],
                    minibatch["responses"],
                )  # [mb,T]

                # Optional entropy if you have it (else pass None and ent
                # bonus=0)
                entropy = None  # or: token_entropy(policy_model, ...)

                # ---- PPO losses from your class ----
                pg_loss = ppo.policy_loss(
                    new_logp=new_logp,
                    old_logp=minibatch["old_logp"],
                    advantages=minibatch["adv"],
                    act_mask=minibatch["act_mask"],
                )

                v_loss = ppo.value_loss(
                    new_values=new_values,
                    returns=minibatch["ret"],
                    val_mask=minibatch["act_mask"],  # usually same mask
                    old_values=minibatch[
                        "old_values"
                    ],  # for value clipping if enabled
                )

                ent_b = ppo.entropy_bonus(entropy, minibatch["act_mask"])
                loss = ppo.total_loss(pg_loss, v_loss, ent_b)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
