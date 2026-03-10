"""
PSEUDO-CODE FOR GRPO TRAINING (Group Relative Policy Optimization):
    p_model   <- policy model (the one we're training)
    ref_model <- reference policy (frozen; for KL regularization)
    reward_model <- frozen model / callable that scores (prompt, resp)
    # [NOTE] No value model / critic. GRPO uses group-relative baselines.

    for i in range(num_iterations):
        prompts <- [B, prompt_len]; Sampled from prompts dataset

        # 1) Rollout: sample a GROUP of responses per prompt
        # Let G = num_generations_per_prompt (a.k.a. group size)
        resps <- [B, G, T]; Rollout from p_model on prompts (G samples each)
        old_logp <- [B, G, T]; Logprobs under rollout policy (snapshot of p_model)
        ref_logp <- [B, G, T]; Logprobs under ref_model for same tokens
        action_mask <- [B, G, T]; 1 on valid action tokens, 0 on padding

        # 2) Compute rewards (often sequence-level)
        # reward_seq is per (prompt, resp) scalar score from reward_model
        reward_seq <- [B, G]; reward_model(prompts, resps)

        # Optional: add formatting penalties, stop-token penalties, etc.
        # reward_seq <- reward_seq + extra_terms

        # If you want token-shaped rewards, place seq reward on last valid token
        reward_tok <- zeros([B, G, T])
        last_idx <- last_valid_index(action_mask)        # [B, G]
        reward_tok[b,g,last_idx[b,g]] += reward_seq[b,g] # scatter add

        # 3) KL term (tokenwise, rollout policy vs reference)
        kl_tok <- [B, G, T]; kl_tok = old_logp - ref_logp

        # 4) Optional KL shaping of reward (same idea as PPO RLHF)
        shaped_reward_tok <- [B, G, T];
                             shaped_reward_tok = reward_tok - beta * kl_tok

        # 5) Construct group-relative advantage (baseline from the GROUP)
        # Most common GRPO: baseline is mean reward within the group (per
        # prompt).
        # Use shaped (sequence) reward or unshaped, depending on your design.
        # Here: use shaped sequence reward = sum over tokens of
        # shaped_reward_tok on valid tokens.
        shaped_reward_seq <- [B, G]
        shaped_reward_seq[b,g] = sum_t(shaped_reward_tok[b,g,t] * action_mask[b,g,t])

        group_mean <- [B, 1]; group_mean[b,1] = mean_g(shaped_reward_seq[b,g])
        group_std  <- [B, 1]; group_std[b,1]  = std_g(shaped_reward_seq[b,g]) + eps

        adv <- [B, G]; adv = (shaped_reward_seq - group_mean) / group_std
            # Alternatively: adv = shaped_reward_seq - group_mean (no normalization)

        # Broadcast to tokens if doing token-level PPO-style objective
        adv_tok <- [B, G, T]; adv_tok = adv[..., None] * action_mask

        batch <- {prompts, resps, old_logp, ref_logp, adv_tok, action_mask}

        # 6) Policy optimization (PPO-style clipped objective, but no value loss)
        for epoch in range(num_epochs_per_rollout):
            for minibatch in iterate_minibatches(batch, mb_size):
                new_logp <- [mb, T]; logprobs from (updated) p_model on minibatch prompts+resps
                old_logp_mb <- [mb, T]; from minibatch old_logp
                adv_tok_mb <- [mb, T]; from minibatch adv_tok
                mask_mb <- [mb, T]; from minibatch action_mask

                # PPO ratio per token
                log_ratio <- new_logp - old_logp_mb
                ratio <- exp(log_ratio)

                # GRPO policy gradient loss (clipped), averaged over valid tokens
                unclipped <- -adv_tok_mb * ratio
                clipped   <- -adv_tok_mb * clip(ratio, 1-eps_clip, 1+eps_clip)
                pg_loss_tok <- max(unclipped, clipped)
                pg_loss <- sum(pg_loss_tok * mask_mb) / sum(mask_mb)

                # Optional entropy bonus
                ent_bonus <- entropy_from_logits(...)   # or from new_logp if available
                loss <- pg_loss - ent_coef * ent_bonus

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(p_model.parameters(), max_grad_norm)  # common
                optimizer.step()

        # Optional: monitor approximate KL to ref, early stop if KL too large
        # approx_kl = mean( (new_logp - ref_logp_mb) * mask_mb )
"""
