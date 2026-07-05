"""Loss head implementing eq. (2) of the analysis doc:

    L(theta) = sum_t 1/2 (V_theta(sg[h_t]) - sg[y_t^lambda])^2      (critic)
               - beta * (sum_t gamma^t r_t + gamma^H V_bar(h_H))    (actor)

One scalar, one backward; the routing is done by the stop-gradients placed in
the model (critic inputs / TD targets / frozen terminal head).

Also computes the diagnostics recommended in §4:
  - corr(delta_t, dr_t): should stay ~0 (Phase-2 self-destruction check)
  - equilibrium consistency |V(h_H) - r_H/(1-gamma)|
  - mean ||u_t|| / ||h_t|| (BPTT explosion watch)
"""
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    ce = F.cross_entropy(logits.to(torch.float32).reshape(-1, logits.shape[-1]), labels.to(torch.long).reshape(-1),
                         ignore_index=ignore_index, reduction="none").view(labels.shape)
    return ce


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return (a * b).sum() / denom.clamp_min(1e-12)


class RVSACLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, gamma: float, lam: float, beta: float, critic_coef: float = 1.0,
                 eq_anchor_coef: float = 0.0):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.gamma = gamma
        self.lam = lam
        self.critic_coef = critic_coef
        self.eq_anchor_coef = eq_anchor_coef
        # Registered as a buffer (not a plain float) so pretrain.py can update it every
        # step (beta warmup) without triggering torch.compile recompilation on value change.
        self.register_buffer("beta", torch.tensor(float(beta)))

    def update_target(self, tau: float):
        self.model.update_target(tau)

    def set_terminal_grad_scale(self, scale: float):
        self.model.set_terminal_grad_scale(scale)

    def set_beta(self, beta: float):
        self.beta.fill_(float(beta))

    def forward(self, batch: Dict[str, torch.Tensor], return_keys: Sequence[str] = ()) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs = self.model(batch)
        labels = batch["labels"]  # [B, L]
        B, H = outputs["values"].shape

        mask = (labels != IGNORE_LABEL_ID)  # [B, L]
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1)

        # Per-step, per-sequence CE on arrival states h_1..h_H (differentiable)
        logits = outputs["logits"]  # [B, H, L, V]
        labels_h = labels.unsqueeze(1).expand(-1, H, -1)
        ce_tok = self.loss_fn(logits, labels_h, ignore_index=IGNORE_LABEL_ID, valid_mask=mask.unsqueeze(1))  # [B, H, L]
        ce = ce_tok.to(torch.float32).sum(-1) / loss_divisor.unsqueeze(-1)  # [B, H]
        r = -ce  # r_t = -CE(h_{t+1}), t = 0..H-1

        values = outputs["values"]                  # V_theta(h_t),     t = 0..H-1
        target_values = outputs["target_values"]    # V_bar(h_{t+1}),   t = 0..H-1 (detached)
        terminal_value = outputs["terminal_value"]  # V_bar(h_H), grad -> h_H
        terminal_online_value = outputs["terminal_online_value"]  # V_theta(sg[h_H]), grad -> value_head params only
        segment_ends = outputs.get("segment_ends", [H])
        segment_starts = [0] + segment_ends[:-1]
        segmented = len(segment_ends) > 1

        # --- Critic: TD(lambda) targets, fully inside sg ---
        with torch.no_grad():
            r_d = r.detach()
            y = torch.empty_like(target_values)
            for start, end in zip(segment_starts, segment_ends):
                y[:, end - 1] = r_d[:, end - 1] + self.gamma * target_values[:, end - 1]
                for t in range(end - 2, start - 1, -1):
                    y[:, t] = r_d[:, t] + self.gamma * ((1 - self.lam) * target_values[:, t] + self.lam * y[:, t + 1])

        critic_loss = (0.5 * (values - y).square()).mean(-1).sum()  # mean over t, sum over batch

        # --- h_H boundary anchor: bootstrap-free regression target (r_{H-1}/(1-gamma)) ---
        # y[H-1] is always a pure TD(0) target (recursion base case, no lambda blending
        # regardless of self.lam) and h_H is never in critic_states, so V_theta never
        # trains directly on it. This anchors terminal_online_value = V_theta(sg[h_H])
        # against a target with zero V-dependence, giving this specific point immediate,
        # real-reward-grounded correction instead of relying solely on the slow TD(lambda)/
        # Polyak path.
        #
        # Penalized in reward units, (1-gamma)*V - r, rather than value units,
        # V - r/(1-gamma): both encode the same fixed point (V = r/(1-gamma)), but the
        # value-unit form squares a target of order 1/(1-gamma) (e.g. 40 at gamma=0.975),
        # inflating the loss ~1/(1-gamma)^2 (~1600x) beyond critic_loss's natural CE^2
        # scale and making eq_anchor_coef effectively far stronger than its nominal value.
        eq_anchor_loss = (0.5 * ((1 - self.gamma) * terminal_online_value - r_d[:, -1]).square()).mean()

        # --- Actor: J = sum gamma^t r_t + gamma^H V_bar(h_H) ---
        discounts = self.gamma ** torch.arange(H, dtype=torch.float32, device=r.device)
        if segmented:
            segment_terminal_values = outputs["segment_terminal_values"]
            J = torch.zeros((B, ), dtype=torch.float32, device=r.device)
            reward_return_abs = torch.zeros((), dtype=torch.float32, device=r.device)
            bootstrap_abs = torch.zeros((), dtype=torch.float32, device=r.device)
            for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
                seg_len = end - start
                local_discounts = self.gamma ** torch.arange(seg_len, dtype=torch.float32, device=r.device)
                start_discount = self.gamma ** start
                reward_return = (local_discounts * r[:, start:end]).sum(-1)
                bootstrap = (self.gamma ** seg_len) * segment_terminal_values[:, seg_idx]
                J = J + start_discount * (reward_return + bootstrap)
                with torch.no_grad():
                    reward_return_abs = reward_return_abs + reward_return.detach().abs().mean()
                    bootstrap_abs = bootstrap_abs + bootstrap.detach().abs().mean()
            reward_return_abs = reward_return_abs / len(segment_ends)
            bootstrap_abs = bootstrap_abs / len(segment_ends)
        else:
            J = (discounts * r).sum(-1) + (self.gamma ** H) * terminal_value  # [B]
            reward_return_abs = (discounts * r).sum(-1).detach().abs().mean()
            bootstrap_abs = ((self.gamma ** H) * terminal_value).detach().abs().mean()
        actor_loss = -J.sum()

        loss = self.critic_coef * critic_loss + self.beta * actor_loss + self.eq_anchor_coef * eq_anchor_loss

        # --- Metrics & diagnostics (no grad) ---
        with torch.no_grad():
            final_logits = logits[:, -1]  # [B, L, V]
            preds = torch.argmax(final_logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid = loss_counts > 0

            delta = r_d[:, 1:] + self.gamma * target_values[:, 1:] - target_values[:, :-1]  # t = 1..H-1
            dr = r_d[:, 1:] - r_d[:, :-1]
            corr = _pearson(delta.flatten(), dr.flatten())

            eq_gap = (terminal_value - r_d[:, -1] / (1 - self.gamma)).abs().mean()
            segment_ce_gains = []
            for start, end in zip(segment_starts, segment_ends):
                segment_ce_gains.append(ce[:, start] - ce[:, end - 1])
            segment_ce_gain = torch.stack(segment_ce_gains, dim=1).mean()
            bootstrap_to_reward = bootstrap_abs / reward_return_abs.clamp_min(1e-8)
            segment_terminal_mean = outputs["segment_terminal_values"].detach().mean()

            metrics = {
                "count": valid.sum(),
                "accuracy": torch.where(valid, is_correct.to(torch.float32).sum(-1) / loss_divisor, 0).sum(),
                "exact_accuracy": (valid & seq_is_correct).sum(),
                "critic_loss": critic_loss.detach(),
                "actor_loss": actor_loss.detach(),
                "eq_anchor_loss": eq_anchor_loss.detach() * valid.sum(),
                "beta": self.beta.detach() * valid.sum(),
                "terminal_grad_scale": outputs["terminal_grad_scale"] * valid.sum(),
                "ce_first": ce[:, 0].detach().sum(),
                "ce_last": ce[:, -1].detach().sum(),
                "value_mean": values.mean(-1).sum(),
                "td_abs": (values - y).abs().mean(-1).sum(),
                "return_J": J.detach().sum(),
                "corr_delta_dr": corr * valid.sum(),      # averaged by count downstream
                "eq_gap": eq_gap * valid.sum(),
                "u_h_ratio": outputs["u_h_ratio"] * valid.sum(),
                "bptt_segment": torch.tensor(segment_ends[0] - segment_starts[0], dtype=torch.float32, device=r.device) * valid.sum(),
                "num_segments": torch.tensor(len(segment_ends), dtype=torch.float32, device=r.device) * valid.sum(),
                "segment_ce_gain": segment_ce_gain * valid.sum(),
                "bootstrap_to_reward": bootstrap_to_reward * valid.sum(),
                "segment_terminal_mean": segment_terminal_mean * valid.sum(),
            }

        detached = {}
        if "preds" in return_keys:
            detached["preds"] = preds
        if "logits" in return_keys:
            detached["logits"] = final_logits.detach()
        if "q_halt_logits" in return_keys:
            # ARC evaluator compatibility: use terminal value as answer-ranking score.
            # This is not a TRM-style halting head in the RVSAC model.
            detached["q_halt_logits"] = terminal_value.detach()

        return loss, metrics, detached
