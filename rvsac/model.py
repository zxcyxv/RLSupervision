"""RVSAC: Recursive Value-State Actor-Critic.

Physical form follows TRM (Transformer blocks, RoPE, puzzle embeddings), but the
recursion is a pure residual RNN:

    h_{t+1} = h_t + f_theta(h_t)          (transition head = deterministic policy)
    v_t     = V_theta(h_t)                (scalar value head)
    r_t     = -CE(y, C_theta(h_{t+1}))    (white-box reward, on arrival state)

Gradient routing (eq. 2 of the analysis doc) lives in the loss head; this module
is responsible for the rollout and for exposing the three differentiation paths:

  - `values`:         V_theta(sg[h_t]) (or V_theta(h_t) if critic_shapes_trunk)
  - `target_values`:  V_bar(sg[h_t])   fully detached, for TD(lambda) targets
  - `terminal_value`: V_bar(h_H)       frozen params, gradient flows into h_H
"""
from typing import Dict, List, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from rvsac.common import trunc_normal_init_
from rvsac.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from rvsac.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


class RVSACConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Rollout
    horizon: int  # H: number of recursion steps per forward

    # Transformer config (TRM defaults)
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False  # use MLP over sequence dim instead of attention (TRM option)
    puzzle_emb_len: int = 16

    # If True, critic regression also shapes the trunk (MuZero-style); if False,
    # the critic sees sg[h_t] and only the value head learns from L_V.
    critic_shapes_trunk: bool = False

    # 0 or >= horizon: full BPTT. Otherwise detach h at segment boundaries and
    # let the loss use K-step bootstrapped actor/critic targets.
    bptt_segment: int = 0


class RVSACBlock(nn.Module):
    """TRM post-norm block: Attention (or seq-MLP) + SwiGLU, RMS-normed residuals."""

    def __init__(self, config: RVSACConfig) -> None:
        super().__init__()
        self.config = config
        if config.mlp_t:
            puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
            self.mlp_t = SwiGLU(hidden_size=config.seq_len + puzzle_emb_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class TransitionHead(nn.Module):
    """f_theta: the deterministic policy. u = W_out(Blocks(rms_norm(h))).

    Input rms_norm keeps the block stack well-conditioned while the outer
    recursion h <- h + u stays purely residual. W_out is zero-initialized
    (ReZero-style) so the rollout starts at the identity flow; gradients still
    reach W_out through the costate.
    """

    def __init__(self, config: RVSACConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([RVSACBlock(config) for _ in range(config.L_layers)])
        self.out_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.norm_eps = config.rms_norm_eps
        with torch.no_grad():
            self.out_proj.weight.zero_()

    def forward(self, h: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        x = rms_norm(h, variance_epsilon=self.norm_eps)
        for layer in self.layers:
            x = layer(cos_sin=cos_sin, hidden_states=x)
        return self.out_proj(x)


class ValueHead(nn.Module):
    """V_theta: scalar value from mean-pooled grid-token positions.

    Reward r_t=-CE_t is itself a mean over the grid-token positions, so the value
    readout uses the same aggregation instead of the puzzle-embedding prefix position
    (which critic_shapes_trunk=False gives no direct gradient to shape into a summary).
    """

    def __init__(self, config: RVSACConfig, puzzle_emb_len: int) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.puzzle_emb_len = puzzle_emb_len
        self.fc1 = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        self.fc2 = CastedLinear(config.hidden_size, 1, bias=True)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = h[:, self.puzzle_emb_len:].mean(dim=1)
        x = rms_norm(pooled, variance_epsilon=self.norm_eps)
        return self.fc2(F.silu(self.fc1(x))).to(torch.float32).squeeze(-1)


class RVSAC(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        config_dict.pop("causal", None)  # pretrain passes it; always non-causal
        self.config = RVSACConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O (TRM defaults)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
        else:
            self.puzzle_emb_len = 0

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Trunk + heads
        self.transition = TransitionHead(self.config)
        self.value_head = ValueHead(self.config, self.puzzle_emb_len)

        # EMA (Polyak) target value head: V_bar. Frozen; updated via update_target().
        self.value_head_target = ValueHead(self.config, self.puzzle_emb_len)
        self.value_head_target.load_state_dict(self.value_head.state_dict())
        self.value_head_target.requires_grad_(False)

        # Scales *only* the actor's terminal-bootstrap gradient (V_bar(h_H) -> h_H),
        # i.e. the exploit channel, independent of the honest reward-sum term. 1.0 =
        # full gradient (original behavior). Ramped 0 -> 1 via set_terminal_grad_scale()
        # from pretrain.py to let the critic calibrate before the actor can lean on it.
        self.register_buffer("terminal_grad_scale", torch.tensor(1.0))

    def set_terminal_grad_scale(self, scale: float):
        self.terminal_grad_scale.fill_(float(scale))

    @torch.no_grad()
    def update_target(self, tau: float):
        """theta_bar <- (1 - tau) * theta_bar + tau * theta"""
        for p_bar, p in zip(self.value_head_target.parameters(), self.value_head.parameters()):
            p_bar.lerp_(p.to(p_bar.dtype), tau)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        H = self.config.horizon
        K = self.config.bptt_segment
        segmented = K > 0 and K < H
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # h_0 = encoder(input); no re-injection afterwards (pure residual RNN)
        h = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        states: List[torch.Tensor] = [h]
        u_h_ratio = h.new_zeros(())
        for _t in range(H):
            if segmented and _t > 0 and (_t % K == 0):
                h = h.detach()
            u = self.transition(h, cos_sin)
            h = h + u
            states.append(h)
            with torch.no_grad():
                u_h_ratio = u_h_ratio + torch.linalg.vector_norm(u.float()) / torch.linalg.vector_norm(h.float()).clamp_min(1e-8)

        # C_theta on arrival states h_1..h_H (differentiable: actor path [A-fixed-sign])
        logits = torch.stack([self.lm_head(rms_norm(states[t], variance_epsilon=self.config.rms_norm_eps))[:, self.puzzle_emb_len:]
                              for t in range(1, H + 1)], dim=1)  # [B, H, L, V]

        # Critic inputs: V_theta over h_0..h_{H-1}; sg[h_t] unless critic shapes trunk
        critic_states = states[:-1] if self.config.critic_shapes_trunk else [s.detach() for s in states[:-1]]
        values = torch.stack([self.value_head(s) for s in critic_states], dim=1)  # [B, H] fp32

        # V_theta(sg[h_H]): online head trained directly on the boundary state, which
        # critic_states above never includes. Anchored in the loss head against a
        # bootstrap-free target (r_{H-1}/(1-gamma)) to give this point real-reward
        # grounding instead of pure TD(0) extrapolation via value_head_target.
        terminal_online_value = self.value_head(states[H].detach())  # [B] fp32

        with torch.no_grad():
            # V_bar over h_1..h_H for TD(lambda) targets (fully detached)
            target_values = torch.stack([self.value_head_target(states[t]) for t in range(1, H + 1)], dim=1)  # [B, H]

        # V_bar(h_H) for the actor's terminal bootstrap: frozen params, live input, but
        # the gradient into h_H is scaled by terminal_grad_scale (alpha*x + (1-alpha)*sg[x]
        # is numerically identical to x for the forward value; only d/dh_H is scaled).
        alpha = self.terminal_grad_scale
        h_H_for_actor = alpha * states[H] + (1 - alpha) * states[H].detach()
        terminal_value = self.value_head_target(h_H_for_actor)  # [B] fp32, grad -> h_H (scaled)

        segment_ends = list(range(K, H, K)) + [H] if segmented else [H]
        segment_states_for_actor = [alpha * states[t] + (1 - alpha) * states[t].detach() for t in segment_ends]
        segment_terminal_values = torch.stack([self.value_head_target(s) for s in segment_states_for_actor], dim=1)

        return {
            "logits": logits,
            "values": values,
            "terminal_online_value": terminal_online_value,
            "target_values": target_values,
            "terminal_value": terminal_value,
            "segment_terminal_values": segment_terminal_values,
            "segment_ends": segment_ends,
            "u_h_ratio": (u_h_ratio / H).detach(),
            "terminal_grad_scale": self.terminal_grad_scale.detach(),
        }
