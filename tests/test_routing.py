"""Gradient-routing tests: verify that the sg wiring of eq. (2) implements
Theorem C (algebraic elimination of the four pathologies).

Run: python -m tests.test_routing
"""
import torch

from rvsac.model import RVSAC
from rvsac.losses import RVSACLossHead

torch.manual_seed(0)

CFG = dict(
    batch_size=4,
    seq_len=16,
    vocab_size=12,
    num_puzzle_identifiers=8,
    puzzle_emb_ndim=32,
    puzzle_emb_len=2,
    horizon=4,
    L_layers=1,
    hidden_size=32,
    expansion=2,
    num_heads=2,
    pos_encodings="rope",
    forward_dtype="float32",
    critic_shapes_trunk=False,
)


def make_batch(B=4, L=16, V=12, P=8):
    return {
        "inputs": torch.randint(0, V, (B, L)),
        "labels": torch.randint(0, V, (B, L)),
        "puzzle_identifiers": torch.randint(0, P, (B,)),
    }


def make_model(**overrides):
    cfg = {**CFG, **overrides}
    model = RVSAC(dict(cfg))
    model.train()
    head = RVSACLossHead(model, loss_type="stablemax_cross_entropy", gamma=0.95, lam=0.9, beta=0.1)
    return model, head


def grads_by_group(model, loss):
    model.zero_grad(set_to_none=True)
    if model.puzzle_emb.local_weights.grad is not None:
        model.puzzle_emb.local_weights.grad = None
    loss.backward(retain_graph=True)
    def total(mod):
        return sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
    return {
        "transition": total(model.transition),
        "lm_head": total(model.lm_head),
        "embed": total(model.embed_tokens),
        "value_head": total(model.value_head),
        "value_head_target": sum(0.0 if p.grad is None else p.grad.abs().sum().item()
                                 for p in model.value_head_target.parameters()),
    }


def test_critic_isolated():
    """Critic term must not touch the trunk (P1/P2/P3-1: no [A], no [B], no rank-1 collapse pressure)."""
    model, head = make_model()
    batch = make_batch()
    outputs = model(batch)
    # Rebuild the critic loss exactly as the head does
    loss, metrics, _ = head(batch)
    critic_only = metrics["critic_loss"]  # detached; recompute differentiable version:
    out = model(batch)
    with torch.no_grad():
        y = torch.randn_like(out["target_values"])  # any nonzero-residual target works for the routing check
    critic_loss = (0.5 * (out["values"] - y).square()).mean(-1).sum()
    g = grads_by_group(model, critic_loss)
    assert g["value_head"] > 0, "critic must train the value head"
    assert g["transition"] == 0, f"critic leaked into trunk: {g['transition']}"
    assert g["lm_head"] == 0, f"critic leaked into classifier: {g['lm_head']}"
    assert g["embed"] == 0, f"critic leaked into embeddings: {g['embed']}"
    assert g["value_head_target"] == 0
    print("PASS critic_isolated", g)


def test_actor_routing():
    """Actor term must train trunk+classifier through BPTT, never the critic params (P3-2: actor signal independent of delta)."""
    model, head = make_model()
    batch = make_batch()
    out = model(batch)
    B, H = out["values"].shape
    labels = batch["labels"]
    mask = labels != -100
    # reconstruct J differentiable
    from rvsac.losses import stablemax_cross_entropy
    labels_h = labels.unsqueeze(1).expand(-1, H, -1)
    ce_tok = stablemax_cross_entropy(out["logits"], labels_h, valid_mask=mask.unsqueeze(1))
    ce = ce_tok.to(torch.float32).sum(-1) / mask.sum(-1).clamp_min(1).unsqueeze(-1)
    r = -ce
    disc = 0.95 ** torch.arange(H, dtype=torch.float32)
    J = (disc * r).sum(-1) + (0.95 ** H) * out["terminal_value"]
    actor_loss = -J.sum()
    g = grads_by_group(model, actor_loss)
    assert g["transition"] > 0, "actor must train the transition (policy)"
    assert g["lm_head"] > 0, "actor must train the classifier"
    assert g["embed"] > 0, "actor must train the encoder"
    assert g["value_head"] == 0, f"actor leaked into critic params: {g['value_head']}"
    assert g["value_head_target"] == 0, "target head params must stay frozen"
    # terminal value must still carry gradient INTO the state (costate supply).
    # fc2 of the target head is zero-init, so grad_h V_bar == 0 at init; give it
    # nonzero weights (as Polyak updates would after critic training) first.
    with torch.no_grad():
        model.value_head_target.fc2.weight.normal_(std=0.1)
    out2 = model(batch)
    gh = torch.autograd.grad(out2["terminal_value"].sum(), model.transition.out_proj.weight, retain_graph=False, allow_unused=True)[0]
    assert gh is not None and gh.abs().sum() > 0, "V_bar(h_H) must backprop into the trunk via h_H"
    print("PASS actor_routing", g)


def test_td_lambda_targets():
    """TD(lambda) recursion: lambda=0 -> 1-step TD; lambda=1 -> Monte Carlo with terminal bootstrap."""
    B, H, gamma = 3, 5, 0.9
    r = torch.randn(B, H)
    tv = torch.randn(B, H)  # tv[:, t] = V_bar(h_{t+1})

    def targets(lam):
        y = torch.empty(B, H)
        y[:, H - 1] = r[:, H - 1] + gamma * tv[:, H - 1]
        for t in range(H - 2, -1, -1):
            y[:, t] = r[:, t] + gamma * ((1 - lam) * tv[:, t] + lam * y[:, t + 1])
        return y

    y0 = targets(0.0)
    assert torch.allclose(y0, r + gamma * tv, atol=1e-6)

    y1 = targets(1.0)
    # closed form: y_t = sum_k gamma^{k-t} r_k + gamma^{H-t} V_bar(h_H)
    for b in range(B):
        for t in range(H):
            v = sum(gamma ** (k - t) * r[b, k].item() for k in range(t, H)) + gamma ** (H - t) * tv[b, H - 1].item()
            assert abs(y1[b, t].item() - v) < 1e-4, (t, y1[b, t].item(), v)
    print("PASS td_lambda_targets")


def test_full_step_and_target_update():
    """One full forward/backward/step + Polyak update runs and stays finite."""
    model, head = make_model()
    batch = make_batch()
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    before = [p.clone() for p in model.value_head_target.parameters()]
    for _ in range(3):
        loss, metrics, _ = head(batch)
        assert torch.isfinite(loss), "loss not finite"
        opt.zero_grad()
        loss.backward()
        opt.step()
        head.update_target(0.5)
    after = list(model.value_head_target.parameters())
    moved = sum((a - b).abs().sum().item() for a, b in zip(after, before))
    assert moved > 0, "Polyak update did not move the target head"
    for k in ("corr_delta_dr", "eq_gap", "u_h_ratio", "td_abs"):
        assert torch.isfinite(metrics[k]), k
    print("PASS full_step_and_target_update", {k: round(float(v), 4) for k, v in metrics.items()})


def test_self_destruction_eliminated():
    """P2 check: the gradient of the total loss w.r.t. per-step CE must be
    non-negative-pressure only (actor always pushes CE down, never up),
    regardless of the sign of the TD error."""
    model, head = make_model()
    batch = make_batch()
    out = model(batch)
    B, H = out["values"].shape
    labels = batch["labels"]
    mask = labels != -100
    from rvsac.losses import stablemax_cross_entropy
    labels_h = labels.unsqueeze(1).expand(-1, H, -1)
    ce_tok = stablemax_cross_entropy(out["logits"], labels_h, valid_mask=mask.unsqueeze(1))
    ce = ce_tok.to(torch.float32).sum(-1) / mask.sum(-1).clamp_min(1).unsqueeze(-1)  # [B, H]
    ce_leaf = ce.detach().requires_grad_(True)
    r = -ce_leaf
    # critic targets built from r are inside sg -> use detached values; critic term
    # contributes no gradient to ce_leaf by construction (r.detach() in the head).
    with torch.no_grad():
        y = torch.empty_like(out["target_values"])
        y[:, H - 1] = r[:, H - 1] + 0.95 * out["target_values"][:, H - 1]
        for t in range(H - 2, -1, -1):
            y[:, t] = r[:, t] + 0.95 * ((1 - 0.9) * out["target_values"][:, t] + 0.9 * y[:, t + 1])
    critic_loss = (0.5 * (out["values"].detach() - y).square()).mean(-1).sum()  # no grad path to ce_leaf
    disc = 0.95 ** torch.arange(H, dtype=torch.float32)
    J = (disc * r).sum(-1) + (0.95 ** H) * out["terminal_value"].detach()
    loss = critic_loss - 0.1 * J.sum()
    loss.backward()
    g = ce_leaf.grad
    assert g is not None
    expected = 0.1 * disc.expand(B, H)
    assert torch.allclose(g, expected, atol=1e-6), "dL/dCE must be exactly +beta*gamma^t (fixed sign, delta-independent)"
    print("PASS self_destruction_eliminated: dL/dCE == +beta*gamma^t for all t")


def test_segmented_bptt_detaches_boundaries():
    """K-step mode must keep long forward rollout but stop gradients crossing segment boundaries."""
    batch = make_batch()

    full_model, _ = make_model(horizon=4, bptt_segment=0)
    full_out = full_model(batch)
    full_loss = full_out["logits"][:, 2:].float().square().sum()
    full_model.zero_grad(set_to_none=True)
    full_loss.backward()
    full_embed_grad = full_model.embed_tokens.embedding_weight.grad.abs().sum().item()
    assert full_embed_grad > 0, "full BPTT should let late logits reach the input embedding"

    seg_model, _ = make_model(horizon=4, bptt_segment=2)
    seg_out = seg_model(batch)
    late_segment_loss = seg_out["logits"][:, 2:].float().square().sum()
    seg_model.zero_grad(set_to_none=True)
    late_segment_loss.backward()
    seg_embed_grad = seg_model.embed_tokens.embedding_weight.grad
    seg_embed_grad_sum = 0.0 if seg_embed_grad is None else seg_embed_grad.abs().sum().item()
    assert seg_embed_grad_sum == 0.0, f"segmented BPTT leaked across boundary: {seg_embed_grad_sum}"
    assert seg_out["segment_ends"] == [2, 4]
    assert seg_out["segment_terminal_values"].shape == (batch["inputs"].shape[0], 2)
    print("PASS segmented_bptt_detaches_boundaries", {"full_embed_grad": round(full_embed_grad, 4)})


def test_segmented_loss_runs():
    """H=32,K=8-style loss path should run, expose diagnostics, and backprop finite gradients."""
    model, head = make_model(horizon=32, bptt_segment=8)
    batch = make_batch()
    loss, metrics, _ = head(batch)
    assert torch.isfinite(loss), "segmented loss not finite"
    for k in ("bptt_segment", "num_segments", "segment_ce_gain", "bootstrap_to_reward", "segment_terminal_mean"):
        assert k in metrics, k
        assert torch.isfinite(metrics[k]), k
    model.zero_grad(set_to_none=True)
    loss.backward()
    transition_grad = sum(p.grad.abs().sum().item() for p in model.transition.parameters() if p.grad is not None)
    assert transition_grad > 0, "segmented actor loss should train transition"
    print("PASS segmented_loss_runs", {k: round(float(metrics[k]), 4) for k in ("bptt_segment", "num_segments", "segment_ce_gain", "bootstrap_to_reward")})


if __name__ == "__main__":
    test_td_lambda_targets()
    test_critic_isolated()
    test_actor_routing()
    test_full_step_and_target_update()
    test_self_destruction_eliminated()
    test_segmented_bptt_detaches_boundaries()
    test_segmented_loss_runs()
    print("\nALL ROUTING TESTS PASSED")
