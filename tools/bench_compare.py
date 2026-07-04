"""Compare training-step cost (time / peak memory / params) of RVSAC vs TRM
on the Sudoku shape (seq_len 81, vocab 11).

Both models run uncompiled, bf16 forward, one full optimizer-step worth of
forward+backward. TRM's unit of work is one ACT step (its train_batch does one
inner forward per optimizer step); RVSAC's is one full H-step rollout with BPTT.

Run: python tools/bench_compare.py
"""
import os
import sys
import time
import gc

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEQ_LEN = 81
VOCAB = 11
N_PUZZLES = 1001  # 1000 puzzles + blank
WARMUP, ITERS = 3, 10


def make_batch(B, device="cuda"):
    torch.manual_seed(0)
    return {
        "inputs": torch.randint(1, VOCAB, (B, SEQ_LEN), device=device),
        "labels": torch.randint(1, VOCAB, (B, SEQ_LEN), device=device),
        "puzzle_identifiers": torch.randint(0, N_PUZZLES, (B,), device=device),
    }


def bench(step_fn, B):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        step_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step_fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak_gb


def bench_rvsac(B, horizon=16):
    from rvsac.model import RVSAC
    from rvsac.losses import RVSACLossHead

    cfg = dict(batch_size=B, seq_len=SEQ_LEN, vocab_size=VOCAB, num_puzzle_identifiers=N_PUZZLES,
               puzzle_emb_ndim=512, puzzle_emb_len=16, horizon=horizon, L_layers=2,
               hidden_size=512, expansion=4, num_heads=8, pos_encodings="rope",
               forward_dtype="bfloat16")
    with torch.device("cuda"):
        model = RVSAC(cfg)
        head = RVSACLossHead(model, loss_type="stablemax_cross_entropy", gamma=0.95, lam=0.9, beta=0.1)
    head.train()
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-4)
    batch = make_batch(B)
    n_params = sum(p.numel() for p in model.parameters())

    def step():
        loss, _, _ = head(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        head.update_target(0.005)

    ms, mem = bench(step, B)
    del model, head, opt
    gc.collect(); torch.cuda.empty_cache()
    return n_params, ms, mem


def bench_trm(B, mlp_t=False):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref", "TinyRecursiveModels"))
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    from models.losses import ACTLossHead

    cfg = dict(batch_size=B, seq_len=SEQ_LEN, vocab_size=VOCAB, num_puzzle_identifiers=N_PUZZLES,
               puzzle_emb_ndim=512, puzzle_emb_len=16, H_cycles=3, L_cycles=6, H_layers=0, L_layers=2,
               hidden_size=512, expansion=4, num_heads=8,
               pos_encodings="none" if mlp_t else "rope",
               halt_max_steps=16, halt_exploration_prob=0.1, forward_dtype="bfloat16",
               mlp_t=mlp_t, no_ACT_continue=True)
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    head.train()
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-4)
    batch = make_batch(B)
    n_params = sum(p.numel() for p in model.parameters())

    with torch.device("cuda"):
        carry_holder = [head.initial_carry(batch)]

    def step():
        carry_holder[0], loss, _, _, _ = head(return_keys=[], carry=carry_holder[0], batch=batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    ms, mem = bench(step, B)
    del model, head, opt, carry_holder
    gc.collect(); torch.cuda.empty_cache()
    return n_params, ms, mem


if __name__ == "__main__":
    rows = []
    for B in (128, 256, 384, 768):
        for name, fn in (("TRM(attn)", lambda B=B: bench_trm(B, mlp_t=False)),
                         ("TRM(mlp_t)", lambda B=B: bench_trm(B, mlp_t=True)),
                         ("RVSAC(H=16)", lambda B=B: bench_rvsac(B, horizon=16)),
                         ("RVSAC(H=8)", lambda B=B: bench_rvsac(B, horizon=8))):
            try:
                n, ms, mem = fn()
                rows.append((name, B, n, ms, mem))
                print(f"{name:14s} B={B:4d}  params={n/1e6:6.2f}M  {ms:8.1f} ms/step  peak {mem:5.2f} GB", flush=True)
            except torch.OutOfMemoryError:
                rows.append((name, B, None, None, None))
                print(f"{name:14s} B={B:4d}  OOM", flush=True)
                gc.collect(); torch.cuda.empty_cache()
