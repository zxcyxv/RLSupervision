"""Evaluate an RVSAC checkpoint on a small fixed number of test batches.

This is a lightweight alternative to `pretrain.py eval_only=True` for quick
checkpoint probes. It loads the saved run config from `all_config.yaml` when it
is available next to the checkpoint, then lets CLI flags override eval-only
settings such as horizon and batch count.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import yaml

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class


def _default_config() -> Dict[str, Any]:
    return {
        "arch": {
            "name": "rvsac.model@RVSAC",
            "loss": {
                "name": "rvsac.losses@RVSACLossHead",
                "loss_type": "stablemax_cross_entropy",
                "gamma": 0.95,
                "lam": 0.9,
                "beta": 0.1,
                "critic_coef": 1.0,
            },
            "horizon": 16,
            "bptt_segment": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4,
            "puzzle_emb_ndim": 512,
            "pos_encodings": "rope",
            "forward_dtype": "bfloat16",
            "mlp_t": False,
            "puzzle_emb_len": 16,
            "critic_shapes_trunk": False,
        },
        "data_paths": ["data/sudoku-extreme-1k-aug-1000"],
        "seed": 0,
    }


def _load_run_config(checkpoint: Path, config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        candidate = checkpoint.parent / "all_config.yaml"
        config_path = candidate if candidate.exists() else None

    if config_path is None:
        return _default_config()

    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} did not contain a mapping.")
    return config


def _resolve_checkpoint(path: str, use_ema: bool) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    pattern = re.compile(r"^step_(\d+)(_ema)?$")
    candidates = []
    for child in checkpoint_path.iterdir():
        match = pattern.match(child.name)
        if match is None or not child.is_file():
            continue
        is_ema = match.group(2) is not None
        if is_ema == use_ema:
            candidates.append((int(match.group(1)), child))

    if not candidates:
        suffix = "_ema" if use_ema else ""
        raise FileNotFoundError(f"No step_N{suffix} checkpoint found in {checkpoint_path}")
    return max(candidates, key=lambda item: item[0])[1]


def _load_state_dict(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state_dict, assign=True)
        return
    except RuntimeError:
        pass

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    else:
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True)


def _create_loader(data_paths: Iterable[str], batch_size: int, seed: int) -> tuple[DataLoader, Any]:
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=seed,
            dataset_paths=list(data_paths),
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata


def _create_model(
    config: Dict[str, Any],
    metadata: Any,
    batch_size: int,
    horizon: Optional[int],
    bptt_segment: Optional[int],
    gamma: Optional[float],
    device: torch.device,
) -> nn.Module:
    arch = dict(config["arch"])
    loss_cfg = dict(arch.pop("loss"))
    model_name = arch.pop("name")
    loss_name = loss_cfg.pop("name")

    if horizon is not None:
        arch["horizon"] = horizon
    if bptt_segment is not None:
        arch["bptt_segment"] = bptt_segment
    if gamma is not None:
        loss_cfg["gamma"] = gamma

    if arch.get("puzzle_emb_ndim") == "${.hidden_size}":
        arch["puzzle_emb_ndim"] = arch["hidden_size"]

    model_cfg = {
        **arch,
        "batch_size": batch_size,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
    }

    model_cls = load_model_class(model_name)
    loss_head_cls = load_model_class(loss_name)
    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_cfg)
    return model


def _slice_batch(batch: Dict[str, torch.Tensor], remaining: Optional[int]) -> Dict[str, torch.Tensor]:
    if remaining is None:
        return batch
    take = max(0, min(remaining, next(iter(batch.values())).shape[0]))
    return {k: v[:take] for k, v in batch.items()}


def evaluate(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    eval_batches = args.eval_batches if args.eval_batches > 0 else None
    checkpoint = _resolve_checkpoint(args.checkpoint, args.ema)
    config = _load_run_config(checkpoint, Path(args.config) if args.config else None)

    data_paths = args.data_path if args.data_path else config.get("data_paths")
    if not data_paths:
        raise ValueError("No data paths provided. Use --data-path or keep all_config.yaml next to the checkpoint.")

    device = torch.device(args.device)
    loader, metadata = _create_loader(data_paths, args.batch_size, config.get("seed", 0))
    model = _create_model(
        config=config,
        metadata=metadata,
        batch_size=args.batch_size,
        horizon=args.horizon,
        bptt_segment=args.bptt_segment,
        gamma=args.gamma,
        device=device,
    )
    _load_state_dict(model, checkpoint, device)
    if args.compile:
        model = torch.compile(model)  # type: ignore
    model.eval()

    metric_sums: Dict[str, Dict[str, float]] = {}
    consumed_samples = 0
    max_samples = args.max_samples

    print(f"checkpoint: {checkpoint}")
    print(f"data_paths: {list(data_paths)}")
    print(f"batch_size: {args.batch_size}")
    print(f"eval_batches: {eval_batches}")
    print(f"max_samples: {max_samples}")
    print(f"horizon: {args.horizon if args.horizon is not None else config['arch'].get('horizon')}")
    print(f"bptt_segment: {args.bptt_segment if args.bptt_segment is not None else config['arch'].get('bptt_segment', 0)}")
    print(f"gamma: {args.gamma if args.gamma is not None else config['arch']['loss'].get('gamma')}")

    with torch.inference_mode():
        iterator = tqdm.tqdm(
            enumerate(loader),
            total=eval_batches,
            desc="eval",
        )
        for batch_idx, (set_name, batch, _global_batch_size) in iterator:
            if eval_batches is not None and batch_idx >= eval_batches:
                break
            if max_samples is not None and consumed_samples >= max_samples:
                break

            remaining = None if max_samples is None else max_samples - consumed_samples
            batch = _slice_batch(batch, remaining)
            batch_size = next(iter(batch.values())).shape[0]
            if batch_size == 0:
                break
            consumed_samples += batch_size

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            _loss, metrics, _preds = model(batch=batch, return_keys=())

            set_metrics = metric_sums.setdefault(set_name, {})
            for key, value in metrics.items():
                set_metrics[key] = set_metrics.get(key, 0.0) + float(value.detach().cpu())

    results: Dict[str, Dict[str, float]] = {}
    for set_name, sums in metric_sums.items():
        count = max(sums.pop("count", 0.0), 1.0)
        results[set_name] = {key: value / count for key, value in sorted(sums.items())}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RVSAC checkpoint on a limited number of test batches.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file, or checkpoint directory containing step_N files.")
    parser.add_argument("--config", default=None, help="Optional all_config.yaml path. Defaults to CHECKPOINT_DIR/all_config.yaml.")
    parser.add_argument("--data-path", action="append", default=None, help="Dataset path. Can be passed multiple times.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=10, help="Number of test batches to evaluate. Default 10 means 10*batch_size examples. Use 0 for all.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional exact cap on evaluated examples.")
    parser.add_argument("--horizon", type=int, default=None, help="Override recursion horizon H at eval time.")
    parser.add_argument("--bptt-segment", type=int, default=None, help="Override bptt_segment for diagnostic metrics.")
    parser.add_argument("--gamma", type=float, default=None, help="Override loss gamma for eval metrics.")
    parser.add_argument("--ema", action="store_true", help="When --checkpoint is a directory, load the latest step_N_ema instead of step_N.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for eval.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = evaluate(args)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
