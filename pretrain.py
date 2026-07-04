"""RVSAC training loop.

Adapted from TRM's pretrain.py with the ACT carry machinery removed (RVSAC does
one full recursion rollout per forward) and a Polyak update of the target value
head added after each optimizer step (two-timescale: beta scales the actor term,
tau delays the target).
"""
from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

try:
    from adam_atan2 import AdamATan2
except ImportError:
    AdamATan2 = None

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from rvsac.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from rvsac.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Polyak rate for the target value head (theta_bar <- (1-tau) theta_bar + tau theta)
    target_tau: float = 0.005

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_only: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]

    step: int
    total_steps: int


def eager_model(model: nn.Module) -> nn.Module:
    """Return the uncompiled module when torch.compile wrapped it."""
    return getattr(model, "_orig_mod", model)


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        if rank == 0:
            load_checkpoint(model, config)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    adam_cls = AdamATan2 if AdamATan2 is not None else torch.optim.AdamW

    if config.arch.__pydantic_extra__.get("puzzle_emb_ndim", 0) == 0:  # type: ignore
        optimizers = [
            adam_cls(model.parameters(), lr=0, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=world_size)
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=world_size),
            adam_cls(model.parameters(), lr=0, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def save_named_train_state(config: PretrainConfig, train_state: TrainState, name: str):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, name))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        try:
            model.load_state_dict(state_dict, assign=True)
        except RuntimeError:
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            else:
                state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)
    return evaluators


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.cuda() for k, v in batch.items()}

    loss, metrics, _ = train_state.model(batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    # Polyak update of the target value head (the sg'd V_bar of eq. 2)
    train_state.model.update_target(config.target_tau)

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k].float() for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)
        return_keys = tuple(sorted(return_keys))

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds = {}
        metric_keys = []
        metric_values = None

        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}

            loss, metrics, preds = train_state.model(batch=batch, return_keys=return_keys)

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
            metric_values[set_id] += torch.stack([metrics[k].float() for k in metric_keys])

            del metrics, preds, batch, loss

        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}
        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))
        del save_preds

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, m in reduced_metrics.items():
                    count = max(m.pop("count"), 1)
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        for evaluator in evaluators:
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(config.checkpoint_path, f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}")
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            shutil.copy(code_file, os.path.join(config.checkpoint_path, os.path.basename(code_file)))

    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-RVSAC-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    torch.random.manual_seed(config.seed + RANK)

    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(),
                   settings=wandb.Settings(_disable_stats=True), mode=os.environ.get("WANDB_MODE", "offline"))
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(eager_model(train_state.model))

    if config.eval_only:
        if eval_loader is None:
            raise ValueError("eval_only=True requires eval data.")
        train_state_eval = copy.copy(train_state)
        train_state_eval.model = eager_model(train_state.model)
        train_state_eval.model.eval()
        metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators,
                           rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)
        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
            print(f"\nEVAL step {train_state.step}: {metrics}")
        if dist.is_initialized():
            dist.destroy_process_group()
        wandb.finish()
        return

    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                if train_state.step % 50 == 0 or train_state.step == 1:
                    shown = {k.removeprefix("train/"): round(float(v), 4) for k, v in metrics.items()
                             if k.split("/")[-1] in ("critic_loss", "ce_first", "ce_last", "accuracy", "exact_accuracy",
                                                     "value_mean", "td_abs", "corr_delta_dr", "u_h_ratio", "return_J")}
                    print(f"\nstep {train_state.step}: {shown}")
            if config.ema:
                ema_helper.update(eager_model(train_state.model))

        if eval_loader is not None and _iter_id >= config.min_eval_interval:
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state)

            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.copy(train_state)
                train_state_eval.model = ema_helper.ema_copy(eager_model(train_state.model))
            else:
                train_state_eval = copy.copy(train_state)
                train_state_eval.model = eager_model(train_state.model)

            train_state_eval.model.eval()
            metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators,
                               rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                print(f"\nEVAL step {train_state.step}: {metrics}")

            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                if config.ema:
                    save_named_train_state(config, train_state_eval, f"step_{train_state.step}_ema")
                elif train_state_eval is not train_state:
                    save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
