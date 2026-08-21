"""
Train Prosit Intensity (PTM features) on local parquet files using PyTorch.

Usage:
  Edit `CONFIG` below, then run:
    python run_scripts/train_prosit_intensity_ptms_torch.py

Notes:
  - This script forces the DLOmix backend to PyTorch by setting DLOMIX_BACKEND=torch
    before importing dlomix.
  - Uses a streaming parquet reader to avoid HuggingFace `datasets` cache materialization.
  - Your parquet needs at least:
      sequence column (default: modified_sequence, ProForma)
      label column    (default: intensities_raw)
      model features  (default: collision_energy_aligned_normed, precursor_charge_onehot)
"""

from __future__ import annotations

import os
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

# -----------------------------------------------------------------------------
# Hugging Face cache bootstrap (must run BEFORE importing `datasets` / `dlomix`)
#
# Default behavior of `datasets` is to cache under `~/.cache/huggingface/...`.
# For large parquet inputs this can quickly consume the user home partition.
# We instead default to a repo-local cache under `<repo>/.hf/`, unless the user
# already set HF_* env vars externally.
#
# To use a different cache location, edit `HF_CACHE_ROOT` below.
# -----------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
HF_CACHE_ROOT = (repo_root / ".hf").expanduser()
DATA_ROOT = (repo_root / "data").expanduser()
# or set HF_CACHE_ROOT = Path("/path/to/your/cache").expanduser() for a custom location
# If ran within container it will refer to the container root

hf_root = Path(os.environ.get("HF_HOME", str(HF_CACHE_ROOT))).expanduser()
hf_datasets_cache = Path(
    os.environ.get("HF_DATASETS_CACHE", str(hf_root / "datasets"))
).expanduser()
hf_hub_cache = Path(os.environ.get("HF_HUB_CACHE", str(hf_root / "hub"))
).expanduser()

hf_datasets_cache.mkdir(parents=True, exist_ok=True)
hf_hub_cache.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(hf_root))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))

data_root = Path(os.environ.get("DATA_HOME", str(DATA_ROOT))).expanduser()
data_location = Path(os.environ.get("DATA_LOCATION", str(data_root / "data"))).expanduser()

train_data = Path(os.environ.get("TRAIN_LOCATION", str(data_location / "all_train_ptms_fixed_na.parquet"))).expanduser()
val_data = Path(os.environ.get("VAL_LOCATION", str(data_location / "all_val_ptms_fixed_na.parquet"))).expanduser()
test_data = Path(os.environ.get("TEST_LOCATION", str(data_location / "test.parquet"))).expanduser()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


os.environ.setdefault("DLOMIX_BACKEND", "torch")

import torch
from tqdm.auto import tqdm
import wandb
import numpy as np

from dlomix.data import StreamingFragmentIonIntensityDataset
from dlomix.losses.intensity_torch import masked_spectral_distance, gaussian_nll
from dlomix.models import PrositIntensityPredictor, PrositIntensityUncertaintyPredictor

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# (1) PROSIT: https://www.nature.com/articles/s41592-019-0426-7
#     "Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning"
# (2) PROSIT-PTM: https://www.biorxiv.org/content/10.1101/2025.11.07.687302v1
#     "Learning the Unseen: Data-Augmented Deep Learning for PTM Discovery with Prosit-PTM"
CONFIG = {
    # --- Model Settings ---
    # Path within container, remember to define the path names when creating the image
    "train": str(train_data),
    "val": str(val_data),
    "test": str(test_data),
    # Bool for model selection, false will use the current Prosit standard of masked spectral distance
    "uncertainty_aware": _env_bool("UNCERTAINTY_AWARE", True),
    # --- Training loop (evidence: `run_scripts/run_prosit_intensity_torch.py`,
    # `run_scripts/run_prosit_intensity_ptms_torch.py`, and TF scripts) ---
    "epochs": int(os.environ.get("N_EPOCHS", 120)),  # according to (2) PROSIT-PTM (FII: max 120 epochs with early stopping)
    # "epochs": 2,  # useful for debugging
    # "epochs": 20,  # evidence: `run_scripts/run_prosit_intensity_torch.py`, `run_scripts/run_prosit_intensity_ptms_torch.py`
    # "epochs": 32,  # according to (1) PROSIT (paper reports 32 epochs)
    "batch_size": int(os.environ.get("BATCH_SIZE", 2048)),  # reasonable laptop default; (2) PROSIT-PTM excerpt doesn't specify FII batch size
    # "batch_size": 8,  # evidence: PTM torch example uses 8
    # "batch_size": 128,  # evidence: non-PTM torch example + TF PTM script use 128
    # "batch_size": 512,  # according to (1) PROSIT (paper reports batch size 512)
    "lr": float(os.environ.get("LEARNING_RATE", 2e-4)),  # according to (2) PROSIT-PTM (upper CLR bound; used when `use_clr=False`)
    # "lr": 1e-4,  # evidence: all Prosit example scripts use Adam(lr=1e-4)
    # "lr": 1e-3,  # according to (1) PROSIT (paper reports initial lr=0.001)
    "max_seq_len": 32,  # evidence: PTM examples use 32 (non-PTM intensity uses 30)
    # "max_seq_len": 30,  # evidence: `run_scripts/run_prosit_intensity_torch.py` (and RT scripts)
    # --- Dataloader settings ---
    "shuffle": True,
    "shuffle_buffer_size": 10_000,
    "parquet_read_batch_size": 50_000,
    "num_workers": int(os.environ.get("NUM_WORKERS", 4)), # Alvis T4 only has 4 workers available, A100 has 16
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": 1,
    "in_order": True,
    "with_termini": True,
    "encoding_scheme": "unmod",  # or "naive-mods"
    "sequence_column": "modified_sequence",
    "label_column": "intensities_raw",
    "collision_energy_column": "collision_energy_aligned_normed",
    "precursor_charge_column": "precursor_charge_onehot",
    "ptm_features": "mod_loss,delta_mass",
    "debug_unknown_tokens": _env_bool("DEBUG_UNKNOWN_TOKENS", False),
    "max_train_batches": int(os.environ.get("MAX_TRAIN_BATCHES", 0)),  # 0 = no cap
    "max_val_batches": int(os.environ.get("MAX_VAL_BATCHES", 0)),
    "max_test_batches": int(os.environ.get("MAX_TEST_BATCHES", 0)),
    "save": None,
    "checkpoint_save": os.environ.get("CHECKPOINT_DIR", None), # Give as /path/to/dir
    # --- Optional torch.compile acceleration ---
    "use_torch_compile": False,
    "torch_compile_backend": "inductor",
    "torch_compile_mode": "reduce-overhead",
    "torch_compile_fullgraph": False,
    "torch_compile_dynamic": False,
    # --- Optional mixed precision ---
    "use_amp": _env_bool("USE_AMP", False),
    "amp_dtype": os.environ.get("AMP_DTYPE", "bf16"),  # bf16 | fp16 | float16 | bfloat16
    # --- Optional CUDA math acceleration ---
    "enable_tf32": True,
    "float32_matmul_precision": "high",  # one of: highest, high, medium
    # --- Optional profiling ---
    "profile_timing": _env_bool("PROFILE_TIMING", False),
    "profile_warmup_batches": 100,
    "profile_num_batches": 500,
    "profile_log_every": 100,
    "profile_cuda_sync": True,  # needed for accurate CUDA phase timing
    "profile_dataloader_only_batches": int(os.environ.get("PROFILE_DATALOADER_ONLY_BATCHES", 0)),  # if >0, run loader-only benchmark then exit
    "profile_dataloader_move_to_device": _env_bool("PROFILE_DATALOADER_MOVE_TO_DEVICE", False),
    # --- Optional optimizer / stability knobs (evidence: repo examples) ---
    "grad_clip_max_norm": float(os.environ.get("GRAD_CLIP_MAX_NORM", 1.0)),  # evidence: torch intensity examples clip with max_norm=1
    # "weight_decay": 0.0,  # evidence: not used in repo examples; keep off unless you add it intentionally
    # --- PROSIT-PTM FII schedule knobs (paper hyperparameters) ---
    "lr_schedule": os.environ.get("LR_SCHEDULE", "auto").strip().lower(),  # auto | clr | warmup_cosine | constant
    "use_clr": _env_bool("USE_CLR", True),  # according to (2) PROSIT-PTM (FII uses cyclic learning rate)
    "clr_base_lr": float(os.environ.get("CLR_BASE_LR", 1e-5)),  # according to (2) PROSIT-PTM (lower lr bound)
    "clr_max_lr": float(os.environ.get("CLR_MAX_LR", 2e-4)),  # according to (2) PROSIT-PTM (upper lr bound)
    "clr_scale_gamma": float(os.environ.get("CLR_SCALE_GAMMA", 0.95)),  # according to (2) PROSIT-PTM (upper bound scaled by 0.95 every 8 epochs)
    "clr_scale_every_epochs": int(os.environ.get("CLR_SCALE_EVERY_EPOCHS", 8)),  # according to (2) PROSIT-PTM
    "warmup_cosine_warmup_steps": int(os.environ.get("WARMUP_COSINE_WARMUP_STEPS", os.environ.get("WARMUP_STEPS", 13_690))),
    "warmup_cosine_start_lr": float(os.environ.get("WARMUP_COSINE_START_LR", 1.6e-5)),
    "warmup_cosine_peak_lr": float(os.environ.get("WARMUP_COSINE_PEAK_LR", 1.2e-4)),
    "warmup_cosine_min_lr": float(os.environ.get("WARMUP_COSINE_MIN_LR", 1.6e-5)),
    "warmup_cosine_total_steps": int(os.environ.get("WARMUP_COSINE_TOTAL_STEPS", 0)),
    # Disabled by default: for the uncertainty-aware objective, validation NLL
    # can move differently from spectral angle/MAE because it also includes
    # variance calibration and presence probabilities. Set >0 to re-enable.
    "early_stopping_patience": int(os.environ.get("EARLY_STOPPING_PATIENCE", 0)),
    "select_best_by_val_loss": _env_bool("SELECT_BEST_BY_VAL_LOSS", False),
    # --- Optional training control (evidence: TF examples) ---
    # "use_reduce_on_plateau": True,  # evidence: TF Prosit scripts use ReduceLROnPlateau
    # "reduce_factor": 0.1,  # evidence: `run_scripts/run_prosit_intensity.py`, `run_scripts/run_prosit_intensity_ptms.py`
    # "reduce_patience": 10,  # evidence: same as above
    # "reduce_min_lr": 0.0,  # evidence: same as above
    # "early_stopping_patience": 20,  # evidence: same as above
    # --- Optional model architecture overrides (evidence: constructor defaults) ---
    # "embedding_output_dim": 16,  # evidence: `src/dlomix/models/prosit_torch.py:PrositIntensityPredictor.__init__`
    # "dropout_rate": 0.2,  # evidence: `src/dlomix/models/prosit_torch.py:PrositIntensityPredictor.__init__`
    "dropout_rate": 0.3,  # according to (2) PROSIT-PTM (encoder/meta/PTM dropout=0.3)
    # "latent_dropout_rate": 0.1,  # evidence: same
    # "recurrent_layers_sizes": (256, 512),  # evidence: same
    # "regressor_layer_size": 512,  # evidence: same
    # "len_fion": 6,  # evidence: same
    # --- PROSIT-PTM architecture notes (paper hyperparameters; informational) ---
    # "ptm_mlp_units": (1024, 64, 16),  # according to (2) PROSIT-PTM (PTM feature MLP sizes)
    # "decoder_dropout_rate": 0.5,  # according to (2) PROSIT-PTM (decoder dropout differs from encoder dropout)
    "wandb_run_name": os.environ.get(str("WANDB_NAME")),
    "bce_weight": float(os.environ.get("BCE_WEIGHT", 1)),
    "nll_weight": float(os.environ.get("NLL_WEIGHT", 1)),
}


def _device_from_torch() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unwrap_model_for_state_dict(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps modules as OptimizedModule with the original module at `_orig_mod`.
    return getattr(model, "_orig_mod", model)


def _maybe_compile_model(
    model: torch.nn.Module, args: SimpleNamespace
) -> torch.nn.Module:
    if not bool(getattr(args, "use_torch_compile", False)):
        return model

    if not hasattr(torch, "compile"):
        print(
            "Warning: torch.compile is unavailable in this torch build; using eager mode."
        )
        return model

    backend = getattr(args, "torch_compile_backend", None)
    mode = getattr(args, "torch_compile_mode", None)
    fullgraph = bool(getattr(args, "torch_compile_fullgraph", False))
    dynamic = bool(getattr(args, "torch_compile_dynamic", False))

    compile_kwargs = {
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if backend:
        compile_kwargs["backend"] = backend
    if mode:
        compile_kwargs["mode"] = mode

    try:
        model = torch.compile(model, **compile_kwargs)
        print(f"Enabled torch.compile with args: {compile_kwargs}")
    except Exception as exc:
        print(f"Warning: torch.compile failed ({exc}); falling back to eager mode.")
    return model


def _as_list(csv: str) -> list[str]:
    if csv.strip() == "":
        return []
    return [item.strip() for item in csv.split(",") if item.strip()]


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _resolve_amp_settings(
    device: torch.device, args: SimpleNamespace
) -> tuple[bool, torch.dtype | None, bool]:
    if not bool(getattr(args, "use_amp", False)):
        return False, None, False

    if device.type != "cuda":
        print("Warning: AMP requested but device is not CUDA; disabling AMP.")
        return False, None, False

    amp_dtype_name = str(getattr(args, "amp_dtype", "bf16")).lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    amp_dtype = dtype_map.get(amp_dtype_name)
    if amp_dtype is None:
        print(
            f"Warning: Unsupported amp_dtype={amp_dtype_name!r}; use bf16 or fp16. Disabling AMP."
        )
        return False, None, False

    if amp_dtype is torch.bfloat16:
        bf16_supported = (
            bool(hasattr(torch.cuda, "is_bf16_supported"))
            and torch.cuda.is_bf16_supported()
        )
        if not bf16_supported:
            print(
                "Warning: CUDA bfloat16 is not supported here; falling back to fp16 AMP."
            )
            amp_dtype = torch.float16

    use_grad_scaler = amp_dtype is torch.float16
    return True, amp_dtype, use_grad_scaler


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _amp_autocast_context(
    device: torch.device, enabled: bool, dtype: torch.dtype | None
):
    if not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _parquet_num_rows(parquet_path: str) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(parquet_path).metadata.num_rows


def _estimate_num_steps(
    parquet_path: str | None, batch_size: int, max_batches: int = 0
) -> tuple[int | None, int | None]:
    if parquet_path is None:
        return None, None

    if max_batches:
        return int(max_batches), None

    try:
        num_rows = int(_parquet_num_rows(parquet_path))
    except Exception as exc:
        print(
            f"Warning: Could not estimate num steps from parquet metadata for {parquet_path!r}: {exc}"
        )
        return None, None

    return int(math.ceil(num_rows / batch_size)), num_rows


def _maybe_cuda_sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _new_timing_totals() -> dict[str, float]:
    return {
        "data_wait_s": 0.0,
        "move_cast_s": 0.0,
        "forward_s": 0.0,
        "loss_s": 0.0,
        "backward_s": 0.0,
        "optim_s": 0.0,
        "step_total_s": 0.0,
    }


def _timing_summary(prefix: str, totals: dict[str, float], count: int) -> str:
    if count <= 0:
        return f"{prefix}: no profiled batches yet"

    per_batch = {k: (v / count) for k, v in totals.items()}
    step = max(per_batch["step_total_s"], 1e-12)  # compute step (after batch is ready)
    cycle = max(
        per_batch["step_total_s"] + per_batch["data_wait_s"], 1e-12
    )  # full loop

    def ms(name: str) -> float:
        return per_batch[name] * 1000.0

    def pct_step(name: str) -> float:
        return (per_batch[name] / step) * 100.0

    def pct_cycle(name: str) -> float:
        return (per_batch[name] / cycle) * 100.0

    batches_per_s = 1.0 / cycle
    return (
        f"{prefix}: n={count} cycle={cycle * 1000.0:.2f}ms ({batches_per_s:.2f} batch/s) "
        f"| data_wait={ms('data_wait_s'):.2f}ms ({pct_cycle('data_wait_s'):.1f}% cycle) "
        f"| step={ms('step_total_s'):.2f}ms "
        f"| move_cast={ms('move_cast_s'):.2f}ms ({pct_step('move_cast_s'):.1f}% step) "
        f"| fwd={ms('forward_s'):.2f}ms ({pct_step('forward_s'):.1f}% step) "
        f"| loss={ms('loss_s'):.2f}ms ({pct_step('loss_s'):.1f}% step) "
        f"| bwd={ms('backward_s'):.2f}ms ({pct_step('backward_s'):.1f}% step) "
        f"| opt={ms('optim_s'):.2f}ms ({pct_step('optim_s'):.1f}% step)"
    )


def _run_dataloader_only_profile(
    dataset: StreamingFragmentIonIntensityDataset,
    columns: "ColumnConfig",
    device: torch.device,
    args: SimpleNamespace,
) -> None:
    max_batches = int(getattr(args, "profile_dataloader_only_batches", 0) or 0)
    if max_batches <= 0:
        return

    move_to_device = bool(getattr(args, "profile_dataloader_move_to_device", False))
    print(
        f"[loader-only] profiling {max_batches} batch(es), move_to_device={move_to_device}"
    )

    total_wait_s = 0.0
    total_move_cast_s = 0.0
    total_examples = 0
    seen = 0
    loop_end = time.perf_counter()
    profile_start = loop_end

    it = tqdm(
        dataset.tensor_train_data,
        total=max_batches,
        desc="Loader-only profile",
        unit="batch",
        leave=False,
    )
    for batch in it:
        iter_start = time.perf_counter()
        total_wait_s += iter_start - loop_end

        t0 = time.perf_counter()
        if move_to_device:
            batch = _move_batch_to_device(batch, device)
        batch = _cast_batch_types(batch, columns)
        total_move_cast_s += time.perf_counter() - t0

        if columns.sequence in batch and torch.is_tensor(batch[columns.sequence]):
            total_examples += int(batch[columns.sequence].shape[0])
        else:
            total_examples += int(args.batch_size)

        seen += 1
        loop_end = time.perf_counter()
        if seen >= max_batches:
            break

    elapsed_s = time.perf_counter() - profile_start
    if seen <= 0:
        print("[loader-only] no batches seen")
        return

    print(
        f"[loader-only] batches={seen} examples={total_examples} elapsed={elapsed_s:.2f}s "
        f"throughput={seen / elapsed_s:.2f} batch/s ({total_examples / elapsed_s:.2f} ex/s)"
    )
    print(
        f"[loader-only] avg_wait={(total_wait_s / seen) * 1000.0:.2f}ms/batch "
        f"avg_move_cast={(total_move_cast_s / seen) * 1000.0:.2f}ms/batch"
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _clip_grad_norm(model: torch.nn.Module, max_norm: float) -> float:
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return float(total_norm.detach().cpu())


def _triangular_lr(progress_0_to_1: float, base_lr: float, max_lr: float) -> float:
    progress_0_to_1 = max(0.0, min(1.0, float(progress_0_to_1)))
    triangle = 1.0 - abs(2.0 * progress_0_to_1 - 1.0)  # 0 -> 1 -> 0
    return base_lr + triangle * (max_lr - base_lr)


def _resolve_lr_schedule(config_value: str, use_clr: bool) -> str:
    schedule = str(config_value or "auto").strip().lower().replace("-", "_")
    if schedule == "auto":
        return "clr" if use_clr else "constant"

    aliases = {
        "none": "constant",
        "off": "constant",
        "cyclic": "clr",
        "cyclic_lr": "clr",
        "triangular": "clr",
        "cosine": "warmup_cosine",
        "warmup": "warmup_cosine",
        "warmup_cosine_decay": "warmup_cosine",
    }
    schedule = aliases.get(schedule, schedule)
    if schedule not in {"constant", "clr", "warmup_cosine"}:
        raise ValueError(
            f"Unknown LR_SCHEDULE={config_value!r}; expected auto, clr, warmup_cosine, or constant."
        )
    return schedule


def _warmup_cosine_lr(
    step_0_based: int,
    total_steps: int,
    warmup_steps: int,
    start_lr: float,
    peak_lr: float,
    min_lr: float,
) -> float:
    step = max(0, int(step_0_based))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    if warmup_steps > 0 and step < warmup_steps:
        warmup_progress = float(step) / float(max(1, warmup_steps))
        return start_lr + warmup_progress * (peak_lr - start_lr)

    decay_steps = max(1, total_steps - warmup_steps)
    decay_progress = float(step - warmup_steps) / float(decay_steps)
    decay_progress = max(0.0, min(1.0, decay_progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + cosine * (peak_lr - min_lr)


@dataclass(frozen=True)
class ColumnConfig:
    sequence: str
    label: str
    collision_energy: str
    precursor_charge: str


def _cast_batch_types(batch: dict, columns: ColumnConfig) -> dict:
    # Embedding indices must be integer type (Long recommended).
    if columns.sequence in batch and torch.is_tensor(batch[columns.sequence]):
        batch[columns.sequence] = batch[columns.sequence].to(dtype=torch.long)

    # Labels and numeric features should be floating for loss/MLP layers.
    float_keys: Iterable[str] = [
        columns.label,
        columns.collision_energy,
        columns.precursor_charge,
        # PTM features (if present)
        "mod_loss",
        "delta_mass",
        "mod_gain",
        "atom_count",
        "red_smiles",
    ]
    for key in float_keys:
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(dtype=torch.float32)

    return batch


def _log_mean_to_intensity(
    pred_log_mean: torch.Tensor,
    pred_presence_logit: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    # The uncertainty-aware mean head predicts mu in log-intensity space.
    # Metrics operate on raw nonnegative intensities, so convert exp(mu) - eps.
    # Clamp only to avoid metric-side overflow if a bad run emits very large
    # positive log means; the loss itself remains unclamped.
    pred_intensity = torch.clamp(
        torch.exp(torch.clamp(pred_log_mean.detach().float(), max=20.0)) - epsilon,
        min=0.0,
    )
    if pred_presence_logit is None:
        return pred_intensity

    # Use the zero-inflated model's expected observed-intensity proxy for
    # monitoring metrics: conditional lognormal median times P(present).
    presence_probability = torch.sigmoid(pred_presence_logit.detach().float())
    return pred_intensity * presence_probability


def _target_intensity_and_valid_mask_for_metrics(
    y_true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = y_true.detach().float()
    valid = target >= 0
    return target, valid


def _mae_from_log_mean(
    y_true: torch.Tensor,
    pred_log_mean: torch.Tensor,
    pred_presence_logit: torch.Tensor | None = None,
) -> float:
    pred_intensity = _log_mean_to_intensity(pred_log_mean, pred_presence_logit)
    target, valid = _target_intensity_and_valid_mask_for_metrics(y_true)
    if not torch.any(valid):
        return 0.0
    return torch.mean(torch.abs(pred_intensity[valid] - target[valid])).item()


def _spectral_angle_from_log_mean(
    y_true: torch.Tensor,
    pred_log_mean: torch.Tensor,
    pred_presence_logit: torch.Tensor | None = None,
) -> float:
    pred_intensity = _log_mean_to_intensity(pred_log_mean, pred_presence_logit)
    return 1.0 - masked_spectral_distance(y_true.detach().float(), pred_intensity).item()


def _variance_diagnostics(
    y_true: torch.Tensor,
    pred_log_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    prefix: str,
    epsilon: float = 1e-7,
) -> dict:
    # The Gaussian term is only evaluated for present fragments, so these
    # diagnostics use the same domain. The loss itself remains unclamped; the
    # metric-side exp() is clipped only to avoid logging inf values to W&B.
    y_true = y_true.detach().float()
    pred_log_mean = pred_log_mean.detach().float()
    pred_log_var = pred_log_var.detach().float()

    present = y_true > 0
    if not torch.any(present):
        return {}

    log_var = pred_log_var[present]
    finite = torch.isfinite(log_var)
    if not torch.any(finite):
        return {}

    log_var = log_var[finite]
    log_target = torch.log(y_true[present][finite] + epsilon)
    squared_log_residual = torch.square(log_target - pred_log_mean[present][finite])

    log_var_for_exp = torch.clamp(log_var, min=-30.0, max=30.0)
    pred_var = torch.exp(log_var_for_exp)
    pred_sigma = torch.exp(0.5 * log_var_for_exp)
    calibration_ratio = squared_log_residual / torch.clamp(pred_var, min=1e-12)

    return {
        f"{prefix}_pred_var_present_mean": pred_var.mean().item(),
        f"{prefix}_pred_var_present_max": pred_var.max().item(),
        f"{prefix}_pred_sigma_present_mean": pred_sigma.mean().item(),
        f"{prefix}_variance_calibration_ratio_present_mean": calibration_ratio
        .mean()
        .item(),
    }


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    finite_count = int(finite.sum().item())
    total = data.numel()
    if finite_count == 0:
        return f"{name}: shape={tuple(data.shape)} finite=0/{total}"
    finite_data = data[finite]
    return (
        f"{name}: shape={tuple(data.shape)} finite={finite_count}/{total} "
        f"min={finite_data.min().item():.6g} max={finite_data.max().item():.6g} "
        f"mean={finite_data.mean().item():.6g}"
    )


def _raise_for_nonfinite_loss(
    loss: torch.Tensor,
    *,
    epoch: int,
    train_batches: int,
    batch: dict,
    columns: ColumnConfig,
    pred_log_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    pred_presence_logit: torch.Tensor,
) -> None:
    if torch.isfinite(loss):
        return

    y_true = batch[columns.label]
    valid = y_true >= 0
    present = y_true > 0
    msg = [
        f"Non-finite training loss at epoch={epoch} batch={train_batches + 1}: {loss.item()}",
        f"label valid={int(valid.sum().item())}/{valid.numel()} present={int(present.sum().item())}/{present.numel()}",
        _tensor_summary("y_true", y_true),
        _tensor_summary("pred_log_mean", pred_log_mean),
        _tensor_summary("pred_log_var", pred_log_var),
        _tensor_summary("pred_presence_logit", pred_presence_logit),
    ]
    raise FloatingPointError("\n".join(msg))



def _configure_wandb_axes() -> None:
    # Make a single x-axis available for every panel. W&B's default `_step` is
    # just the number of log calls, and separate `train_step`/`val_step` fields
    # are awkward to use in shared train/validation dashboards.
    wandb.define_metric("global_step")
    for pattern in (
        "train_*",
        "val_*",
        "current_epoch_*",
        "epoch_average_*",
    ):
        wandb.define_metric(pattern, step_metric="global_step")


def _wandb_log(run, global_step: int, metrics: dict) -> int:
    run.log({"global_step": global_step, **metrics})
    return global_step + 1


def _print_ptm_handling_probe(
    *,
    parquet_path: str,
    sequence_column: str,
    with_termini: bool,
    encoding_scheme: str,
    ptm_features: list[str],
    probe_rows: int = 256,
) -> None:
    """
    Best-effort, low-cost probe showing how PTMs flow through:
    - parsed ProForma tokens
    - (optional) UNMOD PTM stripping step
    - PTM feature lookup coverage for requested `features_to_extract`
    """
    try:
        import pyarrow.parquet as pq

        from dlomix.constants import ALPHABET_UNMOD
        from dlomix.data.processing.feature_extractors import (
            FEATURE_EXTRACTORS_PARAMETERS,
        )
        from dlomix.data.processing.processors import (
            SequenceParsingProcessor,
            SequencePTMRemovalProcessor,
        )
    except Exception as exc:
        print(f"[ptm-probe] skipped (missing deps): {exc}")
        return

    try:
        pf = pq.ParquetFile(parquet_path)
        rb = next(
            pf.iter_batches(batch_size=probe_rows, columns=[sequence_column]),
            None,
        )
        if rb is None:
            print("[ptm-probe] skipped (empty parquet)")
            return
        batch = rb.to_pydict()
    except Exception as exc:
        print(f"[ptm-probe] skipped (failed reading parquet): {exc}")
        return

    parser = SequenceParsingProcessor(
        sequence_column_name=sequence_column, batched=True, with_termini=with_termini
    )
    batch = parser(batch)

    print(
        f"[ptm-probe] config: encoding_scheme={encoding_scheme} with_termini={with_termini} probe_rows={probe_rows}"
    )

    parsed_tokens: list[str] = []
    for n_term, seq, c_term in zip(
        batch["_n_term_mods"], batch["_parsed_sequence"], batch["_c_term_mods"]
    ):
        parsed_tokens.extend([n_term, *seq, c_term])

    parsed_unimod_tokens = sorted({t for t in parsed_tokens if "UNIMOD:" in t})
    parsed_unimod_residue = sorted(
        {t for t in parsed_unimod_tokens if len(t) > 0 and t[0].isalpha()}
    )
    parsed_unimod_termini = sorted(
        {t for t in parsed_unimod_tokens if not (len(t) > 0 and t[0].isalpha())}
    )

    print(
        "[ptm-probe] observed UNIMOD tokens (raw parse): "
        f"total={len(parsed_unimod_tokens)} residues={len(parsed_unimod_residue)} termini={len(parsed_unimod_termini)}"
    )
    print(f"[ptm-probe] UNIMOD sample: {parsed_unimod_tokens[:10]}")

    if encoding_scheme == "unmod":
        remover = SequencePTMRemovalProcessor(
            sequence_column_name=sequence_column, batched=True
        )
        removed = remover(batch)
        # remover returns only {sequence_column: ...}
        batch.update(removed)

        seq_tokens = []
        for seq in batch[sequence_column]:
            seq_tokens.extend(seq)
        remaining_unimod = sorted({t for t in seq_tokens if "UNIMOD:" in t})
        alphabet_unimod = sorted([k for k in ALPHABET_UNMOD.keys() if "UNIMOD:" in k])
        unknown_in_alphabet = sorted(
            [t for t in remaining_unimod if t not in ALPHABET_UNMOD]
        )
        print(
            "[ptm-probe] embedded-seq UNIMOD tokens after UNMOD stripping: "
            f"count={len(remaining_unimod)} tokens={remaining_unimod}"
        )
        print(
            f"[ptm-probe] ALPHABET_UNMOD supports UNIMOD tokens: count={len(alphabet_unimod)} tokens={alphabet_unimod}"
        )
        if unknown_in_alphabet:
            print(
                f"[ptm-probe] NOTE: embedded-seq UNIMOD tokens not in ALPHABET_UNMOD (will become unknown X): {unknown_in_alphabet}"
            )
    else:
        print(
            "[ptm-probe] embedded-seq tokens: PTMs preserved (encoding_scheme=naive-mods)"
        )

    # Feature lookup coverage (for residue/PTM tokens primarily)
    residue_unimod = parsed_unimod_residue
    for feat in ptm_features:
        feat = feat.lower()
        params = FEATURE_EXTRACTORS_PARAMETERS.get(feat)
        if not params:
            continue
        lookup = params["lookup_table"]
        missing = [t for t in residue_unimod if t not in lookup]
        print(
            f"[ptm-probe] feature={feat}: residue UNIMOD tokens missing from lookup={len(missing)} of {len(residue_unimod)} (missing_sample={missing[:10]})"
        )


def main() -> int:
    args = SimpleNamespace(**CONFIG)
    lr_schedule = _resolve_lr_schedule(args.lr_schedule, bool(args.use_clr))

    device = _device_from_torch()
    print(f"Using device: {device}")
    print(f"Using weights: BCE = {args.bce_weight}, NLL = {args.nll_weight}")

    if args.checkpoint_save:
        os.makedirs(args.checkpoint_save, exist_ok=True)

    # Initialize wandb
    run = wandb.init(
        entity="kall",
        project="prosit_uncertainty_aware",
        name=args.wandb_run_name,
        # If additional config variables are uncommented under CONFIG add them here
        config={
            "learning_rate": args.lr,
            "dataset":"PROSPECT",
            "epochs":args.epochs,
            "uncertainty_aware": args.uncertainty_aware, 
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len, 
            "shuffle": args.shuffle,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "parquet_read_batch_size": args.parquet_read_batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "in_order": args.in_order,
            "with_termini": args.with_termini,
            "encoding_scheme": args.encoding_scheme,
            "sequence_column": args.sequence_column,
            "label_column": args.label_column,
            "collision_energy_column": args.collision_energy_column,
            "precursor_charge_column": args.precursor_charge_column,
            "ptm_features": args.ptm_features,
            "debug_unknown_tokens": args.debug_unknown_tokens,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "max_test_batches": args.max_test_batches,
            "save": args.save,
            "checkpoint_save": args.checkpoint_save,
            "use_torch_compile": args.use_torch_compile,
            "torch_compile_backend": args.torch_compile_backend,
            "torch_compile_mode": args.torch_compile_mode,
            "torch_compile_fullgraph": args.torch_compile_fullgraph,
            "torch_compile_dynamic": args.torch_compile_dynamic,
            "use_amp": args.use_amp,
            "amp_dtype": args.amp_dtype,
            "enable_tf32": args.enable_tf32,
            "float32_matmul_precision": args.float32_matmul_precision,
            "profile_timing": args.profile_timing,
            "profile_warmup_batches": args.profile_warmup_batches,
            "profile_num_batches": args.profile_num_batches,
            "profile_log_every": args.profile_log_every,
            "profile_cuda_sync": args.profile_cuda_sync,
            "profile_dataloader_only_batches": args.profile_dataloader_only_batches,
            "profile_dataloader_move_to_device": args.profile_dataloader_move_to_device,
            "grad_clip_max_norm": args.grad_clip_max_norm,
            "lr_schedule": lr_schedule,
            "use_clr": lr_schedule == "clr",
            "clr_base_lr": args.clr_base_lr,
            "clr_max_lr": args.clr_max_lr,
            "clr_scale_gamma": args.clr_scale_gamma,
            "clr_scale_every_epochs": args.clr_scale_every_epochs,
            "warmup_cosine_warmup_steps": args.warmup_cosine_warmup_steps,
            "warmup_cosine_start_lr": args.warmup_cosine_start_lr,
            "warmup_cosine_peak_lr": args.warmup_cosine_peak_lr,
            "warmup_cosine_min_lr": args.warmup_cosine_min_lr,
            "warmup_cosine_total_steps": args.warmup_cosine_total_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "dropout_rate": args.dropout_rate,
            "bce_weight": args.bce_weight,
            "nll_weight": args.nll_weight,
        }
    )

    _configure_wandb_axes()

    if device.type == "cuda":
        try:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(
                    str(getattr(args, "float32_matmul_precision", "high"))
                )
            if bool(getattr(args, "enable_tf32", False)):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 for CUDA matmul/cuDNN.")
        except Exception as exc:
            print(f"Warning: failed to configure CUDA TF32 options: {exc}")

    amp_enabled, amp_dtype, amp_use_grad_scaler = _resolve_amp_settings(device, args)
    scaler = _make_grad_scaler(enabled=True) if amp_use_grad_scaler else None
    if amp_enabled:
        amp_dtype_name = "bf16" if amp_dtype is torch.bfloat16 else "fp16"
        print(
            f"Enabled AMP autocast with dtype={amp_dtype_name}; grad_scaler={amp_use_grad_scaler}"
        )

    columns = ColumnConfig(
        sequence=args.sequence_column,
        label=args.label_column,
        collision_energy=args.collision_energy_column,
        precursor_charge=args.precursor_charge_column,
    )

    train_path = args.train
    val_path = args.val
    test_path = args.test

    ptm_features = _as_list(args.ptm_features)
    model_features = [columns.collision_energy, columns.precursor_charge]

    _print_ptm_handling_probe(
        parquet_path=train_path,
        sequence_column=columns.sequence,
        with_termini=args.with_termini,
        encoding_scheme=args.encoding_scheme,
        ptm_features=ptm_features,
    )

    if val_path is None:
        raise ValueError(
            "Streaming mode requires an explicit --val parquet (no in-script train/val split)."
        )

    dataset = StreamingFragmentIonIntensityDataset(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=0,
        with_termini=args.with_termini,
        encoding_scheme=args.encoding_scheme,
        model_features=model_features,
        features_to_extract=ptm_features,
        sequence_column=columns.sequence,
        label_column=columns.label,
        parquet_read_batch_size=args.parquet_read_batch_size,
        return_debug_tokens=bool(args.debug_unknown_tokens),
        num_workers=int(getattr(args, "num_workers", 0) or 0),
        pin_memory=bool(getattr(args, "pin_memory", False)),
        prefetch_factor=(
            int(getattr(args, "prefetch_factor"))
            if getattr(args, "prefetch_factor", None) is not None
            else None
        ),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
        in_order=bool(getattr(args, "in_order", True)),
    )
    print(
        f"DataLoader config: num_workers={int(getattr(args, 'num_workers', 0) or 0)} "
        f"pin_memory={bool(getattr(args, 'pin_memory', False))} "
        f"persistent_workers={bool(getattr(args, 'persistent_workers', False))} "
        f"prefetch_factor={getattr(args, 'prefetch_factor', None)} "
        f"in_order={bool(getattr(args, 'in_order', True))}"
    )

    _run_dataloader_only_profile(dataset, columns, device, args)
    if int(getattr(args, "profile_dataloader_only_batches", 0) or 0) > 0:
        return 0

    if args.uncertainty_aware == True:
        model = PrositIntensityUncertaintyPredictor(
            seq_length=args.max_seq_len,
            **{
                k: getattr(args, k)
                for k in (
                    "embedding_output_dim",
                    "dropout_rate",
                    "latent_dropout_rate",
                    "recurrent_layers_sizes",
                    "regressor_layer_size",
                    "len_fion",
                )
                if hasattr(args, k)
            },
            use_prosit_ptm_features=True,
            input_keys={"SEQUENCE_KEY": columns.sequence},
            meta_data_keys={
                "COLLISION_ENERGY_KEY": columns.collision_energy,
                "PRECURSOR_CHARGE_KEY": columns.precursor_charge,
            },
            with_termini=args.with_termini,
        ).to(device)
        model = _maybe_compile_model(model, args)
    else:
        model = PrositIntensityPredictor(
            seq_length=args.max_seq_len,
            **{
                k: getattr(args, k)
                for k in (
                    "embedding_output_dim",
                    "dropout_rate",
                    "latent_dropout_rate",
                    "recurrent_layers_sizes",
                    "regressor_layer_size",
                    "len_fion",
                )
                if hasattr(args, k)
            },
            use_prosit_ptm_features=True,
            input_keys={"SEQUENCE_KEY": columns.sequence},
            meta_data_keys={
                "COLLISION_ENERGY_KEY": columns.collision_energy,
                "PRECURSOR_CHARGE_KEY": columns.precursor_charge,
            },
            with_termini=args.with_termini,
        ).to(device)
        model = _maybe_compile_model(model, args)

    clr_enabled = lr_schedule == "clr"
    warmup_cosine_enabled = lr_schedule == "warmup_cosine"
    clr_base_lr = float(getattr(args, "clr_base_lr", args.lr))
    clr_max_lr = float(getattr(args, "clr_max_lr", args.lr))
    clr_gamma = float(getattr(args, "clr_scale_gamma", 1.0))
    clr_every = int(getattr(args, "clr_scale_every_epochs", 0) or 0)
    warmup_cosine_warmup_steps = int(
        getattr(args, "warmup_cosine_warmup_steps", 13_690) or 0
    )
    warmup_cosine_start_lr = float(
        getattr(args, "warmup_cosine_start_lr", 1.6e-5)
    )
    warmup_cosine_peak_lr = float(
        getattr(args, "warmup_cosine_peak_lr", 1.2e-4)
    )
    warmup_cosine_min_lr = float(
        getattr(args, "warmup_cosine_min_lr", warmup_cosine_start_lr)
    )

    if warmup_cosine_enabled:
        initial_lr = warmup_cosine_start_lr
    elif clr_enabled:
        initial_lr = clr_base_lr
    else:
        initial_lr = float(args.lr)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)

    grad_clip = float(getattr(args, "grad_clip_max_norm", 1.0))
    early_patience = int(getattr(args, "early_stopping_patience", 0) or 0)
    select_best_by_val_loss = bool(getattr(args, "select_best_by_val_loss", False)) or early_patience > 0

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    train_steps_est, train_rows = _estimate_num_steps(
        train_path, args.batch_size, int(getattr(args, "max_train_batches", 0) or 0)
    )
    val_steps_est, val_rows = _estimate_num_steps(
        val_path, args.batch_size, int(getattr(args, "max_val_batches", 0) or 0)
    )
    test_steps_est, test_rows = _estimate_num_steps(
        test_path, args.batch_size, int(getattr(args, "max_test_batches", 0) or 0)
    )

    total_train_steps_for_schedule = None
    if warmup_cosine_enabled:
        total_train_steps_for_schedule = int(
            getattr(args, "warmup_cosine_total_steps", 0) or 0
        )
        if total_train_steps_for_schedule <= 0:
            if train_steps_est is None:
                raise ValueError(
                    "LR_SCHEDULE=warmup_cosine requires estimated train steps or "
                    "WARMUP_COSINE_TOTAL_STEPS."
                )
            total_train_steps_for_schedule = max(
                1, int(train_steps_est) * int(args.epochs)
            )
        if total_train_steps_for_schedule <= warmup_cosine_warmup_steps:
            print(
                "Warning: warmup steps cover the full scheduled training run "
                f"(warmup_steps={warmup_cosine_warmup_steps}, "
                f"total_steps={total_train_steps_for_schedule})."
            )
        run.config.update(
            {"warmup_cosine_resolved_total_steps": total_train_steps_for_schedule},
            allow_val_change=True,
        )

    if train_steps_est is not None:
        if train_rows is None:
            print(f"Estimated train steps/epoch: {train_steps_est} (capped)")
        else:
            print(
                f"Estimated train steps/epoch: {train_steps_est} (rows={train_rows}, batch_size={args.batch_size})"
            )
    if val_steps_est is not None:
        if val_rows is None:
            print(f"Estimated val steps/epoch: {val_steps_est} (capped)")
        else:
            print(
                f"Estimated val steps/epoch: {val_steps_est} (rows={val_rows}, batch_size={args.batch_size})"
            )
    if test_steps_est is not None:
        if test_rows is None:
            print(f"Estimated test steps: {test_steps_est} (capped)")
        else:
            print(
                f"Estimated test steps: {test_steps_est} (rows={test_rows}, batch_size={args.batch_size})"
            )

    profile_enabled = bool(getattr(args, "profile_timing", False))
    profile_warmup = int(getattr(args, "profile_warmup_batches", 100) or 0)
    profile_num_batches = int(getattr(args, "profile_num_batches", 500) or 0)
    profile_log_every = int(getattr(args, "profile_log_every", 100) or 0)
    profile_cuda_sync = bool(getattr(args, "profile_cuda_sync", True))
    prof_totals = _new_timing_totals()
    prof_count = 0
    seen_train_batches = 0
    if profile_enabled:
        print(
            f"[profile] enabled (warmup={profile_warmup}, sample_batches={profile_num_batches}, "
            f"log_every={profile_log_every}, cuda_sync={profile_cuda_sync})"
        )
    global_step = 0

    if args.uncertainty_aware == True:
        print("Running uncertainty aware training")
        train_step = 0
        val_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_mae_total = 0.0
            train_msa_total = 0.0
            train_batches = 0
            loop_end = time.perf_counter()

            train_it = tqdm(
                dataset.tensor_train_data,
                total=train_steps_est,
                desc=f"Epoch {epoch:03d} [train]",
                unit="batch",
                leave=False,
            )
            for batch in train_it:
                iter_start = time.perf_counter()
                data_wait_s = iter_start - loop_end

                do_profile = (
                    profile_enabled
                    and seen_train_batches >= profile_warmup
                    and (profile_num_batches <= 0 or prof_count < profile_num_batches)
                )

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_move_cast_0 = time.perf_counter()
                batch = _move_batch_to_device(batch, device)
                batch = _cast_batch_types(batch, columns)
                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    move_cast_s = time.perf_counter() - t_move_cast_0

                optimizer.zero_grad(set_to_none=True)

                if clr_enabled:
                    denom = max(1, (train_steps_est or 1) - 1)
                    progress = float(train_batches) / float(denom)
                    lr_now = _triangular_lr(progress, clr_base_lr, clr_max_lr)
                    _set_optimizer_lr(optimizer, lr_now)
                elif warmup_cosine_enabled:
                    lr_now = _warmup_cosine_lr(
                        train_step,
                        int(total_train_steps_for_schedule or 1),
                        warmup_cosine_warmup_steps,
                        warmup_cosine_start_lr,
                        warmup_cosine_peak_lr,
                        warmup_cosine_min_lr,
                    )
                    _set_optimizer_lr(optimizer, lr_now)

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_fwd_0 = time.perf_counter()
                with _amp_autocast_context(device, amp_enabled, amp_dtype):
                    pred_log_mean, pred_log_var, pred_presence_logit = model(batch)
                    if do_profile:
                        _maybe_cuda_sync(device, profile_cuda_sync)
                        forward_s = time.perf_counter() - t_fwd_0

                    if args.debug_unknown_tokens:
                        seq = batch[columns.sequence]
                        unknown_token_index = getattr(dataset, "unknown_token_index", 23)
                        any_unknowns = (seq == unknown_token_index).sum()
                        if any_unknowns > 0:
                            print(
                                f"Warning: Found {any_unknowns} unknown tokens (id={unknown_token_index}) in batch sequences."
                            )
                            if "_debug_input_tokens" in batch:
                                debug_tokens = batch["_debug_input_tokens"]
                                printed = 0
                                for ex_i in range(seq.shape[0]):
                                    pos = (seq[ex_i] == unknown_token_index).nonzero(
                                        as_tuple=False
                                    )
                                    if pos.numel() == 0:
                                        continue
                                    for p in pos.flatten().tolist():
                                        try:
                                            tok = debug_tokens[ex_i][p]
                                        except Exception:
                                            tok = "<unavailable>"
                                        print(f"  example={ex_i} pos={p} token={tok!r}")
                                        printed += 1
                                        if printed >= 10:
                                            break
                                    if printed >= 10:
                                        break

                    if do_profile:
                        _maybe_cuda_sync(device, profile_cuda_sync)
                        t_loss_0 = time.perf_counter()
                    loss = gaussian_nll(batch[columns.label], pred_log_mean, pred_log_var, pred_presence_logit, batch[columns.sequence], args.bce_weight, args.nll_weight)
                    _raise_for_nonfinite_loss(
                        loss,
                        epoch=epoch,
                        train_batches=train_batches,
                        batch=batch,
                        columns=columns,
                        pred_log_mean=pred_log_mean,
                        pred_log_var=pred_log_var,
                        pred_presence_logit=pred_presence_logit,
                    )
                    if do_profile:
                        _maybe_cuda_sync(device, profile_cuda_sync)
                        loss_s = time.perf_counter() - t_loss_0

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_bwd_0 = time.perf_counter()
                grad_norm = None
                if amp_use_grad_scaler and scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        grad_norm = _clip_grad_norm(model, grad_clip)
                else:
                    loss.backward()
                    if grad_clip > 0:
                        grad_norm = _clip_grad_norm(model, grad_clip)
                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    backward_s = time.perf_counter() - t_bwd_0

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_opt_0 = time.perf_counter()
                if amp_use_grad_scaler and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    optim_s = time.perf_counter() - t_opt_0
                train_loss_total += loss.item()
                train_batches += 1
                train_it.set_postfix(loss=f"{loss.item():.4f}")

                batch_mae = _mae_from_log_mean(
                    batch[columns.label], pred_log_mean, pred_presence_logit
                )
                batch_msa = _spectral_angle_from_log_mean(
                    batch[columns.label], pred_log_mean, pred_presence_logit
                )
                batch_variance_stats = _variance_diagnostics(
                    batch[columns.label], pred_log_mean, pred_log_var, "train_batch"
                )

                train_mae_total += batch_mae
                train_msa_total += batch_msa

                train_step += 1

                train_metrics = {
                    "train_step": train_step,
                    "train_batch_loss": loss.item(),
                    "train_batch_learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss_total": train_loss_total,
                    "current_epoch_average_train_loss": train_loss_total
                    / max(1, train_batches),
                    "train_batch_mean_absolute_error": batch_mae,
                    "train_batch_mean_spectral_angle": batch_msa,
                }
                if grad_norm is not None:
                    train_metrics["train_batch_grad_norm"] = grad_norm
                train_metrics.update(batch_variance_stats)
                global_step = _wandb_log(run, global_step, train_metrics)

                iter_end = time.perf_counter()
                if do_profile:
                    prof_totals["data_wait_s"] += data_wait_s
                    prof_totals["move_cast_s"] += move_cast_s
                    prof_totals["forward_s"] += forward_s
                    prof_totals["loss_s"] += loss_s
                    prof_totals["backward_s"] += backward_s
                    prof_totals["optim_s"] += optim_s
                    prof_totals["step_total_s"] += iter_end - iter_start
                    prof_count += 1

                    if profile_log_every > 0 and (prof_count % profile_log_every == 0):
                        print(_timing_summary("[profile][train]", prof_totals, prof_count))

                seen_train_batches += 1
                loop_end = iter_end
                if args.max_train_batches and train_batches >= args.max_train_batches:
                    break

            avg_train_loss = train_loss_total / max(1, train_batches)
            avg_train_mae = train_mae_total / max(1, train_batches)
            avg_train_sa = train_msa_total / max(1, train_batches)

            # Validation
            model.eval()
            val_loss_total = 0.0
            val_mean_absolute_error_total = 0.0
            val_spectral_angle_total = 0.0
            val_batches = 0
            with torch.no_grad():
                val_it = tqdm(
                    dataset.tensor_val_data,
                    total=val_steps_est,
                    desc=f"Epoch {epoch:03d} [val]",
                    unit="batch",
                    leave=False,
                )
                for batch in val_it:
                    batch = _move_batch_to_device(batch, device)
                    batch = _cast_batch_types(batch, columns)
                    with _amp_autocast_context(device, amp_enabled, amp_dtype):
                        pred_log_mean, pred_log_var, pred_presence_logit = model(batch)
                        val_loss = gaussian_nll(batch[columns.label], pred_log_mean, pred_log_var, pred_presence_logit, batch[columns.sequence], args.bce_weight, args.nll_weight)
                    _raise_for_nonfinite_loss(
                        val_loss,
                        epoch=epoch,
                        train_batches=val_batches,
                        batch=batch,
                        columns=columns,
                        pred_log_mean=pred_log_mean,
                        pred_log_var=pred_log_var,
                        pred_presence_logit=pred_presence_logit,
                    )
                    val_loss_total += val_loss.item()
                    val_batches += 1
                    
                    batch_mae = _mae_from_log_mean(
                        batch[columns.label], pred_log_mean, pred_presence_logit
                    )
                    batch_msa = _spectral_angle_from_log_mean(
                        batch[columns.label], pred_log_mean, pred_presence_logit
                    )
                    batch_variance_stats = _variance_diagnostics(
                        batch[columns.label], pred_log_mean, pred_log_var, "val_batch"
                    )

                    val_mean_absolute_error_total += batch_mae
                    val_spectral_angle_total += batch_msa

                    val_step += 1

                    val_metrics = {
                        "val_step": val_step,
                        "val_batch_loss": val_loss.item(),
                        "val_loss_total": val_loss_total,
                        "current_epoch_average_val_loss": val_loss_total
                        / max(1, val_batches),
                        "val_batch_mean_absolute_error": batch_mae,
                        "val_batch_mean_spectral_angle": batch_msa,
                    }
                    val_metrics.update(batch_variance_stats)
                    global_step = _wandb_log(run, global_step, val_metrics)

                    val_it.set_postfix(loss=f"{val_loss.item():.4f}")
                    if args.max_val_batches and val_batches >= args.max_val_batches:
                        break
            avg_val_loss = val_loss_total / max(1, val_batches)
            avg_val_mae = val_mean_absolute_error_total / max(1, val_batches)
            avg_val_sa = val_spectral_angle_total / max(1, val_batches)

            if select_best_by_val_loss:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    raw_model = _unwrap_model_for_state_dict(model)
                    best_state = {
                        k: v.detach().cpu() for k, v in raw_model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            print(
                f"Epoch {epoch:03d}: train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}"
            )
            if profile_enabled:
                print(
                    _timing_summary(
                        f"[profile][epoch={epoch:03d}]", prof_totals, prof_count
                    )
                )
            
            global_step = _wandb_log(run, global_step, {
                "epoch": epoch,
                "epoch_average_train_loss": avg_train_loss, 
                "epoch_average_train_mean_absolute_error": avg_train_mae,
                "epoch_average_train_mean_spectral_angle": avg_train_sa,
                "epoch_average_validation_loss": avg_val_loss,
                "epoch_average_validation_mean_absolute_error": avg_val_mae,
                "epoch_average_validation_mean_spectral_angle": avg_val_sa,
            })

            if args.checkpoint_save:
                torch.save(
                    {
                        "uncertainty_aware": args.uncertainty_aware,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    f"{args.checkpoint_save}/checkpoint_epoch_{epoch}.pth",
                )

            if clr_enabled and clr_every > 0 and (epoch % clr_every == 0):
                clr_max_lr *= clr_gamma

            if early_patience > 0 and epochs_without_improvement >= early_patience:
                print(
                    f"Early stopping: no val improvement for {epochs_without_improvement} epoch(s) (patience={early_patience})."
                )
                break
    else:
        print("Running standard training")
        train_step = 0
        val_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_mae_total = 0.0
            train_msa_total = 0.0
            train_batches = 0
            loop_end = time.perf_counter()

            train_it = tqdm(
                dataset.tensor_train_data,
                total=train_steps_est,
                desc=f"Epoch {epoch:03d} [train]",
                unit="batch",
                leave=False,
            )
            for batch in train_it:
                iter_start = time.perf_counter()
                data_wait_s = iter_start - loop_end

                do_profile = (
                    profile_enabled
                    and seen_train_batches >= profile_warmup
                    and (profile_num_batches <= 0 or prof_count < profile_num_batches)
                )

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_move_cast_0 = time.perf_counter()
                batch = _move_batch_to_device(batch, device)
                batch = _cast_batch_types(batch, columns)
                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    move_cast_s = time.perf_counter() - t_move_cast_0

                optimizer.zero_grad(set_to_none=True)

                if clr_enabled:
                    denom = max(1, (train_steps_est or 1) - 1)
                    progress = float(train_batches) / float(denom)
                    lr_now = _triangular_lr(progress, clr_base_lr, clr_max_lr)
                    _set_optimizer_lr(optimizer, lr_now)
                elif warmup_cosine_enabled:
                    lr_now = _warmup_cosine_lr(
                        train_step,
                        int(total_train_steps_for_schedule or 1),
                        warmup_cosine_warmup_steps,
                        warmup_cosine_start_lr,
                        warmup_cosine_peak_lr,
                        warmup_cosine_min_lr,
                    )
                    _set_optimizer_lr(optimizer, lr_now)

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_fwd_0 = time.perf_counter()
                with _amp_autocast_context(device, amp_enabled, amp_dtype):
                    pred = model(batch)
                    if do_profile:
                        _maybe_cuda_sync(device, profile_cuda_sync)
                        forward_s = time.perf_counter() - t_fwd_0

                    if args.debug_unknown_tokens:
                        seq = batch[columns.sequence]
                        unknown_token_index = getattr(dataset, "unknown_token_index", 23)
                        any_unknowns = (seq == unknown_token_index).sum()
                        if any_unknowns > 0:
                            print(
                                f"Warning: Found {any_unknowns} unknown tokens (id={unknown_token_index}) in batch sequences."
                            )
                            if "_debug_input_tokens" in batch:
                                debug_tokens = batch["_debug_input_tokens"]
                                printed = 0
                                for ex_i in range(seq.shape[0]):
                                    pos = (seq[ex_i] == unknown_token_index).nonzero(
                                        as_tuple=False
                                    )
                                    if pos.numel() == 0:
                                        continue
                                    for p in pos.flatten().tolist():
                                        try:
                                            tok = debug_tokens[ex_i][p]
                                        except Exception:
                                            tok = "<unavailable>"
                                        print(f"  example={ex_i} pos={p} token={tok!r}")
                                        printed += 1
                                        if printed >= 10:
                                            break
                                    if printed >= 10:
                                        break

                    if do_profile:
                        _maybe_cuda_sync(device, profile_cuda_sync)
                        t_loss_0 = time.perf_counter()
                    loss = masked_spectral_distance(batch[columns.label], pred)
                    if do_profile:
                        _maybe_cuda_sync(device, profile_cuda_sync)
                        loss_s = time.perf_counter() - t_loss_0

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_bwd_0 = time.perf_counter()
                grad_norm = None
                if amp_use_grad_scaler and scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        grad_norm = _clip_grad_norm(model, grad_clip)
                else:
                    loss.backward()
                    if grad_clip > 0:
                        grad_norm = _clip_grad_norm(model, grad_clip)
                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    backward_s = time.perf_counter() - t_bwd_0

                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    t_opt_0 = time.perf_counter()
                if amp_use_grad_scaler and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if do_profile:
                    _maybe_cuda_sync(device, profile_cuda_sync)
                    optim_s = time.perf_counter() - t_opt_0
                train_loss_total += loss.item()
                train_batches += 1
                train_it.set_postfix(loss=f"{loss.item():.4f}")
                
                # pred is the predicted mean when using the spectral angle based loss function
                # Create a y_true that is the batch[columns.label] with -1 values turned into 0
                y_true = batch[columns.label].detach().clone()
                present = y_true > 0
                y_true[~present] = 0
                batch_mae = torch.mean(torch.abs(torch.sub(pred, y_true))).item()
                # Mean adjusted spectral angle calculation (just 1 - loss value for standard prosit)
                batch_msa = 1 - loss.item()

                train_mae_total += batch_mae
                train_msa_total += batch_msa

                train_step += 1

                train_metrics = {
                    "train_step": train_step,
                    "train_batch_loss": loss.item(),
                    "train_batch_learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss_total": train_loss_total,
                    "current_epoch_average_train_loss": train_loss_total / max(1, train_batches),
                    "train_batch_mean_absolute_error": batch_mae,
                    "train_batch_mean_spectral_angle": batch_msa,
                }
                if grad_norm is not None:
                    train_metrics["train_batch_grad_norm"] = grad_norm
                global_step = _wandb_log(run, global_step, train_metrics)
                
                iter_end = time.perf_counter()
                if do_profile:
                    prof_totals["data_wait_s"] += data_wait_s
                    prof_totals["move_cast_s"] += move_cast_s
                    prof_totals["forward_s"] += forward_s
                    prof_totals["loss_s"] += loss_s
                    prof_totals["backward_s"] += backward_s
                    prof_totals["optim_s"] += optim_s
                    prof_totals["step_total_s"] += iter_end - iter_start
                    prof_count += 1

                    if profile_log_every > 0 and (prof_count % profile_log_every == 0):
                        print(_timing_summary("[profile][train]", prof_totals, prof_count))

                seen_train_batches += 1
                loop_end = iter_end
                if args.max_train_batches and train_batches >= args.max_train_batches:
                    break

            avg_train_loss = train_loss_total / max(1, train_batches)
            avg_train_mae = train_mae_total / max(1, train_batches)
            avg_train_sa = train_msa_total / max(1, train_batches)

            # Validation
            model.eval()
            val_loss_total = 0.0
            val_mean_absolute_error_total = 0.0
            val_spectral_angle_total = 0.0
            val_batches = 0
            with torch.no_grad():
                val_it = tqdm(
                    dataset.tensor_val_data,
                    total=val_steps_est,
                    desc=f"Epoch {epoch:03d} [val]",
                    unit="batch",
                    leave=False,
                )
                for batch in val_it:
                    batch = _move_batch_to_device(batch, device)
                    batch = _cast_batch_types(batch, columns)
                    with _amp_autocast_context(device, amp_enabled, amp_dtype):
                        pred = model(batch)
                        val_loss = masked_spectral_distance(batch[columns.label], pred)
                    val_loss_total += val_loss.item()
                    val_batches += 1

                    # pred is the predicted mean when using the spectral angle based loss function
                    y_true = batch[columns.label].detach().clone()
                    present = y_true > 0
                    y_true[~present] = 0
                    batch_mae = torch.mean(torch.abs(torch.sub(pred, y_true))).item()
                    # Mean adjusted spectral angle calculation (just 1 - loss value for standard prosit)
                    batch_msa = 1 - val_loss.item()

                    val_mean_absolute_error_total += batch_mae
                    val_spectral_angle_total += batch_msa

                    val_step += 1

                    global_step = _wandb_log(run, global_step, {
                        "val_step": val_step,
                        "val_batch_loss": val_loss.item(),
                        "val_loss_total": val_loss_total,
                        "current_epoch_average_val_loss": val_loss_total / max(1, val_batches),
                        "val_batch_mean_absolute_error": batch_mae,
                        "val_batch_mean_spectral_angle": batch_msa,
                    })

                    val_it.set_postfix(loss=f"{val_loss.item():.4f}")
                    if args.max_val_batches and val_batches >= args.max_val_batches:
                        break
            avg_val_loss = val_loss_total / max(1, val_batches)
            avg_val_mae = val_mean_absolute_error_total / max(1, val_batches)
            avg_val_sa = val_spectral_angle_total / max(1, val_batches)

            if select_best_by_val_loss:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    raw_model = _unwrap_model_for_state_dict(model)
                    best_state = {
                        k: v.detach().cpu() for k, v in raw_model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            print(
                f"Epoch {epoch:03d}: train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}"
            )
            if profile_enabled:
                print(
                    _timing_summary(
                        f"[profile][epoch={epoch:03d}]", prof_totals, prof_count
                    )
                )

            global_step = _wandb_log(run, global_step, {
                "epoch": epoch,
                "epoch_average_train_loss": avg_train_loss, 
                "epoch_average_train_mean_absolute_error": avg_train_mae,
                "epoch_average_train_mean_spectral_angle": avg_train_sa,
                "epoch_average_validation_loss": avg_val_loss,
                "epoch_average_validation_mean_absolute_error": avg_val_mae,
                "epoch_average_validation_mean_spectral_angle": avg_val_sa,
            })

            if args.checkpoint_save:
                torch.save(
                    {
                        "uncertainty_aware": args.uncertainty_aware,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    f"{args.checkpoint_save}/checkpoint_epoch_{epoch}.pth",
                )

            if clr_enabled and clr_every > 0 and (epoch % clr_every == 0):
                clr_max_lr *= clr_gamma

            if early_patience > 0 and epochs_without_improvement >= early_patience:
                print(
                    f"Early stopping: no val improvement for {epochs_without_improvement} epoch(s) (patience={early_patience})."
                )
                break

    if best_state is not None:
        raw_model = _unwrap_model_for_state_dict(model)
        raw_model.load_state_dict(best_state)
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            torch.save(best_state, args.save)
            print(f"Saved best model to: {args.save}")
    elif args.save:
        raw_model = _unwrap_model_for_state_dict(model)
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(raw_model.state_dict(), args.save)
        print(f"Saved final model to: {args.save}")

    # Test
    if args.uncertainty_aware == True:
        if test_path is not None:
            model.eval()
            test_loss_total = 0.0
            test_batches = 0
            with torch.no_grad():
                test_it = tqdm(
                    dataset.tensor_test_data,
                    total=test_steps_est,
                    desc="Test",
                    unit="batch",
                    leave=False,
                )
                for batch in test_it:
                    batch = _move_batch_to_device(batch, device)
                    batch = _cast_batch_types(batch, columns)
                    with _amp_autocast_context(device, amp_enabled, amp_dtype):
                        pred_log_mean, pred_log_var, pred_presence_logit = model(batch)
                        test_loss = gaussian_nll(batch[columns.label], pred_log_mean, pred_log_var, pred_presence_logit, batch[columns.sequence], args.bce_weight, args.nll_weight)
                    test_loss_total += test_loss.item()
                    test_batches += 1
                    test_it.set_postfix(loss=f"{test_loss.item():.4f}")
                    if args.max_test_batches and test_batches >= args.max_test_batches:
                        break
            avg_test_loss = test_loss_total / max(1, test_batches)
            print(f"Test loss: {avg_test_loss:.6f}")
    else:
        if test_path is not None:
            model.eval()
            test_loss_total = 0.0
            test_batches = 0
            with torch.no_grad():
                test_it = tqdm(
                    dataset.tensor_test_data,
                    total=test_steps_est,
                    desc="Test",
                    unit="batch",
                    leave=False,
                )
                for batch in test_it:
                    batch = _move_batch_to_device(batch, device)
                    batch = _cast_batch_types(batch, columns)
                    with _amp_autocast_context(device, amp_enabled, amp_dtype):
                        pred = model(batch)
                        test_loss = masked_spectral_distance(batch[columns.label], pred)
                    test_loss_total += test_loss.item()
                    test_batches += 1
                    test_it.set_postfix(loss=f"{test_loss.item():.4f}")
                    if args.max_test_batches and test_batches >= args.max_test_batches:
                        break
            avg_test_loss = test_loss_total / max(1, test_batches)
            print(f"Test loss: {avg_test_loss:.6f}")
        

    if profile_enabled:
        print(_timing_summary("[profile][final]", prof_totals, prof_count))

    run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
