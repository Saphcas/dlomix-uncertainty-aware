#!/bin/bash
#SBATCH -A naiss2026-3-479-gpu -p gpu --gpus=1
#SBATCH -t 42:00:00
#SBATCH --signal=B:TERM@120
#SBATCH -J dlomix-ptm-ua-warmup-cosine-train
#SBATCH -o /nobackup/proj/disk/kall/personal/%u/logs/%x-%j.out

set -Eeuo pipefail

#set -u
#set -o pipefail

scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_PTMs/PTMs_Train $TMPDIR

export HF_HOME=$TMPDIR/.hf
export HF_HUB_CACHE=$TMPDIR/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/.hf/transformers

export LR_SCHEDULE=warmup_cosine

export WARMUP_COSINE_START_LR=1.6e-5
export WARMUP_COSINE_PEAK_LR=1.2e-4
export WARMUP_COSINE_MIN_LR=1.6e-5

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

export TRAIN_LOCATION=$TMPDIR/PTMs_Train/all_train_ptms_fixed_na.parquet
export VAL_LOCATION=$TMPDIR/PTMs_Train/all_val_ptms_fixed_na.parquet
export TEST_LOCATION=$TMPDIR/PTMs_Train/test.parquet

export CHECKPOINT_DIR=$TMPDIR/"$SLURM_JOB_NAME"_"$SLURM_JOB_ID"_checkpoints
export WANDB_NAME="$SLURM_JOB_NAME"_"$SLURM_JOB_ID"

mkdir -p $CHECKPOINT_DIR

PERSISTENT_DIR=/nobackup/proj/disk/kall/personal/$USER/checkpoints/

sync_back() {
    echo "[sync] $(date) copying $CHECKPOINT_DIR -> $PERSISTENT_DIR"
    scp -r $CHECKPOINT_DIR $PERSISTENT_DIR
}

cleanup() {
    status=$?
    echo "[cleanup] exit status: $status"
    sync_back || echo "[cleanup] WARNING: scp failed"
    exit "$status"
}

on_term() {
    echo "[signal] caught SIGTERM, syncing before exit"
    sync_back || true
    exit 143
}

trap cleanup EXIT
trap on_term TERM
trap 'echo "[signal] caught SIGINT"; exit 130' INT

apptainer exec --bind $TMPDIR/ \
    /nobackup/proj/disk/kall/personal/$USER/containers/dlomix-ngc-26.06.sif \
    python /opt/dlomix-src/run_scripts/train_prosit_intensity_ptms_torch.py

# scp -r $CHECKPOINT_DIR /nobackup/proj/disk/kall/personal/$USER/checkpoints/
