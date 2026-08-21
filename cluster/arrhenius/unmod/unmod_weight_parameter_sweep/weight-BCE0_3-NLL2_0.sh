#!/bin/bash
#SBATCH -A naiss2026-3-479-gpu -p gpu --gpus=1
#SBATCH -t 42:00:00
#SBATCH --signal=B:TERM@120
#SBATCH -J dlomix-unmod-weight
#SBATCH -o /nobackup/proj/disk/kall/personal/%u/logs/%x-%j.out

set -Eeuo pipefail

#set -u
#set -o pipefail

scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_unmod/intensity $TMPDIR

export HF_HOME=$TMPDIR/.hf
export HF_HUB_CACHE=$TMPDIR/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/.hf/transformers

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

export BCE_WEIGHT=0.3
export NLL_WEIGHT=2
export SEEDED_RUN=True
export SEED=15

export TRAIN_LOCATION=$TMPDIR/intensity/unmod_train.parquet
export VAL_LOCATION=$TMPDIR/intensity/unmod_val.parquet
export TEST_LOCATION=$TMPDIR/intensity/unmod_test.parquet

export CHECKPOINT_DIR=$TMPDIR/"$SLURM_JOB_NAME"_"$SLURM_JOB_ID"_BCE-0-3_NLL-2-0

export WANDB_NAME="$SLURM_JOB_NAME"_"$SLURM_JOB_ID"_BCE"$BCE_WEIGHT"_NLL"$NLL_WEIGHT"

mkdir -p $CHECKPOINT_DIR

PERSISTENT_DIR=/nobackup/proj/disk/kall/personal/$USER/checkpoints/unmod_weighting/

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


scp -r $CHECKPOINT_DIR /nobackup/proj/disk/kall/personal/$USER/checkpoints/unmod_weighting/
