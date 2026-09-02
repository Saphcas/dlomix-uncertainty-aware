#!/bin/bash

#SBATCH -A naiss2026-3-479-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 42:00:00
#SBATCH --signal=B:TERM@120

#SBATCH -J dlomix-unmod-weight
#SBATCH --array=0-24

#SBATCH -o /nobackup/proj/disk/kall/personal/%u/logs/%x-%A_%a.out

set -Eeuo pipefail

# ----- Parameter sweep configurations -----

# Will be a 5x5 matix (25 runs (up to array index 24))
BCE_WEIGHTS=(0.25 0.5 1 1.5 2)
NLL_WEIGHTS=(0.25 0.5 1 1.5 2)

NUM_BCE=${#BCE_WEIGHTS[@]}
NUM_NLL=${#NLL_WEIGHTS[@]}

EXPECTED_JOBS=$((NUM_BCE * NUM_NLL))

if (( SLURM_ARRAY_TASK_ID >= EXPECTED_JOBS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside parameter grid."
    exit 1
fi


# Convert the 1D Slurm array index into a 2D parameter index.
BCE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_NLL))
NLL_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_NLL))

export BCE_WEIGHT="${BCE_WEIGHTS[$BCE_INDEX]}"
export NLL_WEIGHT="${NLL_WEIGHTS[$NLL_INDEX]}"

# ----- Dataset -----

scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_PTMs/PTMs_Train $TMPDIR

# ----- HuggingFace variables -----

export HF_HOME="$TMPDIR/.hf"
export HF_HUB_CACHE="$TMPDIR/.hf/hub"
export HF_DATASETS_CACHE="$TMPDIR/.hf/datasets"
export TRANSFORMERS_CACHE="$TMPDIR/.hf/transformers"

# ----- Training configuration -----

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=60

export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

export SEEDED_RUN=True
export SEED=15

# ----- File locations -----

export TRAIN_LOCATION="$TMPDIR/PTMs_Train/all_train_ptms_fixed_na.parquet"
export VAL_LOCATION="$TMPDIR/PTMs_Train/all_val_ptms_fixed_na.parquet"
export TEST_LOCATION="$TMPDIR/PTMs_Train/test.parquet"

# ----- Output variables -----

export CHECKPOINT_DIR="${TMPDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_BCE-${BCE_WEIGHT}_NLL-${NLL_WEIGHT}"

export WANDB_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_BCE${BCE_WEIGHT}_NLL${NLL_WEIGHT}"

PERSISTENT_DIR="/nobackup/proj/disk/kall/personal/${USER}/checkpoints/unmod_weighting"

mkdir -p "$CHECKPOINT_DIR"
#mkdir -p "$PERSISTENT_DIR"

# ----- Checkpoint persistence -----

sync_back()
{
    echo "[sync] $(date) copying:"
    echo "       $CHECKPOINT_DIR"
    echo "    -> $PERSISTENT_DIR"

    scp -r "$CHECKPOINT_DIR" "$PERSISTENT_DIR"
}


cleanup()
{
    status=$?

    echo "[cleanup] exit status: $status"

    sync_back || echo "[cleanup] WARNING: checkpoint sync failed"

    exit "$status"
}


on_term()
{
    echo "[signal] caught SIGTERM"
    exit 143
}


on_int()
{
    echo "[signal] caught SIGINT"
    exit 130
}


trap cleanup EXIT
trap on_term TERM
trap on_int INT

# ----- Training -----

apptainer exec --bind "$TMPDIR/" \
    "/nobackup/proj/disk/kall/personal/${USER}/containers/unmod-parameter-sweep.sif" \
    python /opt/dlomix-src/run_scripts/train_prosit_intensity_ptms_torch.py