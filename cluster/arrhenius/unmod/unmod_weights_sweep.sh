#!/bin/bash

#SBATCH -A naiss2026-3-479-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 42:00:00
#SBATCH --signal=B:TERM@120

#SBATCH -J dlomix-unmod-weight
#SBATCH --array=0-41

#SBATCH -o /nobackup/proj/disk/kall/personal/%u/logs/%x-%A_%a.out

set -Eeuo pipefail

# ----- Parameter sweep configurations -----

# Will be a 7x6 matix (42 runs (up to array index 41))
BCE_WEIGHTS=(0.2 0.25 0.3 0.35 0.4 0.45 0.5)
NLL_WEIGHTS=(1 1.2 1.4 1.6 1.8 2)

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

scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_unmod/intensity $TMPDIR

# ----- HuggingFace variables -----

export HF_HOME="$TMPDIR/.hf"
export HF_HUB_CACHE="$TMPDIR/.hf/hub"
export HF_DATASETS_CACHE="$TMPDIR/.hf/datasets"
export TRANSFORMERS_CACHE="$TMPDIR/.hf/transformers"

# ----- Training configuration -----

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120

export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

export SEEDED_RUN=True
export SEED=15

# ----- File locations -----

export TRAIN_LOCATION="$TMPDIR/intensity/unmod_train.parquet"
export VAL_LOCATION="$TMPDIR/intensity/unmod_val.parquet"
export TEST_LOCATION="$TMPDIR/intensity/unmod_test.parquet"

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