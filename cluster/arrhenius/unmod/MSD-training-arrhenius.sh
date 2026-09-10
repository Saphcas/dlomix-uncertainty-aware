#!/bin/bash
#SBATCH -A naiss2026-3-479-gpu 
#SBATCH -p gpu 
#SBATCH --gpus=1
#SBATCH -t 42:00:00
#SBATCH --signal=B:TERM@120

#SBATCH -J dlomix-unmod-msd-train

#SBATCH -o /nobackup/proj/disk/kall/personal/%u/logs/%x-%j.out

set -Eeuo pipefail

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
export UNCERTAINTY_AWARE=False

export BCE_WEIGHT=1
export NLL_WEIGHT=1

export SEEDED_RUN=False
#export SEED=15

# ----- File locations -----

export TRAIN_LOCATION="$TMPDIR/intensity/unmod_train.parquet"
export VAL_LOCATION="$TMPDIR/intensity/unmod_val.parquet"
export TEST_LOCATION="$TMPDIR/intensity/unmod_test.parquet"

# ----- Output variables -----

export CHECKPOINT_DIR="${TMPDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_checkpoints"

export WANDB_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

PERSISTENT_DIR=/nobackup/proj/disk/kall/personal/$USER/checkpoints/

mkdir -p "$CHECKPOINT_DIR"
#mkdir -p "$PERSISTENT_DIR"

# ----- Checkpoint persistance -----

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
    "/nobackup/proj/disk/kall/personal/${USER}/containers/dlomix-ngc-26.06.sif" \
    python /opt/dlomix-src/run_scripts/train_prosit_intensity_ptms_torch.py

# ----- In case sync_back fails -----

scp -r $CHECKPOINT_DIR /nobackup/proj/disk/kall/personal/$USER/checkpoints/
