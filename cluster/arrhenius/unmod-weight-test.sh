# scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_unmod/intensity $TMPDIR

export HF_HOME=$TMPDIR/.hf
export HF_HUB_CACHE=$TMPDIR/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/.hf/transformers

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

export BCE_WEIGHT=0.2
export NLL_WEIGHT=1
export SEEDED_RUN=True
export SEED=15

export TRAIN_LOCATION=$TMPDIR/intensity/unmod_train.parquet
export VAL_LOCATION=$TMPDIR/intensity/unmod_val.parquet
export TEST_LOCATION=$TMPDIR/intensity/unmod_test.parquet

export WANDB_NAME="$SLURM_JOB_NAME"_"$SLURM_JOB_ID"_test

PERSISTENT_DIR=/nobackup/proj/disk/kall/personal/$USER/checkpoints/unmod_weighting/

apptainer exec --bind $TMPDIR/ \
    /nobackup/proj/disk/kall/personal/$USER/containers/unmod-parameter-sweep.sif \
    python /opt/dlomix-src/run_scripts/train_prosit_intensity_ptms_torch.py

