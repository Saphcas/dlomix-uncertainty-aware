# scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_PTMs/PTMs_Train $TMPDIR

export HF_HOME=$TMPDIR/.hf
export HF_HUB_CACHE=$TMPDIR/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/.hf/transformers

export DATA_LOCATION=$TMPDIR/PTMs_Train
export WANDB_NAME=weight-test

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

export BCE_WEIGHT=0.5
export NLL_WEIGHT=1.5

PERSISTENT_DIR=/nobackup/proj/disk/kall/personal/$USER/checkpoints/

apptainer exec --bind $TMPDIR/ \
    /nobackup/proj/disk/kall/personal/$USER/containers/dlomix-weight.sif \
    python /opt/dlomix-src/run_scripts/train_prosit_intensity_ptms_torch.py


