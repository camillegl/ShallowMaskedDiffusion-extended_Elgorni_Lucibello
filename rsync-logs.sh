#!/bin/bash
# Usage: ./rsync_logs.sh

# REMOTE=deathstar
# REMOTE_DIR="~/Git/ShallowMaskedDiffusion/logs"
# LOCAL_DIR="."


REMOTE_HOST="lnode02"
REMOTE_DIR="~/Git/ShallowMaskedDiffusion/logs"
LOCAL_DIR="."

# rsync -avzP ${REMOTE_HOST}:${REMOTE_DIR} ${LOCAL_DIR}
# rsync -avzP --exclude='last.ckpt' ${REMOTE_HOST}:${REMOTE_DIR} ${LOCAL_DIR}
rsync -avzP --exclude='*tensor8*' ${REMOTE_HOST}:${REMOTE_DIR} ${LOCAL_DIR}
