#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
GPUS=$2
OUTPUT_DIR=$3
PORT=${PORT:-29000}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir $OUTPUT_DIR --launcher pytorch ${@:3}
