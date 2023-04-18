# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

DEVICES="0"
TRAIN_FILE=""
DEV_FILE=""
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=256
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
LEARNING_RATE=5e-4
NUM_EPOCHS=5
MODEL="t5-small"
OUTPUT_DIR="checkpoints/t5-small-dstc11"

CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
    --model_name_or_path $MODEL \
    --train_txt_file $TRAIN_FILE \
    --dev_txt_file $DEV_FILE \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \

CUDA_VISIBLE_DEVICES=$DEVICES python evaluate.py \
    --model_name_or_path $OUTPUT_DIR \
    --dev_txt_file $DEV_FILE \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $OUTPUT_DIR\

