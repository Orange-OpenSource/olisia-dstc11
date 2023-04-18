# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

#!/bin/bash

DEVICES="0"
MODEL_SIZE="large"
split="dev"
step="raw"

for MANIFEST in ./manifests/${split}_manifests/*.json
do
    OUT_PATH=$(dirname "$MANIFEST")/$step/$(basename "$MANIFEST" .json)_Whisper_$MODEL_SIZE.json
    echo "Transcribing files in ${MANIFEST} with Whisper ${MODEL_SIZE}. Outputing in ${OUT_PATH}."
    CUDA_VISIBLE_DEVICES=$DEVICES python ./utils/transcribe.py --manifest_input ${MANIFEST} --manifest_output ${OUT_PATH} --model_size $MODEL_SIZE
done
