# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

#!/bin/bash

wav_path_prefix= "../data/" # Where to find the wav dialogue files e.g. "../data"
split="dev" # Choose between dev and test split
version="tts" # Choose from tts human or human_paraphrased (for test split only)

echo "Building manifest from wav files in ${wav_path_prefix}/DSTC11_${split}_${version}. Creating file at ./manifests/${split}_manifests/${split}_${version}.json"
python ./utils/txt2manifest.py --wav_path_prefix ${wav_path_prefix}/DSTC11_${split}_${version} --reference_txt ${wav_path_prefix}/${split}_manifest.txt --manifest_output ./manifests/${split}_manifests/${split}_${version}.json
