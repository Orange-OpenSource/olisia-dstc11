# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

#!/bin/bash

split="dev"
step="raw"

for FILE in ./manifests/${split}_manifests/$step/*.json 
do
    echo "Normalizing predictions in ${FILE}. Outputing in ./manifests/${split}_manifests/time_normed/."
    python ./utils/normalize_time.py --manifest_input ${FILE} --output_folder ./manifests/${split}_manifests/time_normed/
    mv ./manifests/${split}_manifests/time_normed/$(basename "$FILE") ./manifests/${split}_manifests/time_normed/$(basename "$FILE" .json)_time_normed.json
done