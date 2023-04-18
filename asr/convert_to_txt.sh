# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

#!/bin/bash

step="time_normed"
split="dev"
data_path="../data"

for FILE in ./manifests/${split}_manifests/$step/*.json
do
    echo "Building text file for ${FILE}."
    if [ "$split" = "dev" ]; then
        python ./utils/manifest2txt.py --reference_txt_path $data_path/dev_manifest.txt --manifest_path ${FILE} 
    else
        python ./utils/manifest2txt.py --reference_txt_path $data_path/test_manifest.txt --manifest_path ${FILE}
    fi
done
