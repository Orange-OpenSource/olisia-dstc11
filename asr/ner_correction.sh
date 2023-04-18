# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin, Valentin Vielzeuf

#!/bin/bash

split="dev"
step="time_normed"
data_path="../data"

for FILE in ./manifests/${split}_manifests/$step/*.json 
do
    echo "Correcting Named-Entities in ${FILE}. Outputing in ./manifests/${split}_manifests/ner_corrected/."
    if [[ $FILE == *"tts"* ]]; then
        python ./utils/ner.py --folder $data_path/DSTC11_${split}_tts/ --pred_file ${FILE} 
    else
        python ./utils/ner.py --folder $data_path/DSTC11_${split}_human/ --pred_file ${FILE}
    fi
    mv ./manifests/${split}_manifests/${step}/$(basename "$FILE" .json)_ner_0.4.json ./manifests/${split}_manifests/ner_corrected/$(basename "$FILE" .json)_ner_corrected.json
done