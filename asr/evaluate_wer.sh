# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin, Valentin Vielzeuf

#!/bin/bash

split="dev"
step="ner_corrected"

for FILE in ./manifests/${split}_manifests/${step}/*.json
do
    echo "Scoring manifest file ${FILE}." >> ./manifests/${split}_manifests/${step}/wer_${split}_$step.txt
    python ./utils/wer.py ${FILE} >> ./manifests/${split}_manifests/${step}/wer_${split}_$step.txt    
done
