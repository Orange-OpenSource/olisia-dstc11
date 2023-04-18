# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

#!/bin/bash

data_path="../data/DSTC11_dev_tts"
reference_txt_path="../data/dev_manifest.txt"

echo "Extracting audio into ${data_path} with the reference text file ${reference_txt_path}."
python ./utils/extract_audio.py --data_path ${data_path} --reference_txt_path ${reference_txt_path}

