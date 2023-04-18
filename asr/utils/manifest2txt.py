# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import json
import os
import argparse
from tqdm import tqdm

def manifest_to_txt(manifest_path: str, output_path: str, reference_path: str):
    """
    Converts the manifest which contains the predictions of an ASR model to the reference dstc11 text format.

    Arguments:
        - manifest_path: The name of the manifest file which contains the predicted transcriptions.
        - output_path: The path where to store the outputted predictions in the reference txt format.
        - reference_path: The path to the reference to extract the values which are not present in the manifest.
    """
    dialogues_preds = {}
    with open(manifest_path, "r") as manifest:
        for line in manifest:
            data = json.loads(line)
            dialogue_id = os.path.dirname(data['audio_filepath']).split('/')[-1] + '.json'
            turn_id = os.path.basename(data['audio_filepath']).split('.')[0].split('-')[1]
            dialogues_preds[dialogue_id + "_" + turn_id] = data["pred_text"]
    
    with open(output_path, "w") as out_txt, open(reference_path, "r") as in_txt:
        for line in tqdm(in_txt):
            splits = line.split(' ', 7)
            if splits[0] != 'END_OF_DIALOG\n':
                turn_id = int(splits[5])
                dialogue_id = splits[3]
                if turn_id % 2 == 1:
                    text = splits[-1]
                    if text.__contains__("state:"):
                        state = text.split("state:")[-1]
                    else:
                        state = "\n"
                    new_text = "user: " + dialogues_preds[dialogue_id + "_" + str(turn_id)].strip() + " state:" + state
                    out_txt.write(" ".join(splits[:7] + [new_text]))
                else:
                    out_txt.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_txt_path", type=str, help="The path to the reference txt file for which we want to replace the user transcriptions.")
    parser.add_argument("--manifest_path", type=str, help="The path where to find the manifest which contains the user turns transcriptions.")
    args = parser.parse_args()

    output_path = ".".join(args.manifest_path.split(".")[:-1]) + ".txt"
    manifest_to_txt(args.manifest_path, output_path, args.reference_txt_path)

if __name__=="__main__":
    main()
