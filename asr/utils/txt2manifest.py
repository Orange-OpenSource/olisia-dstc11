# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import scipy.io.wavfile as wav
import json
import os
import argparse

def build_manifest(wav_path_prefix: str, manifest_path: str, reference_path: str):
    """
    The in and out data streams are handled with manifest files which contain the paths to the wav files, their respective duration and reference transcription. 
    When we predict, it updates the manifest file with a field for predicted transcription.

    Arguments :
        - wav_path_prefix : The path where we can find all the wav files regrouped by dialogue ID folders.
        - manifest_path : The path where the manifest will be written.
        - reference_path : The path where we can find the reference text transcription. We consider it to follow the dstc11 format i.e. 
                    each line corresponds to a dialogue turn formated as "line_nr: [X] dialog_id: [Y] turn_id: [Z] text: [A] state: [B]"
    """
    with open(manifest_path, "w") as manifest, open(reference_path, "r") as reference:
        for line in reference:
            # The file is formated as line_nr: [X] dialog_id: [Y] turn_id: [Z] text: [A] state: [B]
            # We limit the number of splits to 7 to not split the text
            splits = line.split(' ', 7)
            if splits[0] != 'END_OF_DIALOG\n':
                turn = int(splits[5])
                if turn % 2 == 1:
                    # The wav path is completed with the dialogue folder and the Turn id
                    file_path = os.path.join(wav_path_prefix, splits[3].split(".json")[0] + "/Turn-{}.wav".format(turn))
                    source_rate, source_sig = wav.read(file_path)
                    duration = len(source_sig) / float(source_rate)
                    # removing the user state and removing the 5 first characters which are "user: "
                    text = splits[-1].split(" state:")[0][6:]
                    manifest_line = {"audio_filepath": "{}".format(file_path), "duration": duration, "text": text}
                    manifest.write(json.dumps(manifest_line) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path_prefix", type=str, help="The path prefix where we can find the wav files of each user turn sorted by dialogue.")
    parser.add_argument("--reference_txt", type=str, help="The path towards the reference txt file which will be used as reference text for the manifest.")
    parser.add_argument('--manifest_output', type=str, help="The path where to store the outputed manifest.")
    args = parser.parse_args()

    build_manifest(args.wav_path_prefix, args.manifest_output, args.reference_txt)

if __name__=="__main__":
    main()