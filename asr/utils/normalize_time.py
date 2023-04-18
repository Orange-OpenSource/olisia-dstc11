# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import json
import os
import re
import argparse

def process_time(transcription: str):
    """
    This function returns a string with time corrections. It should be applied after normalization since it expects transcriptions to contain time in figures.

    Arguments:
        - transcription: a normalized transcription.

    Returns:
        - corrected_transcriptions: transcription with corrected time.
    """

    # For Whisper there are several separators: o/or, to, space, dot, coma, 

    def _process_o_time(match):
        return match.group(1) + ":0" + match.group(3) + " " + match.group(4).replace(".", "").replace(" ", "").lower() + " "
    o_pattern = r'(\d|\d\d) (or|o) (\d)( a\.m\.| p\.m\.| am| pm|a\.m\.|p\.m\.|am|pm| AM|AM|PM| PM|a\.m| a\.m| p\.m|p\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(o_pattern, _process_o_time, transcription)

    def _process_to_time(match):
        return match.group(1) + ":2" + match.group(3) + " " + match.group(4).replace(".", "").replace(" ", "").lower() + " "
    to_pattern = r'(\d|\d\d) (to) (\d)( a\.m\.| p\.m\.| am| pm|a\.m\.|p\.m\.|am|pm| AM|AM|PM| PM|a\.m| a\.m| p\.m|p\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(to_pattern, _process_to_time, corrected_string)

    def _process_space_time(match):
        return match.group(1) + ":" + match.group(2).replace(" ", "") + " " + match.group(3).replace(".", "").replace(" ", "").lower() + " "
    space_pattern = r'(\d|\d\d) (\d\d|\d \d)( a\.m\.| p\.m\.| am| pm|a\.m\.|p\.m\.|am|pm| AM|AM|PM| PM|a\.m| a\.m| p\.m|p\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(space_pattern, _process_space_time, corrected_string)

    def _process_coma_time(match):
        return match.group(1) + ":" + match.group(2).replace(", ", "") + " " + match.group(3).replace(".", "").replace(" ", "").lower() + " "
    coma_pattern = r'(\d|\d\d), (\d|\d, \d|\d\d)( a\.m\.| p\.m\.| am| pm|a\.m\.|p\.m\.|am|pm| AM|AM|PM| PM|a\.m| a\.m| p\.m|p\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(coma_pattern, _process_coma_time, corrected_string)

    def _process_dot_time(match):
        return match.group(1) + ":" + match.group(2).replace(".", "") + " " + match.group(3).replace(".", "").replace(" ", "").lower() + " "
    dot_pattern = r'(\d|\d\d)\.(\d|\d\.\d|\d\d)( a\.m\.| p\.m\.| am| pm|a\.m\.|p\.m\.|am|pm| AM|AM|PM| PM|a\.m| a\.m| p\.m|p\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(dot_pattern, _process_dot_time, corrected_string)

    def _process_only_digit(match):
        if len(match.group(1)) == 4:
            hours = match.group(1)[:2]
            minutes = match.group(1)[2:]
            time = hours + ":" + minutes
        elif len(match.group(1)) == 3:
            hours = match.group(1)[0]
            minutes = match.group(1)[1:]
            time = hours + ":" + minutes
        elif len(match.group(1)) == 2 and int(match.group(1)) > 12:
            hours = "0"
            minutes = match.group(1)
            time = hours + ":" + minutes
        elif len(match.group(1)) == 1:
            if match.group(2).replace(".", "").lower() == "am":
                hours = "0"
                minutes = "0" + match.group(1)
                time = hours + ":" + minutes
            else:
                hours = match.group(1)
                time = hours
        else:
            time = match.group(1)
        return " " + time + " " + match.group(2).replace(".", "").replace(" ", "").lower() + " "
    pattern = r' (\d+)( a\.m\.| p\.m\.| am| pm|a\.m\.|p\.m\.|am|pm| AM|AM|PM| PM|a\.m| a\.m| p\.m|p\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(pattern, _process_only_digit, corrected_string)

    hours_pattern = r'(\d\d)(\d\d) hours'
    corrected_string = re.sub(hours_pattern, lambda m: m.group(1) + ":" + m.group(2) + " " + "hrs", corrected_string)

    # Finally convert the 12 am into 0 am
    am_pattern = r'12:(\d\d)(a\.m\.|am| am| a\.m\.| AM|AM|a\.m| a\.m| a m(?![a-z])| p m(?![a-z])| aem| pem)'
    corrected_string = re.sub(am_pattern, lambda m: "0:" + m.group(1) + " " + m.group(2).replace(".", "").replace(" ", "").lower() + " ", corrected_string)

    midnight_pattern = r'midnight(a\.m\.|am| am| a\.m\.| AM|AM|a\.m| a\.m| a m(?![a-z])| p m(?![a-z])| aem| pem|)'
    corrected_string = re.sub(midnight_pattern, lambda m: "0:00 am", corrected_string)

    def _process_past_time(match):
        if match.group(1) == 'half':
            hours = match.group(4)
            minutes = '30'
        elif match.group(1) == 'quarter':
            hours = match.group(4)
            minutes = '15'
        elif match.group(2) == " minutes":
            hours = match.group(4)
            minutes = match.group(1)
        else:
            if int(match.group(4)) > 12:
                hours = match.group(1)
                minutes = match.group(4)
            elif len(match.group(1)) > 1:
                hours = match.group(4)
                minutes = match.group(1)
            else:
                hours = match.group(4)
                minutes = "0" + match.group(1)
        return hours + ":" + minutes + " " + match.group(5)
    past_pattern = r'(\d+|half|quarter)( minutes|) past (0:|)(\d+) (am|pm)'
    corrected_string = re.sub(past_pattern, _process_past_time, corrected_string)

    return corrected_string

def normalize_preds(manifest_path_in: str, output_path: str):
    """
    This function attempts to perform a time normalization of the predictions made by the Whisper ASR system and outputs a new manifest.

    Arguments:
        - manifest_path_in: the path to the manifest we want to normalize
        - output_path: the path where to save the manifest.
    """
    with open(manifest_path_in, "r") as pred_manifest, open(output_path, "w") as out_manifest:
        for line in pred_manifest:
            data = json.loads(line)
            data["pred_text"] = process_time(data["pred_text"])
            out_manifest.write(json.dumps(data) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_input", type=str, help="The path where to find the manifest for which the predictions should be normalized.")
    parser.add_argument("--output_folder", type=str, help="The folder in which we should store the normalized manifest.")
    args = parser.parse_args()

    output_manifest_path = os.path.join(args.output_folder, os.path.basename(args.manifest_input))
    normalize_preds(args.manifest_input, output_manifest_path)

if __name__=="__main__":
    main()