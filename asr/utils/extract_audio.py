# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import os
from glob import iglob
import h5py
import scipy.io.wavfile
import tqdm
import argparse

def pcm2wav(filepath):
    """
    Given a filepath to a h5p file (.hd5) we read the pcm audio of each turn and convert it to wav format to hear it.
    """
    data = h5py.File(filepath, 'r')
    for group in list(data.keys()):
        # For each group (=turn?) we select the pcm audio and convert it to wav
        audio_pcm = data[group]['audio'][:]
        # FIXME: the turn id's index is strangely handled, seems to be an offset of one except on the first turn
        turn = group.split(' ')[-1]
        dialogue = group.split(' ')[-3].split('.')[0]
        if int(turn) > 1:
            turn = str(int(turn) - 1)
        #print("Converting " + group + " : " + data[group].attrs['hyp'])
        newPath = os.path.join(os.path.dirname(filepath), dialogue, "Turn-" + turn + ".wav")
        if not os.path.isdir(os.path.dirname(newPath)):
            os.mkdir(os.path.dirname(newPath))
        scipy.io.wavfile.write(newPath, 16000, audio_pcm)


def getFullDialogueTranscript(filepath: str, reference_txt_path: str):
    """
    Given a filepath to a h5p file (.hd5) we read the transcript and link it to the system turns to get the full dialogue transcript
    """
    dialogue = filepath.split('/')[-1].split('.')[0] + '.json'
    transcript = ""
    with open(reference_txt_path, 'r') as fullDialogues:
        for line in fullDialogues:
            # The file is formated as line_nr: X dialog_id: Y turn_id: Z text:
            # We limit the number of splits to 7 to not split the text
            splits = line.split(' ', 7)
            if splits[0] != 'END_OF_DIALOG\n':
                if splits[3] == dialogue:
                    transcript += splits[-1]
    
    # write the transcript in a txt file
    newPath = os.path.join(os.path.dirname(filepath), dialogue.split(".")[0], dialogue + ".txt")
    if not os.path.isdir(os.path.dirname(newPath)):
        os.mkdir(os.path.dirname(newPath))
    with open(newPath, "w") as dialTranscript:
        dialTranscript.write(transcript)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_txt_path", required=True, type=str, help="The path to the reference txt file for which we want to extract the agent transcriptions.")
    parser.add_argument("--data_path", required=True, type=str, help="The path where to find the hd5 files which contain the audio.")
    args = parser.parse_args()
    
    for filePath in tqdm.tqdm(iglob(args.data_path + "/*.hd5")):
        pcm2wav(filePath)
        getFullDialogueTranscript(filePath, reference_txt_path=args.reference_txt_path)

main()