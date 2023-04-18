# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import os
import tqdm
import argparse
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from speechbrain.pretrained import EncoderDecoderASR
from utils.normalization import normalize_w2n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", '-i', type=str)
    parser.add_argument("--audio_folder", '-f', type=str)
    parser.add_argument("--output_file", '-o', type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dialogues = parse_txt(args.input_file)
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_model", run_opts={"device":"cuda"}) 
    dialogues = transcribe_and_replace(dialogues, args.audio_folder, asr_model)
    write_new_file(dialogues, args.output_file)


def transcribe_and_replace(dialogues, audio_folder, asr_model):
    c = 0
    for dial_id, data in tqdm.tqdm(dialogues.items()):
        audio_files = []
        for i in range(0, len(data['lines']), 2):
            audio_file = f'{audio_folder}/{dial_id}/user_{i // 2}.wav'
            audio_files.append(audio_file)

        sigs = []
        lens = []
        for audio_file in audio_files:
          snt, fs = torchaudio.load(audio_file)
          sigs.append(snt.squeeze())
          lens.append(snt.shape[1])
        batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
        lens = torch.Tensor(lens) / batch.shape[1]
        try:
            predicted_batch, token_ids = asr_model.transcribe_batch(batch, lens)

            for i, prediction in enumerate(predicted_batch):
                prediction = normalize_w2n(prediction.lower())
                dialogues[dial_id]['texts'][i * 2] = 'user: ' + prediction
        except:
            pass
    return dialogues


def parse_txt(file_path):
    """Parse reference txt file and store dialogues in a dict."""

    dialogues = {}
    with open(file_path, 'r') as istr:
        for line in istr.readlines():
            if 'END_OF_DIALOG' in line: # only in the train set, not in dev
                continue
            rest_of_line, state = line.split('state:')
            rest_of_line, text = rest_of_line.split('text: ')
            _, _, _, dialog_id, _, _ = rest_of_line.split(' ', 5)
            dialog_id = dialog_id.replace('.json', '')
            if dialog_id not in dialogues:
                dialogues[dialog_id] = {
                    'lines': [],
                    'texts': [],
                    'states': [],
                }
            dialogues[dialog_id]['lines'].append(rest_of_line.strip())
            dialogues[dialog_id]['texts'].append(text.strip())
            dialogues[dialog_id]['states'].append(state.strip())
    return dialogues


def write_new_file(dialogues, file_path):
    with open(file_path, 'w') as ostr:
        for dial_id, data in dialogues.items():
            for i in range(len(data['lines'])):
                new_line = f"{data['lines'][i]} text: {data['texts'][i]} state: {data['states'][i]}"
                print(new_line, file=ostr)
            print('END_OF_DIALOG', file=ostr)


if __name__=='__main__':
    main()
