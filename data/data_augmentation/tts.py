# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import torch
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained.interfaces import HIFIGAN
from speechbrain.utils.text_to_sequence import text_to_sequence

import os
import tqdm
import time
import argparse
from utils.normalization import normalize_n2w


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", '-i', type=str)
    parser.add_argument("--folder", '-o', type=str, help='where to store the produced wav files')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_models/tmpdir_tts", run_opts={"device":"cuda"})
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tmpdir_vocoder", run_opts={"device":"cuda"})
    dialogues = parse_txt(args.input_file)
    create_audio_files(dialogues, args.folder, tacotron2, hifi_gan)


def create_audio_files(dialogues, folder, tacotron2, hifi_gan):
    for dial_id, data in tqdm.tqdm(dialogues.items()):
        path = f'{folder}/{dial_id}'
        os.makedirs(path, exist_ok=True)
        usr_utterances = [normalize_n2w(data['texts'][i].replace('user: ', '')) for i in range(0, len(data['lines']), 2)]
        waveforms = tts(usr_utterances, tacotron2, hifi_gan)
        for i in range(len(waveforms)):
            torchaudio.save(f'{path}/user_{i}.wav', waveforms[i].squeeze(1).cpu(), 22050)


def tts(utterances, tacotron2, hifi_gan):
    # tacotron2 excepts a list sorted in decreasing order
    utterances = [(utterance, text_to_sequence(utterance, tacotron2.text_cleaners)) for utterance in utterances]
    orig_pos, utterances = zip(*sorted(enumerate(utterances), key=lambda x: len(x[1][1]), reverse=True))
    utterances = [utterance for utterance, sequence in utterances]
    mel_outputs_postnet, mel_lengths, alignments = tacotron2.encode_batch(utterances)
    mel_outputs_no_padding = []
    for i in range(mel_outputs_postnet.shape[0]):
        mel_outputs_no_padding.append(mel_outputs_postnet[i][:, :mel_lengths[i].item() + 1])
    mel_outputs_postnet = [mel_output for _, mel_output in sorted(zip(orig_pos, mel_outputs_no_padding), key=lambda pair: pair[0])]
    waveforms = []
    for i in range(len(mel_outputs_postnet)):
        waveforms.append(hifi_gan.decode_spectrogram(mel_outputs_postnet[i]))
    return waveforms


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

if __name__=='__main__':
    main()
