# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

#!/usr/bin/env python3

import whisper
import argparse
import json
from tqdm import tqdm

def predict_transcription_whisper(manifest_path: str, output_path: str, model_size: str):
    """
    Performs the prediction of the transcriptions of the wav files contained in the manifest with OpenAI's whisper asr model.

    Arguments :
        - manifest_path: The path to the manifest file.
        - output_path: The path where to store the output manifest which contains the computed predictions.
        - model_size: the size of Whisper model to consider.
    """

    # load model and processor
    model = whisper.load_model(model_size)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    # writing the output manifest
    with open(output_path, "w", encoding="utf-8") as out_manifest:
        with open(manifest_path, 'r') as in_manifest:
                for line in tqdm(in_manifest):
                    item = json.loads(line)
                    audio_file = item['audio_filepath']
                    whisper.load_audio(audio_file)
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)
                    item['pred_text'] = whisper.decode(model, mel, options).text
                    out_manifest.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_input", type=str, help="The path where to find the manifest to use for the transcriptions.")
    parser.add_argument("--model_size", required=False, default="large", type=str, help="The size of the Whisper model to use to perform the transcription (tiny, small, medium, large).")
    parser.add_argument("--manifest_output", type=str, default=None, help="The path where to write the manifest with the transcriptions.")
    args = parser.parse_args()

    predict_transcription_whisper(manifest_path=args.manifest_input, output_path=args.manifest_output, model_size=args.model_size)

if __name__=="__main__":
    main()
