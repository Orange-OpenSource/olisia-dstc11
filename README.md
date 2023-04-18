# DSTC11

A repository for our participation to the DSTC11 track 3 challenge "[Dialog Systems Technology Challenge](https://dstc11.dstc.community/)" associated with the paper ["OLISIA: a Cascade System for Spoken Dialogue State Tracking"]().


# Quick start

## Data download

 The data is available on their [final website](https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html). It consists of audio files for the user turns and mappings to integrate them with the written agent turns in order to form a full dialogue:

- Training data (TTS):
    - [train.tts-verbatim.2022-07-27.zip](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.zip) contains 4 subdirectories, one for each TTS speaker (tpa, tpb, tpc, tpd), and each subdirectories contains all the 8434 dialogs corresponding to the original training set. The TTS outputs were generated using speakers that are available via Google Cloud Speech API.
    - [train.tts-verbatim.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.txt) contains the original dialog training data, which is used to generate the TTS outputs.
- Training data (Augmentation):
    - [train.100x.text.2022-10-17.txt](https://storage.googleapis.com/gresearch/dstc11/train.100x.text.2022-10-17.txt) contains the transcription of the organizers' Automatic Speech Recognition system on the synthetic audio they generated (not available).
    - [train.100x.text.2022-10-17.txt](https://storage.googleapis.com/gresearch/dstc11/train.100x.text.2022-10-17.txt) contains the mapping from the transcription of the organizers' Automatic Speech Recognition system to the augmented dialogues.
- Dev data (TTS):
    - [dev-dstc11.tts-verbatim.2022-07-27.zip](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.tts-verbatim.2022-07-27.zip) contains all the 1000 dialogs corresponding to TTS output from a held-out speaker.
    - [dev-dstc11.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-07-27.txt) contains the mapping from user utterances back to the original dialog.
- Dev data (Human):
    - [dev-dstc11.human-verbatim.2022-09-29.zip](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.human-verbatim.2022-09-29.zip) contains all the 1000 dialogs turns spoken by crowd workers.
- Test data (TTS):
    - [test-dstc11-tts-verbatim.2022-09-21.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11-tts-verbatim.2022-09-21.zip) contains all the 1000 dialogs corresponding to TTS output from a held-out speaker.
    - [test-dstc11.2022-09-21.txt](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.2022-09-21.txt) contains the mapping from user utterances back to the original dialog.
- Test data (Human):
    - [test-dstc11.human-verbatim.2022-09-29.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.human-verbatim.2022-09-29.zip) contains all the 1000 dialogs turns spoken by crowd workers.
- Test data (Human paraphrased):
    - [test-dstc11.human-paraphrased.2022-10-17.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.human-paraphrased.2022-10-17.zip) contains all the 1000 dialogs turns paraphrased by crowd workers.
- Test data DST annotations:
    - [test-dstc11.2022-1102.gold.json](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-1102.gold.json) contains the gold DST annotations for the test set.

## Installation

Using a conda environment (recommended):

```
pip install -r requirements.txt
```

## Usage

The code base is divided into three folders for the respective steps: `data`, `asr` and `dst`.

### ASR

The folder [asr/](asr/) is organized with:

- [manifests](asr/manifests/) folder which hosts the manifests for the dev and test set at the different steps of the pipeline (raw transcriptions, time normalization, NER correction).
- [utils](asr/utils/) folder which contains the python scripts for each step of the ASR pipeline (audio extraction, building manifests, transcription, time normalization, NER correction, integration of the transcriptions in the rest of the dialogue and Word Error Rate evaluation).
- Bash scripts to run each step of the pipeline. These scripts simply call the python scripts with specified parameters such as data split or version, model size...

Once you have downloaded the respective files and extracted them from the zip archive, the pipeline expects the data to be in the data folder organized as `data/DSTC11_{split}_{version}/` with split being train, dev or test and version being tts (for train, dev and test) or human (for dev and test) or human_paraphrased (for test only). The text reference files should be in `data/{split}_manifest.txt`.

The pipeline should then follow this process:

- Use the script [extract_audio.sh](asr/extract_audio.sh) to extract the wav files from the downloaded hd5 files.
- Use the script [prepare_manifests.sh](asr/prepare_manifests.sh) to build a manifest file to perform the transcriptions.
- Use the script [transcription.sh](asr/transcription.sh) to transcribe the paths in the manifest file with Whisper.
- Use the script [time_normalization.sh](time_normalization.sh) to perform the time normalization on the transcriptions.
- Use the script [ner_correction.sh](ner_correction.sh) to perform the Named-Entity correction step on the transcriptions.
- Use the script [convert_to_txt.sh](asr/convert_to_txt.sh) to integrate the transcriptions back into the original dialogues text files.

### Data 

The folder [data/data_augmentation](data/data_augmentation) includes our scripts to augment the original DST training data ([train.tts-verbatim.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.txt)):
- Value replacement to prevent memorization of common values: the ontology used to replace values is included in `custom_ontology.json` and `replace_values.py` will perform the replacement based on a txt file in the DSTC11 format.
- TTS-ASR pipeline to simulate ASR errors: `tts.py` and `asr.py`.

### DST 

Based on the augmented train set and the different ASR transcriptions of the dev and test sets obtained from the previous steps, you can train a DST model, evaluate it, and perform inference.

An example shell script is provided for training and evaluation (`run_dst.sh`).

By default the inference script will store all of the evaluation examples with the respective predictions in separate files.

# License

Copyright (c) 2023 Orange

This code is released under the MIT license. See the `LICENSE` file for more information.
