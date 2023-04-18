# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import time
import argparse
import logging
import math
import os
import json
import random
from pathlib import Path
from pprint import pformat

import datasets
import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from filelock import FileLock
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from utils.preprocess import expand_dialogue, parse_txt, convert_format, integrate_asr_user_utterances
from utils.postprocess import postprocess_state
from utils.metrics import DSTmetrics

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_txt_file", type=str, default=None)
    parser.add_argument("--dev_hd5_dir", type=str, default=None)
    parser.add_argument("--use_asr_hyp", type=bool, default=False)
    parser.add_argument("--max_source_length", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--max_target_length", type=int, default=512, help="The maximum total sequence length for target text after tokenization.") 
    parser.add_argument("--max_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the results.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    device = torch.device('cuda')
    logger.info(device)
    
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    max_target_length = args.max_target_length
    padding = False

    dialogue_level_devset = parse_txt(args.dev_txt_file)
    if args.use_asr_hyp:
        dialogue_level_devset = integrate_asr_user_utterances(dialogue_level_devset, args.dev_hd5_dir)
    turn_level_devset = convert_format(dialogue_level_devset) # create examples formatted for datasets

    def preprocess_function(examples):
        inputs = examples['contexts']
        targets = examples['states']
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        #model_inputs["dialogue_ids"] = examples["dialogue_id"]
        return model_inputs

    test_dataset = turn_level_devset.map(
        preprocess_function,
        batched=True,
        remove_columns=turn_level_devset.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    metrics = DSTmetrics()

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    progress_bar = tqdm(range(len(test_dataloader)))

    model.eval()
    model = model.to(device)

    gen_kwargs = {
        "max_length": args.max_target_length,
    }

    output_dials = {} # dialogue_id: [turn_id: context, gold state, pred state, ...]
    samples = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            #dialogue_ids = batch.pop("dialogue_ids")
            #turn_ids = batch.pop("turn_ids")

            #dialogue_ids = dialogue_ids.numpy()
            #turn_ids = turn_ids.numpy()
            #dialogue_ids = [int2dialogue_ids[i] for i in dialogue_ids]

            batch = batch.to(device)
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            labels = batch["labels"]

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = [postprocess_state(pred) for pred in decoded_preds]
            decoded_labels = [postprocess_state(ref) for ref in decoded_labels]

            #logger.info(f"\nSample pred: {decoded_preds[0]} \n Reference: {decoded_labels[0]}\n")
            samples.append(
                    {'context': tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True),
                     'pred': decoded_preds[0],
                     'ref': decoded_labels[0]}
                    )

            metrics.add_batch(decoded_preds, decoded_labels, tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
            progress_bar.update(1)

        #if args.output_dir is not None:
        #    for i in range(len(batch)):
        #        if decoded_preds[i] != decoded_labels[i]:
        #            context = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
        #            if dialogue_ids[i] not in wrong_predictions:
        #                wrong_predictions[dialogue_ids[i]] = {}
        #            wrong_predictions[dialogue_ids[i]][turn_ids[i]] = {
        #                    'context': context,
        #                    'prediction': decoded_preds[i],
        #                    'label': decoded_labels[i],
        #                    }

    result = metrics.compute()
    for spl in random.sample(samples, 10):
        logger.info(f"\nSample pred: {spl['pred']} \n Reference: {spl['ref']}\n")
    logger.info(pformat(result))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}/results', exist_ok=True)
        os.makedirs(f'{args.output_dir}/predictions', exist_ok=True)
        results_name = "results" +  os.path.basename(args.dev_txt_file)[12:-4]
        with open(f'{args.output_dir}/results/{results_name}.json', 'w') as ostr:
            json.dump(result, ostr)
        with open(f'{args.output_dir}/predictions/{results_name}.json', 'w') as ostr:
            json.dump(samples, ostr)


if __name__ == "__main__":
    main()
