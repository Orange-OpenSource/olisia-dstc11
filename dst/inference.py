# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import argparse
import logging
import math
import os
import json
import random
from pathlib import Path
from collections import defaultdict

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
from utils.postprocess import postprocess_state_dstc11_submission as postprocess_state

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_txt_file", type=str, default=None)
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

    dialogue_level_testset = parse_txt(args.test_txt_file)
    turn_level_testset = convert_format(dialogue_level_testset) # create examples formatted for datasets

    dialogue_ids2int = defaultdict(lambda: len(dialogue_ids2int))

    def preprocess_function(examples):
        inputs = examples['contexts']
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        model_inputs["dialogue_ids"] =  [dialogue_ids2int[_id] for _id in examples["dialogue_ids"]]
        model_inputs["turn_ids"] = examples["turn_ids"]
        return model_inputs

    test_dataset = turn_level_testset.map(
        preprocess_function,
        batched=True,
        remove_columns=turn_level_testset.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    int2dialogue_ids = {v: k for k, v in dialogue_ids2int.items()}

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    logger.info("***** Running inference *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    progress_bar = tqdm(range(len(test_dataloader)))

    model.eval()
    model = model.to(device)

    gen_kwargs = {
        "max_length": args.max_target_length,
    }

    outputs = {}
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            dialogue_ids = batch.pop("dialogue_ids")
            turn_ids = batch.pop("turn_ids")
            dialogue_ids = dialogue_ids.numpy()
            turn_ids = turn_ids.numpy()
            dialogue_ids = [int2dialogue_ids[_id] for _id in dialogue_ids]

            batch = batch.to(device)
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_preds = [postprocess_state(pred) for pred in decoded_preds]
            
            for pred, dialogue_id, turn_id in zip(decoded_preds, dialogue_ids, turn_ids):
                if dialogue_id not in outputs:
                    outputs[dialogue_id] = [None] * 40
                outputs[dialogue_id][turn_id] = pred

            progress_bar.update(1)

    for dialogue_id, turns in outputs.items():
        outputs[dialogue_id] = [
                    {
                    'response': None,
                    'state': state,
                    'active_domains': None
                    }
                for state in turns if state != None
                ]

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}', exist_ok=True)
        results_name = "results_" +  os.path.basename(args.test_txt_file)[:-4]
        with open(f'{args.output_dir}/{results_name}.json', 'w') as ostr:
            json.dump(outputs, ostr)


if __name__ == "__main__":
    main()
