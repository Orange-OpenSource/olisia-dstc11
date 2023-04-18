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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm.auto import tqdm

import transformers
#from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    Adafactor,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from utils.preprocess import expand_dialogue, parse_txt, convert_format, integrate_asr_user_utterances
from utils.postprocess import postprocess_state
from utils.metrics import DSTmetrics

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt_file", type=str, default=None)
    parser.add_argument("--dev_txt_file", type=str, default=None)
    parser.add_argument("--train_hd5_dir", type=str, default=None)
    parser.add_argument("--dev_hd5_dir", type=str, default=None)
    parser.add_argument("--use_asr_hyp", action='store_true', default=False)
    parser.add_argument("--max_source_length", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--max_target_length", type=int, default=512, help="The maximum total sequence length for target text after tokenization.") 
    parser.add_argument("--max_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_learning_rate", action='store_true', default=False)
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--patience", type=int, default=0, help="Number of epochs to wait before early stop if no progress on the validation set.")
    parser.add_argument("--min_epoch", type=int, default=0, help="Minimum number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    #parser.add_argument("--local_rank", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.device(local_rank)
    logger.info(device)

    logger.setLevel(logging.INFO)
    #datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = False

    dialogue_level_trainset = parse_txt(args.train_txt_file)
    if args.use_asr_hyp:
        dialogue_level_trainset = integrate_asr_user_utterances(dialogue_level_trainset, args.train_hd5_dir)
    turn_level_trainset = convert_format(dialogue_level_trainset)

    dialogue_level_devset = parse_txt(args.dev_txt_file)
    if args.use_asr_hyp:
        dialogue_level_devset = integrate_asr_user_utterances(dialogue_level_devset, args.dev_hd5_dir)
    turn_level_devset = convert_format(dialogue_level_devset)

    dialogue_ids2int = {}
    for dataset in [turn_level_trainset, turn_level_devset]:
        for dialogue_id in dataset['dialogue_ids']:
            dialogue_ids2int[dialogue_id] = len(dialogue_ids2int)

    def preprocess_function(examples):
        inputs = examples['contexts']
        targets = examples['states']
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["turn_ids"] = examples["turn_ids"]

        model_inputs["dialogue_ids"] = []
        for dialogue_id in examples['dialogue_ids']:
            model_inputs["dialogue_ids"].append(dialogue_ids2int[dialogue_id])
        return model_inputs

    train_dataset = turn_level_trainset.map(
        preprocess_function,
        batched=True,
        remove_columns=turn_level_trainset.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    eval_dataset = turn_level_devset.map(
        preprocess_function,
        batched=True,
        remove_columns=turn_level_devset.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    int2dialogue_ids = {v: k for k, v in dialogue_ids2int.items()}

    #eval_dataset = eval_dataset.select([i for i in range(500, 564)])

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {turn_level_trainset[index]}.")
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.warmup_learning_rate:
        num_warmup_steps = num_update_steps_per_epoch
    else:
        num_warmup_steps = 0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    metrics = DSTmetrics()

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    torch.distributed.init_process_group(backend='nccl')  #nccl gloo
    #torch.distributed.init_process_group(backend='YOUR BACKEND',  init_method='env://')

    early_stopping = EarlyStopping(patience=args.patience, min_epoch=args.min_epoch)
    wrong_predictions = {}

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    last_epoch = args.num_train_epochs
    for epoch in range(args.num_train_epochs):
        model.train()
        train_losses = []
        eval_losses = []
        for step, batch in enumerate(train_dataloader):
            dialogue_ids = batch.pop("dialogue_ids").numpy()
            turn_ids = batch.pop("turn_ids").numpy()
            dialogue_ids = [int2dialogue_ids[i] for i in dialogue_ids]
                
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss)
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': loss.item()})
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        train_loss = loss
        model.eval()
        args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
        }


        logger.info("***** Running validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                dialogue_ids = batch.pop("dialogue_ids")
                turn_ids = batch.pop("turn_ids")

                batch = batch.to(device)
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                labels = batch["labels"]

                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()
                dialogue_ids = dialogue_ids.numpy()
                turn_ids = turn_ids.numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds = [postprocess_state(pred) for pred in decoded_preds]
                decoded_labels = [postprocess_state(ref) for ref in decoded_labels]

                metrics.add_batch(decoded_preds, decoded_labels)

                if step % 5 == 0:
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_losses.append(loss)

        eval_loss = sum(eval_losses) / len(eval_losses)
        train_loss = sum(train_losses) / len(train_losses)

        result = metrics.compute()
        logger.info(
                f"""Epoch {epoch}\n
                Sample pred: {decoded_preds[0]}
                Reference: {decoded_labels[0]}\n
                Train loss: {round(train_loss.item(), 5)}
                Validation loss: {round(eval_loss.item(), 5)}
                JGA: {pformat(result['joint goal accuracy'])}
                """
        )
        #early_stopping.on_epoch_end(result['joint goal accuracy'], epoch)
        early_stopping.on_epoch_end(eval_loss, epoch)
        if early_stopping.stop_training:
            logger.info(f"Has not improved for {args.patience} epochs, stop training.")
            last_epoch = epoch
            break
        logger.info(pformat(result))

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(args.output_dir + '/config.txt', 'w') as ostr:
            print(f"""
            Model: {args.model_name_or_path}\n
            Num epochs: {last_epoch}\n
            Learning rate: {args.learning_rate}\n
            """, ostr)

class EarlyStopping:
    def __init__(self, patience=5, min_epoch=25):
        self.patience = patience
        self.min_epoch = min_epoch
        self.values = []
        self.best_value = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, value):
        if value > self.best_value:
            self.best_value = value
        self.values.append(value)
        if epoch > self.min_epoch:
            if all(val < self.best_value for val in self.values[-self.patience:]):
                self.stop_training = True
                print("Epoch %05d: early stopping THR" % epoch)

if __name__ == "__main__":
    main()

