# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import os
import json
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", '-i', type=str, default=None)
    parser.add_argument("--output_file", '-o', type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    results = get_results(args.input_folder)
    write_results_file(results, args.output_file)

def get_results(input_folder):
    results = {}
    p = Path(input_folder)
    exp_files = [x for x in p.iterdir() if not x.is_dir()]
    for exp_file in exp_files:
        with open(exp_file, 'r') as istr:
            exp_result = json.load(istr)
        results[exp_file.name] = {'JGA': exp_result['joint goal accuracy']}
        for slot, scores in exp_result['slot scores'].items():
            results[exp_file.name][slot] = scores['accuracy']
    return results

def write_results_file(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(results)
    with open(output_file, 'w') as md:
        df.to_markdown(buf=md)

if __name__ == "__main__":
    main()
