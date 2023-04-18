# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_file", type=str)
    parser.add_argument('-o', "--output_file", type=str, required=False)
    parser.add_argument('--inplace', action='store_true')
    parser.add_argument("--gold_states_file", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dialogues = parse_txt(args.input_file)
    with open(args.gold_states_file, 'r') as istr:
        gold_states = json.load(istr)
    dialogues = integrate_gold_states(dialogues, gold_states)
    if args.inplace:
        write_new_file(dialogues, args.input_file)
    else:
        write_new_file(dialogues, args.output_file)


def integrate_gold_states(dialogues, gold_states):
    for dial_id, turns in gold_states.items():
        for i, state in enumerate(turns):
            state_str = flatten_state(state)
            dialogues[dial_id]['states'][i * 2] = state_str
    return dialogues

        
def flatten_state(state_dict):
    pairs = []
    for domain, domain_dict in state_dict.items():
        for slot, value in domain_dict.items():
            if 'book' in slot:
                slot = slot.replace('book', '')
            pairs.append(f"{domain}-{slot}={value}")
    state_str = '; '.join(pairs)
    return state_str


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
                rest, turn_id = data['lines'][i].split(' turn_id: ')
                new_line = f"{rest} turn_id: {turn_id} text: {data['texts'][i]} state: {data['states'][i]}"
                print(new_line, file=ostr)
            print('END_OF_DIALOG', file=ostr)


if __name__ == "__main__":
    main()
