# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import json
import random
import argparse
import tqdm

with open('custom_ontology.json', 'r') as istr:
    ONTOLOGY = json.load(istr)
SLOTS_TO_UPDATE = ['train-departure', 'train-destination', 'hotel-name', 'restaurant-name',
        'train-arriveby', 'train-leaveat', 'taxi-arriveby', 'taxi-leaveat', 'restaurant-time']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument('-o', "--output_file", '-o', type=str)
    parser.add_argument("--num_dataset_copies", '-n', type=int, default=1)
    parser.add_argument('--do_paraphrase', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_copies = []
    if args.do_paraphrase:
        from LAUG.aug import Text_Paraphrasing
        TP = Text_Paraphrasing()
    else:
        TP = None
    for i in range(args.num_dataset_copies):
        print(i)
        dialogues = parse_txt(args.input_file)
        dialogues = filter_domains(dialogues)
        if i == 0:
            new_dialogues = replace_values(dialogues, False, TP)
        else:
            new_dialogues = replace_values(dialogues, args.do_paraphrase, TP)
        dataset_copies.append(new_dialogues)
    write_new_file(dataset_copies, args.output_file)


def replace_values(dialogues, do_paraphrase, TP):
    """Replace values for selected slots in both utterances and dialogue states using custom ontology."""

    for dial_id, data in tqdm.tqdm(dialogues.items()):
        updated_slots = {}
        mapping_value_changes = {}
        mapping_placeholders = {}

        # 1) Get new values and replace them in dialogue states
        for i in range(len(data['lines'])):
            new_state = []
            text = data['texts'][i]
            if text.startswith('user:') and data['states'][i]: # user turn and non-empty dialogue state
                state = {slot_value.split('=')[0]: slot_value.split('=')[1] for slot_value in data['states'][i].split('; ')}
                for slot, value in state.items():
                    if slot in SLOTS_TO_UPDATE:
                        if value == 'dontcare':
                            continue
                        if slot not in updated_slots: # value hasn't been replaced yet for this slot
                            if value in mapping_value_changes: # value is shared between slots
                                new_value = mapping_value_changes[value]
                            else:
                                new_value = get_new_value(slot, value, ONTOLOGY)
                                mapping_value_changes[value] = new_value
                            updated_slots[slot] = {
                                    'previous': value,
                                    'new': new_value,
                                    }
                        elif value != updated_slots[slot]['previous']: # value has changed in the course of the dialogue (eg. booking unsuccessful)
                            if value in mapping_value_changes: # value is shared between slots
                                new_value = mapping_value_changes[value]
                            else:
                                new_value = get_new_value(slot, value, ONTOLOGY)
                                mapping_value_changes[value] = new_value
                            updated_slots[slot]['previous'] = value
                            updated_slots[slot]['new'] = new_value
                        new_state.append(slot + '=' + updated_slots[slot]['new']) 
                    else:
                        new_state.append(slot + '=' + value)
                dialogues[dial_id]['states'][i] = '; '.join(new_state) 

        # 2) Replace values in utterances based on mapping
        mapping_value_changes = dict(sorted(mapping_value_changes.items(), key=lambda item: len(item[0]), reverse=True)) # sort by length to avoid replacing a subset   
        for i in range(len(data['lines'])):
            new_text = data['texts'][i]
            for old_val, new_val in mapping_value_changes.items():
                if old_val in data['texts'][i]:
                    new_text = new_text.replace(old_val, new_val)

            if do_paraphrase and new_text.startswith('user:') and dialogues[dial_id]['states'][i]:
                state = {slot_value.split('=')[0]: slot_value.split('=')[1] for slot_value in dialogues[dial_id]['states'][i].split('; ')}
                if i > 0 and dialogues[dial_id]['states'][i - 2]:
                    prev_state = {slot_value.split('=')[0]: slot_value.split('=')[1] for slot_value in dialogues[dial_id]['states'][i - 2].split('; ')}
                else:
                    prev_state = {}
                state_change = {slot: value for slot, value in state.items() if slot not in prev_state}

                if state_change:
                    new_text = paraphrase(new_text.replace('user: ', ''), state_change, TP)
                    new_text = 'user: ' + new_text
            dialogues[dial_id]['texts'][i] = new_text
        if do_paraphrase:
            TP.model.init_session()

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


def get_new_value(slot, old_value, ontology):
    if slot == 'train-departure' or slot == 'train-destination':
        value = random.sample(ontology['towns'], 1)[0]
    elif slot == 'hotel-name':
        value = random.sample(ontology['hotels'], 1)[0]
    elif slot == 'restaurant-name':
        value = random.sample(ontology['restaurants'], 1)[0]
    elif any(time_name in slot for time_name in ['time', 'leaveat', 'arriveby']):
        if ':' in old_value: # leave round hours as is
            if '|' in old_value:
                old_value = old_value.split('|')[1] + ' pm' # few cases with multiple options 
            hour, rest = old_value.split(':')
            minutes, time_of_day = rest.split()
            new_minutes = str(random.randint(1, 59))
            if len(new_minutes) == 1:
                new_minutes = '0' + new_minutes
            value = f'{hour}:{new_minutes} {time_of_day}'
        else:
            value = old_value
    return value.lower()


def write_new_file(dataset_copies, file_path):
    with open(file_path, 'w') as ostr:
        for j, dialogues in enumerate(dataset_copies):
            for dial_id, data in dialogues.items():
                for i in range(len(data['lines'])):
                    rest, turn_id = data['lines'][i].split(' turn_id: ')
                    rest, dial_id = rest.rsplit(' ', 1)
                    new_line = f"{rest} {j}-{dial_id} turn_id: {turn_id} text: {data['texts'][i]} state: {data['states'][i]}"
                    print(new_line, file=ostr)
                print('END_OF_DIALOG', file=ostr)



def paraphrase(text, state, TP):
    span_info = [[f"{slot.split('-')[0].capitalize()}-Inform", slot.split('-')[1], value, 0, 0] for slot, value in state.items()]
    paraphrased_text, _ = TP.aug(text, span_info)
    if any(value not in paraphrased_text for slot, value in state.items() if slot not in ['hotel-internet', 'hotel-parking']):
        return text
    else:
        return paraphrased_text


def filter_domains(dialogues):
    """Filter function to remove dialogues in the bus or hospital domain which only appear in the training set."""

    filtered_dialogues = {}
    for dial_id, data in dialogues.items():
        states = []
        for i in range(len(data['states'])):
            if data['states'][i]:
                states.append({slot_value.split('=')[0]: slot_value.split('=')[1] for slot_value in data['states'][i].split('; ')}) 
        filtering = False
        for state in states:
            if any('bus' in key for key in state.keys()) or any('hospital' in key for key in state.keys()):
                filtering = True
        if filtering:
            continue
        else:
            filtered_dialogues[dial_id] = data
    return filtered_dialogues


if __name__ == "__main__":
    main()
