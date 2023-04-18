# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import random
import json
import re
import h5py
from string import ascii_lowercase
from datasets import Dataset

slot_names = ['attraction-area', 'attraction-name', 'attraction-type', 'hotel-area', 'hotel-day', 'hotel-people', 
              'hotel-stay', 'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', 
              'hotel-type', 'restaurant-area', 'restaurant-day', 'restaurant-people', 'restaurant-time',
              'restaurant-food', 'restaurant-name', 'restaurant-pricerange', 'taxi-arriveby', 'taxi-departure', 'taxi-destination',
              'taxi-leaveat', 'train-arriveby', 'train-people', 'train-day', 'train-departure', 'train-destination', 'train-leaveat']

def parse_txt(file_path):
    """
    Parse reference txt file and store dialogues in a dict in the following example format:
     'mul1881': {'states': [{'train-leaveat': '7:04 am', 'train-destination': 'village of four seasons'},
                            {'train-leaveat': '7:04 am', 'train-destination': 'village of four seasons', 'train-day': 'saturday'},
                            ...
                  'turns': [{'agent': '', 'user': 'i need to take a train, going to village of four seasons leaving after 7:04 am.'},
                            {'agent': 'ok, and what day are you traveling?', 'user': 'i'm traveling on saturday.'},
                            ...
                }, ...
    Agent and user utterances are combined in one turn except for the first turn which is user only.
    """
    dialogues = {}
    agent_utterance, user_utterance=None, None
    print(file_path)
    with open(file_path, 'r') as istr:
        for line in istr.readlines():
            if 'END_OF_DIALOG' in line: # only in the train set, not in dev
                continue
            _, _, _, dialog_id, _, turn_id, _, text_with_state = line.split(' ', 7)
            dialog_id = dialog_id.replace('.json', '')
            turn_id = int(turn_id)
            if dialog_id not in dialogues:
                dialogues[dialog_id] = {
                    'turns': [],
                    'states': [],
                }
            text, state = (x.strip() for x in text_with_state.split('state:'))
            state_dict = {}
            if len(state) > 0:
                for slot_value in state.split('; '):
                    slot, value = slot_value.split('=')
                    state_dict[slot] = value
            if turn_id == 1: # first turn is from the user
                user_utterance = text.replace('user: ', '')
                turn = {'user': user_utterance, 'agent': ''}
                dialogues[dialog_id]['turns'].append(turn)
                dialogues[dialog_id]['states'].append(state_dict)
            elif turn_id % 2 == 0: # store agent utterance to combine with next user utterance 
                agent_utterance = text.replace('agent: ', '')
            else:
                user_utterance = text.replace('user: ', '')
                turn = {'agent': agent_utterance, 'user': user_utterance}
                dialogues[dialog_id]['turns'].append(turn)
                dialogues[dialog_id]['states'].append(state_dict)
    dialogues = filter_domains(dialogues)
    return dialogues


def convert_format(dialogue_level_dataset):
    """Convert a dialogue level dataset in a table format for the datasets library."""

    turn_level_dataset = {
             'dialogue_ids': [], 
             'turn_ids': [], 
             'contexts': [], 
             'states': []
             }
    for dialogue_id, data in dialogue_level_dataset.items():
        dialogue_ids, turn_ids, contexts, states = expand_dialogue(dialogue_id, data)
        turn_level_dataset['dialogue_ids'] += dialogue_ids
        turn_level_dataset['turn_ids'] += turn_ids
        turn_level_dataset['contexts'] += contexts
        turn_level_dataset['states'] += states
    turn_level_dataset = Dataset.from_dict(turn_level_dataset)
    return turn_level_dataset


def expand_dialogue(dialogue_id, data):
    """Format training examples by prepending preceding turns as context for each dialogue turn."""

    turn_ids = []
    contexts = []
    states = []
    turns = data['turns']
    for idx in range(len(turns)):
        turn_context = ''
        for i in range(idx + 1): # loop through preceding turns
            if i > 0:
                turn_context += 'agent: ' + turns[i]['agent'] + ' '
            turn_context += 'user: ' + turns[i]['user'] + ' '
        turn_context = turn_context.lower()
        turn_state = '; '.join(slot + '=' + value for slot, value in data['states'][idx].items())
        turn_ids.append(idx)
        contexts.append(turn_context.strip())
        states.append(turn_state)
    dialogue_ids = [dialogue_id] * len(turns)
    return dialogue_ids, turn_ids, contexts, states


def integrate_asr_user_utterances(dialogue_level_dataset, hd5dir):
    """Replace text user utterances with ASR hypothesis counterpart from hd5 file."""

    for dialogue_id, text_data in dialogue_level_dataset.items():
        speech_data = h5py.File(f'{hd5dir}/{dialogue_id}.hd5', 'r')
        for i, turn_id in zip(range(len(text_data['turns'])), speech_data.keys()):
            asr_hyp = speech_data[turn_id].attrs['hyp']
            dialogue_level_dataset[dialogue_id]['turns'][i]['user'] = asr_hyp
    return dialogue_level_dataset


def filter_domains(dialogues):
    """Filter function to remove dialogues in the hospital or police domain which only appear in the training set."""

    filtered_dialogues = {}
    for dial_id, data in dialogues.items():
        if all(len(state) == 0 for state in data['states']):
            continue
        bus = False
        for state in data['states']:
            if any('bus' in key for key in state.keys()):
                bus = True
        if bus:
            continue
        else:
            filtered_dialogues[dial_id] = data
    return filtered_dialogues
