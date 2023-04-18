# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

number_replacements = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'eleven': '11',
        'twelve': '12',
        'thirteen': '13',
        'fourteen': '14',
        'fifteen': '15',
        'sixteen': '16',
        'seventeen': '17',
        'eighteen': '18',
        'nineteen': '19',
        'twenty': '20',
        }
        
def postprocess_state(state):
    """Returns predicted and gold states as dictionaries of slot-value pairs."""
    processed_state = {}
    state = state.replace("__unk__", "")
    if ';' not in state:
        return processed_state
    for pair in state.split(';'):
        if '=' not in pair:
            continue
        pair = pair.split('=')
        if len(pair) != 2:
            continue
        slot, value = pair
        processed_state[slot.strip()] = value.strip()
    return processed_state

