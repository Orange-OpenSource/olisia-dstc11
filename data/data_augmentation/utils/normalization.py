# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import re
from .w2n import word_to_num

TIME_PATTERN_N2W = re.compile(r'\b(\d?\d):(\d\d) (\wm)\b')
TIME_PATTERN_N2W2 = re.compile(r'\b(\d?\d) (\wm)\b')

TIME_PATTERN_W2N = re.compile(
    "(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s"
    "(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
    "\s(p m|a m)"
)
TIME_PATTERN_W2N2 = re.compile(
    "(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s"
    "(twenty|thirty|forty|fifty)?\s?(one|two|three|four|five|six|seven|eight|nine)?"
    "\s?(p m|a m)"
)
DIGIT_PATTERN_N2W = re.compile(r"\b(\d)\b")
DIGIT_PATTERN_W2N = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine)\b")

def normalize_n2w(text):
    text = normalize_time_n2w(text)
    text = normalize_digit_n2w(text)
    return text

def normalize_w2n(text):
    text = normalize_time_w2n(text)
    text = normalize_digit_w2n(text)
    return text


def normalize_digit_n2w(text):
    text = re.sub(DIGIT_PATTERN_N2W, lambda x: n2w(x.group(0)), text)
    return text

def normalize_time_n2w(text):
    match = re.search(TIME_PATTERN_N2W, text)
    match2 = re.search(TIME_PATTERN_N2W2, text)
    if match:
        full_string = match.group(0)
        hour = match.group(1)
        minutes = match.group(2)
        time_of_day = match.group(3)
        if time_of_day == 'am':
            time_of_day = 'ay em'
        elif time_of_day == 'pm':
            time_of_day = 'pee em'
        time = n2w(hour) + ' '  + n2w(minutes) + ' ' + time_of_day
        text = text.replace(full_string, time)
    elif match2:
        full_string = match2.group(0)
        hour = match2.group(1)
        time_of_day = match2.group(2)
        if time_of_day == 'am':
            time_of_day = 'ay em'
        elif time_of_day == 'pm':
            time_of_day = 'pee em'
        time = n2w(hour) + ' ' + time_of_day
        text = text.replace(full_string, time)
    return text


def normalize_time_w2n(text):
    match = re.search(TIME_PATTERN_W2N, text)
    match2 = re.search(TIME_PATTERN_W2N2, text)
    if match:
        full_string = match.group(0)
        hour = match.group(1)
        minutes = match.group(2)
        time_of_day = match.group(3)
        time_of_day = time_of_day[0] + time_of_day[2]
        time = f'{word_to_num(hour)}:{word_to_num(minutes)} {time_of_day}'
        text = text.replace(full_string, time)
    elif match2:
        full_string = match2.group(0)
        hour = match2.group(1)
        dozens = match2.group(2)
        minutes = match2.group(3)
        time_of_day = match2.group(4)
        time_of_day = time_of_day[0] + time_of_day[2]
        if dozens and minutes:
            minutes = dozens + ' ' + minutes
            time = f'{word_to_num(hour)}:{word_to_num(minutes)} {time_of_day}'
        elif dozens and not minutes:
            time = f'{word_to_num(hour)}:{word_to_num(dozens)} {time_of_day}'
        elif minutes and not dozens:
            time = f'{word_to_num(hour)}:0{word_to_num(minutes)} {time_of_day}'
        elif not minutes and not dozens:
            time = f'{word_to_num(hour)} {time_of_day}'
        text = text.replace(full_string, time)
    return text


def normalize_digit_w2n(text):
    text = re.sub(DIGIT_PATTERN_W2N, lambda x: str(word_to_num(x.group(0))), text)
    return text


def n2w(n):
    if n in num2words:
        return num2words[n]
    else:
        tens = str(int(n) - int(n) % 10)
        digits = str(int(n) % 10)
        return num2words[tens] + '-' + num2words[digits].lower()

num2words = {'1': 'one',
             '2': 'two',
             '3': 'three',
             '4': 'four',
             '5': 'five',
             '6': 'six',
             '7': 'seven',
             '8': 'eight',
             '9': 'nine',
             '10': 'ten',
             '11': 'eleven',
             '12': 'twelve',
             '13': 'thirteen',
             '14': 'fourteen',
             '15': 'fifteen',
             '16': 'sixteen',
             '17': 'seventeen',
             '18': 'eighteen',
             '19': 'nineteen',
             '20': 'twenty',
             '30': 'thirty',
             '40': 'forty',
             '50': 'fifty',
             '60': 'sixty',
             '0': 'zero'
            }
