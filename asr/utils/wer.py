# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin, Valentin Vielzeuf

import jiwer
import numpy as np
import json
import sys
from whisper.normalizers.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

transformations =  jiwer.Compose([
                lambda l: normalizer(l[0]),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveSpecificWords(["\n","??",")",":",">","-",]),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
            ])

per_sample_counts = np.zeros(4)
with open(sys.argv[1],"r") as f: 
    n = 0
    per_sample_dico = {}
    for line in f.readlines():
        dico = json.loads(line)
        result_dico = jiwer.compute_measures(dico["text"].replace(" ' ","'"), dico["pred_text"], truth_transform=transformations, hypothesis_transform=transformations)
        h,s,d,i = result_dico["hits"], result_dico["substitutions"], result_dico["deletions"], result_dico["insertions"]
        per_sample_counts += np.array([h,s,d,i])
        per_sample_dico[dico["audio_filepath"]] = {}
        per_sample_dico[dico["audio_filepath"]]["duration"] = dico["duration"]
        per_sample_dico[dico["audio_filepath"]]["wer"] = (s+d+i)/(h+s+d)
        n += 1

    h,s,d,i = per_sample_counts
    final_wer = (s+d+i)/(h+s+d)

    print("Final WER", final_wer) 


