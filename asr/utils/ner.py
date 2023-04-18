# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin, Valentin Vielzeuf

import nemo.collections.nlp as nemo_nlp
import json
import numpy as np
import argparse
import glob
import os
import jiwer
from tqdm import tqdm
import logging
logging.disable(logging.CRITICAL)

"""
    LOC = Geographical Entity
    ORG = Organization
    PER = Person
    GPE = Geopolitical Entity
    TIME = Time indicator
    ART = Artifact
    EVE = Event
    NAT = Natural Phenomenon
"""
def ontology_search(word,vocab,thresh=0.2):
    score = lambda gt : jiwer.wer(" ".join([*word.lower()])," ".join([*gt.lower()]))
    scores = np.vectorize(score)(vocab) 
    mini = np.min(scores)
    if  mini > 0 and mini < thresh:
        return vocab[np.argmin(scores)],1
    else:
        return word,0

def ontology_correction(dico,vocab,pred_context,thresh=0.2):
    counts = 0
    for p in pred_context:
        if p!= "":
            word, c = ontology_search(p,vocab,thresh)
            dico["pred_text"] = dico["pred_text"].replace(p,word)
            counts += c
    return dico, counts


def contextual_correction(dico,agent_context,pred_context,thresh=0.2):
    counts = 0
    for p in pred_context:
        for agent in agent_context:
            if agent != "" and p!= "":
                w = jiwer.wer(" ".join([*p.lower()])," ".join([*agent.lower()])) 
                if w < thresh:
                    dico["pred_text"] = dico["pred_text"].replace(p,agent)
                    if w > 0.0:
                        counts+=1
    return dico, counts

def read_agent(textfile):
    agent_turns = []
    with open(textfile,"r") as f:
        for line in f.readlines():
            if "agent:" in line:
                agent_turns.append(line.replace("agent: ","").replace(" state:",""))
    return agent_turns

def parse_output(semantic_output):
    proper_nouns = []
    semantic_output = semantic_output.split(" ")
    for el in semantic_output:
        if "[" in el and "TIME" not in el:
            proper_nouns.append(el.split("[")[0])
    return proper_nouns        

def collect_proper_nouns(turn,model):
    preds = model.add_predictions(turn)
    out = []
    for pred in preds:
        out+=parse_output(pred)
    return list(np.unique(np.array(out)))

parser = argparse.ArgumentParser(
    description='ASR inference.'
)
parser.add_argument("--folder", required=False, default="/home/kfvn9537/data/Lucas/DSTC11_dev_human/dev.audio-verbatim/", type=str, help="The folder with wav files to process.")
parser.add_argument("--pred_file", required=False, default="/home/kfvn9537/workspace/challenge-slu-lucas/dstc11/asr/nemo_asr_ngram/results/human_train_trainaug_nopara_inverse_normalized.json", type=str, help="The folder with wav files to process.")
parser.add_argument("--ontology", required=False, default=None, type=str, help="The ontolgy to use.")
parser.add_argument("--ner_thresh", required=False, default=0.4, type=float, help="The CER threshhold to replace word.")
parser.add_argument("--ontology_thresh", required=False, default=0.2, type=float, help="The CER threshhold to replace word.")
args = parser.parse_args()

model = nemo_nlp.models.TokenClassificationModel.from_pretrained(model_name="ner_en_bert")


if os.path.exists(args.folder+"/contextual_dict.npy"):
    context_dico = np.load(args.folder+"/contextual_dict.npy",allow_pickle=True).item()
else:
    context_dico = {}
    dialogs = glob.glob(args.folder+"/*/")
    for dialog in tqdm(dialogs):
        agent_turn = glob.glob(dialog+"*.json.txt")[0]
        agent_turn = read_agent(agent_turn)
        context = collect_proper_nouns(agent_turn,model)
        dico = {}
        context_dico[dialog.split("/")[-2]] = context
    np.save(args.folder+"/contextual_dict.npy",context_dico)


vocab = []
if args.ontology != None:
    with open(args.ontology,'r') as f:
        for line in f.readlines():
            vocab.append(line[:-1].lower())
    vocab = np.array(vocab)

replaced_values = 0
with open(args.pred_file,"r") as f:
    if args.ontology != None:
        filename = args.pred_file.split(".json")[0]+"_ner_onto_{}_{}.json".format(args.ner_thresh,args.ontology_thresh)
    else:
        filename = args.pred_file.split(".json")[0]+"_ner_{}.json".format(args.ner_thresh)
    with open(filename,'a') as g:
        iterator = tqdm(f.readlines())
        for line in iterator:
            dico = json.loads(line)
            dialog = dico["audio_filepath"].split("/")[-2]
            print(dialog)
            agent_context = context_dico[dialog]
            print(agent_context)
            pred_context = collect_proper_nouns([dico["pred_text"]],model)
            dico, counts = contextual_correction(dico,agent_context,pred_context,thresh=args.ner_thresh)
            replaced_values += counts
            if args.ontology != None:
                dico, counts = ontology_correction(dico,vocab,pred_context,thresh=args.ontology_thresh)
            replaced_values += counts
            json.dump(dico,g)
            g.write("\n")
            iterator.set_description("Replaced values:{}".format(replaced_values))
print("Number of replaced words:",replaced_values)
