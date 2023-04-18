# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: 
    references: 
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

slot_names = ['attraction-area', 'attraction-name', 'attraction-type', 'hotel-area', 'hotel-day', 'hotel-people', 
              'hotel-stay', 'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', 
              'hotel-type', 'restaurant-area', 'restaurant-day', 'restaurant-people', 'restaurant-time',
              'restaurant-food', 'restaurant-name', 'restaurant-pricerange', 'taxi-arriveby', 'taxi-departure', 'taxi-destination',
              'taxi-leaveat', 'train-arriveby', 'train-people', 'train-day', 'train-departure', 'train-destination', 'train-leaveat']

class DSTmetrics:
    """TODO: Short description of my metric."""

    def __init__(self):
        self.predictions = []
        self.references = []
        self.contexts = []

    def add_batch(self, preds, refs, contexts):
        self.predictions += preds
        self.references += refs
        self.contexts += contexts 

    def compute(self):
        """Returns the scores"""
        joint_goal_accuracy = compute_joint_goal_accuracy(self.predictions, self.references)
        average_goal_accuracy = compute_average_goal_accuracy(self.predictions, self.references)
        slot_scores, macro_averaged_scores = compute_slot_scores(self.predictions, self.references)
        relative_slot_accuracy = compute_relative_slot_accuracy(self.predictions, self.references)
        forgotten_mean_prop, invented_mean_prop = slot_name_scores(self.predictions, self.references)
        hallucination_rate = slot_value_scores(self.predictions, self.references, self.contexts)
        return {
            "joint goal accuracy": round(joint_goal_accuracy * 100, 2),
            "average goal accuracy": round(average_goal_accuracy * 100, 2),
            "slot accuracy": round(macro_averaged_scores['accuracy'] * 100, 2),
            "F1 slot": round(macro_averaged_scores['f1'] * 100, 2),
            "slot scores": {slot: {score: round(value * 100, 2) for score, value in slot_scores[slot].items()} for slot in slot_scores},
            "Forgotten slot mean proportion (lower = better)": round(forgotten_mean_prop*100, 2),
            "Invented slot mean proportion (lower=better)": round(invented_mean_prop*100, 2),
            "Hallucination proportion in named entities errors": round(hallucination_rate*100, 2)
        }


def slot_name_scores(predictions, references):
    """
    Returns the mean of the proportion of forgotten and invented slot types per dialogue turn.
    """
    name_forgotten_measures = []
    name_invented_measures = []
    for pred, ref in zip(predictions, references):
        ref_slot_names = ref.keys()
        pred_slot_names = pred.keys()
        turn_measures = {"forgotten": 0, "invented": 0}
        temp_ref = set(ref_slot_names)
        invented = [value for value in pred_slot_names if value not in temp_ref]
        temp_pred = set(pred_slot_names)
        forgotten = [value for value in ref_slot_names if value not in temp_pred]
        
        if len(ref_slot_names) > 0:
            turn_measures["forgotten"] = len(forgotten)/len(ref_slot_names)
        if len(pred_slot_names) > 0:
            turn_measures["invented"] = len(invented)/len(pred_slot_names)
        name_forgotten_measures.append(turn_measures["forgotten"])
        name_invented_measures.append(turn_measures['invented']) 

    return sum(name_forgotten_measures)/len(name_forgotten_measures), sum(name_invented_measures)/len(name_invented_measures)

def slot_value_scores(predictions, references, contexts):
    """
    Returns the average presence of the reference value in the context which provides a measure of hallucination.
    """
    total_errors = 0
    hallucinations = 0
    considered_slots = ['attraction-name', 'hotel-name', 'restaurant-name', 'taxi-departure', 'train-departure', 'taxi-destination', 'train-destination']
    for pred, ref, cont in zip(predictions, references, contexts):
        if pred != ref and any([slot in ref.keys() for slot in considered_slots]):
            for slot in considered_slots:
                if slot in ref.keys():
                    if (slot in pred.keys() and ref[slot] != pred[slot]):
                        total_errors += 1
                        if pred[slot] not in cont:
                            hallucinations += 1
                    elif slot not in pred.keys():
                        total_errors += 1
    return hallucinations/total_errors

def compute_joint_goal_accuracy(predictions, references):
    """Strict match with reference dialogue state."""
    scores = []
    for pred, ref in zip(predictions, references):
        if len(ref) != 0:
            if pred == ref:
                scores.append(1)
            else:
                scores.append(0)
        else:
            if len(pred) == 0:
                scores.append(1)
            else:
                scores.append(0)
    joint_goal_accuracy = sum(scores) / len(scores) if len(scores) != 0 else 0
    return joint_goal_accuracy


def compute_average_goal_accuracy(predictions, references):
    """Slots which have a non-empty assignment in the ground truth dialogue state are considered for accuracy."""
    active_predictions = []
    active_references = []
    for pred, ref in zip(predictions, references):
        active_ref = {slot: value for slot, value in ref.items() if value != 'none'}
        active_pred = {slot: value for slot, value in pred.items() if slot in active_ref}
        if len(active_ref) != 0:
            active_references.append(active_ref)
            active_predictions.append(active_pred)
    average_goal_accuracy = sum(i == j for i, j in zip(active_predictions, active_references)) / len(active_predictions) if active_predictions else 0
    return average_goal_accuracy


def compute_slot_scores(predictions, references):
    scores = []
    total = 0
    slot_counts = {
            slot: {
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0,
                }
        for slot in slot_names
        }

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    for pred, ref in zip(predictions, references):
        true_positives += [slot for slot in pred if (slot in ref) and (pred[slot] == ref[slot]) and (ref[slot] != 'none')]
        true_negatives += [slot for slot in pred if (slot in ref) and (pred[slot] == ref[slot]) and (ref[slot] == 'none')]
        false_positives += [slot for slot in pred if (slot not in ref) or (slot in ref and pred[slot] != ref[slot] and pred[slot] != 'none')]
        false_negatives += [slot for slot in ref if (slot not in pred) and (ref[slot] != 'none')]

    for slot in true_positives:
        if slot not in slot_counts:
            continue
        slot_counts[slot]['tp'] += 1
    for slot in true_negatives:
        if slot not in slot_counts:
            continue
        slot_counts[slot]['tn'] += 1
    for slot in false_positives:
        if slot not in slot_counts:
            continue
        slot_counts[slot]['fp'] += 1
    for slot in false_negatives:
        if slot not in slot_counts:
            continue
        slot_counts[slot]['fn'] += 1

    for slot in slot_counts:
        slot_counts[slot]['total'] = sum(slot_counts[slot].values())

    slot_scores = {
            slot: {
                'accuracy': (slot_counts[slot]['tp'] + slot_counts[slot]['tn']) / slot_counts[slot]['total'] if slot_counts[slot]['total'] != 0 else 0,
                'precision': slot_counts[slot]['tp'] / (slot_counts[slot]['tp'] + slot_counts[slot]['fp']) if slot_counts[slot]['tp'] + slot_counts[slot]['fp'] != 0 else 0,
                'recall': slot_counts[slot]['tp'] / (slot_counts[slot]['tp'] + slot_counts[slot]['fn']) if slot_counts[slot]['tp'] + slot_counts[slot]['fn'] != 0 else 0,
                }
            for slot in slot_counts
            }
    for slot in slot_scores:
        slot_scores[slot]['f1'] = 2 * slot_scores[slot]['precision'] * slot_scores[slot]['recall'] \
                / (slot_scores[slot]['precision'] + slot_scores[slot]['recall']) if slot_scores[slot]['precision'] + slot_scores[slot]['recall'] != 0 else 0

    macro_averaged_scores = {
        'accuracy': [slot_scores[slot]['accuracy'] for slot in slot_scores],
        'f1': [slot_scores[slot]['f1'] for slot in slot_scores],
        }
    macro_averaged_scores = {metric: sum(scores) / len(scores) if len(scores) != 0 else 0 for metric, scores in macro_averaged_scores.items()}

    return slot_scores, macro_averaged_scores
