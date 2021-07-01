import re
import torch
import numpy as np

from textattack.shared import AttackedText

from collections import defaultdict, namedtuple

from .sequence import pad_sequence


def get_tokens(text, tokenizer):
    return tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))


def predictions_per_character(out_tokens, predictions):
    single_char_text = []
    single_char_predictions = []
    single_char_raw_output = []

    for out_token, pred in zip(out_tokens, predictions):
        prediction_label = np.argmax(pred)
        prediction_confidence = pred[prediction_label]

        token_chars = list(out_token)

        single_char_text.extend(token_chars)
        single_char_predictions.extend(
            [(prediction_label, prediction_confidence)] * len(token_chars)
        )

        single_char_raw_output.extend(list(
            np.tile(pred, (len(token_chars), 1))
        ))

    return single_char_text, single_char_predictions, single_char_raw_output


def postprocess_ner_output(original_text, predictions, out_tokens):
    """
        Postprocesses NER models output, fixing the issue of a model
        predicting labels for sub-tokens.

        # TODO: We should write automated tests for this
    """
    original_tokens = original_text.split(" ")

    # Remove ## (BERT) BPE artifacts
    original_tokens = [t.replace("##", "") for t in original_tokens]
    out_tokens = [t.replace("##", "") for t in out_tokens]

    # Remove [CLS] / [SEP] tokens
    out_tokens = out_tokens[1:-1]
    predictions = predictions[1:-1]

    single_char_text, single_char_predictions, single_char_raw_output = predictions_per_character(
        out_tokens,
        predictions
    )

    prediction_idx = 0
    current_out_token = ""

    # current_out_token_predictions is an array of tuples in the form
    # (predicted_label, confidence, token_length)
    current_out_token_predictions, preds_per_token, confidence_per_token = [], [], []

    # For each token in the original sentence find the corresponding
    # token (or list of subtokens) in the output predictions
    for token in original_tokens:
        token_start_index = prediction_idx

        while current_out_token != token:
            current_out_token = f"{current_out_token}{single_char_text[prediction_idx]}"

            prediction_label, prediction_confidence = single_char_predictions[prediction_idx]
            current_out_token_predictions.append((prediction_label, prediction_confidence, 1))

            prediction_idx += 1

        # After finding all the subtokens choose the most
        # likely prediction for the whole token
        chosen_label, _ = choose_best_ner_prediction(current_out_token_predictions)
        
        # Calculate all the label confidences
        label_confidences = {}

        for label in range(0, len(single_char_raw_output[0])):
            label_confidences[label] = sum([
                single_char_raw_output[i][label] for i in range(token_start_index, prediction_idx)
            ]) / len(token)

        # Get the label confidence
        chosen_label_confidence = label_confidences[chosen_label]

        preds_per_token.append(
            (chosen_label, chosen_label_confidence, label_confidences)
        )

        current_out_token = ""
        current_out_token_predictions = []

    final_tokens, final_labels = [], []
    final_label_confidence, final_confidence_dicts = [], []

    for token, (label, label_confidence, confidences) in zip(original_tokens, preds_per_token):
        final_tokens.append(token)
        final_labels.append(label)
        final_label_confidence.append(label_confidence)
        final_confidence_dicts.append(confidences)

    return final_tokens, torch.tensor(final_labels), torch.tensor(final_label_confidence), final_confidence_dicts


def choose_best_ner_prediction(preds_list):
    """
        Given a list of subtoken predictions weight them by
        confidence score and length and choose the most likely
        prediction
    """
    # Count how many characters each class covers for this token
    # and weight them by model confidence
    class_weights = defaultdict(int)

    for pred in preds_list:
        label, confidence, length = pred

        # Multiply the text length by the label's confidence
        class_weights[int(label)] += length * confidence

    weights = [(k, class_weights[k]) for k in class_weights.keys()]
    weights = sorted(weights, key=lambda x: x[1], reverse=True)

    return weights[0]


def tensor_mask(tensor):
    """
        Given a 1xN tensor returns a 1xN tensor where
        each element is 1 if its original counterpart
        is > 0 and 0 otherwise
    """
    return torch.tensor([
        1 if x > 0 else 0 for x in tensor
    ])


def diff_elements_count(a: torch.tensor, b: torch.tensor):
    """
        Returns the number of different elements between
        tensor a and tensor b
    """
    return len(elements_diff(a, b))


def elements_diff(a: list, b: list):
    indices = set()

    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            indices.add(i)

    return indices


def is_ascii(s):
    return all(ord(c) < 128 for c in s)
