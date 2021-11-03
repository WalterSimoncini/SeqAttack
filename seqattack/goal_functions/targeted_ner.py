import torch
import numpy as np

from .ner_goal_function_result import NERGoalFunctionResult

from textattack.goal_functions.goal_function import GoalFunction
from seqattack.utils import diff_elements_count, tensor_mask

from .ner import NERGoalFunction


class TargetedNERGoalFunction(NERGoalFunction):
    """
        Goal function for NER that attempts to coerce named entities
        to be classified as "O" - a.k.a. no entity
    """
    def _is_goal_complete(self, model_output, attacked_text):
        preds, _, _, _ = self._preprocess_model_output(
            model_output,
            attacked_text)

        return self._get_score_labels(preds, attacked_text) >= self.min_percent_entities_mispredicted

    def _get_score(self, model_output, attacked_text):
        return self._get_score_confidence(
            model_output,
            attacked_text)

    def _get_score_labels(self, model_output, attacked_text):
        mapped_ground_truth = self._preprocess_ground_truth(attacked_text)
        named_entities_mask = tensor_mask(mapped_ground_truth)

        if named_entities_mask.sum() == 0:
            # All entities are already "O", nothing to do here
            return 1

        total_score = 0

        for truth, model_out in zip(named_entities_mask, model_output):
            if truth > 0 and model_out == 0:
                total_score += 1

        # Return the percentage of mispredicted entities
        return (total_score / named_entities_mask.sum()).item()

    def _get_score_confidence(self, model_output, attacked_text):
        _, _, truth_entities_mask, confidence_scores = self._preprocess_model_output(
            model_output,
            attacked_text)

        if truth_entities_mask.sum() == 0:
            # Nothing to do here, we have no named entities
            return 1

        no_entity_confidences = [confs[0] for confs in confidence_scores]

        total_score = 0

        for is_entity, no_entity_confidence in zip(truth_entities_mask, no_entity_confidences):
            if is_entity == 1:
                total_score += no_entity_confidence

        return float(total_score / truth_entities_mask.sum())

    @property
    def name(self):
        return "Targeted NER goal function"
