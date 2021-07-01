import torch
import numpy as np

from .ner_goal_function_result import NERGoalFunctionResult

from textattack.goal_functions.goal_function import GoalFunction

from textattackner.utils import get_tokens, tensor_mask


class NERGoalFunction(GoalFunction):
    """
        Base goal function that determines whether an attack on
        named entity recognition was successful
    """
    def _process_model_outputs(self, inputs, scores):
        """
            Processes and validates a list of model outputs.
        """
        return scores

    def __init__(
        self,
        model_wrapper,
        tokenizer,
        maximizable=False,
        use_cache=True,
        query_budget=float("inf"),
        model_cache_size=2 ** 20,
        min_percent_entities_mispredicted=0.5,
        ner_postprocess_func=None,
        # Array of labels
        label_names=None
    ):
        super().__init__(
            model_wrapper,
            maximizable=maximizable,
            use_cache=use_cache,
            query_budget=query_budget,
            model_cache_size=model_cache_size)

        assert ner_postprocess_func is not None, "A post-processing function is required!"

        self.ner_postprocess_func = ner_postprocess_func
        self.tokenizer = tokenizer
        self.min_percent_entities_mispredicted = min_percent_entities_mispredicted
        self.label_names = label_names

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return self._create_goal_result

    def _create_goal_result(
        self,
        attacked_text,
        raw_output,
        displayed_output,
        goal_status,
        goal_function_score,
        num_queries,
        ground_truth_output):
        """
            Utility function that creates a NER Goal function result
            with both the raw and processed model outputs
        """
        formatted_preds, _, _, _ = self._preprocess_model_output(
            raw_output,
            attacked_text)

        result = NERGoalFunctionResult(
            attacked_text,
            formatted_preds,
            displayed_output,
            goal_status,
            goal_function_score,
            self.num_queries,
            self.ground_truth_output,
            raw_output
        )

        return result

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        return int(raw_output.argmax())

    def _preprocess_model_output(self, model_output, attacked_text):
        """
            Given a raw model output and the input sample this function
            returns the predictions as a list of numeric labels, a list of
            the confidence scores for each predicted label, a binary
            mask of the named entities in the ground truth and the confidence
            of the no-entity class
        """
        named_entity_mask = tensor_mask(
            self._preprocess_ground_truth(attacked_text)
        )

        tokenized_input = get_tokens(attacked_text.text, self.tokenizer)

        _, preds, confidence_scores, all_labels_confidences = self.ner_postprocess_func(
            attacked_text.text,
            model_output,
            tokenized_input)

        return preds, confidence_scores, named_entity_mask, all_labels_confidences

    def _preprocess_ground_truth(self, attacked_text):
        return torch.tensor(attacked_text.attack_attrs["ground_truth"])

    def set_min_percent_entities_mispredicted(self, new_value):
        assert (new_value > 0 and new_value <= 1.0), "Min percent entities mispredicted should be in the interval (0, 1]"
        self.min_percent_entities_mispredicted = new_value

    def class_for_label(self, label):
        return label.replace("I-", "").replace("B-", "")
