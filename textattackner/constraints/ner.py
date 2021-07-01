from textattack.constraints import Constraint

from textattackner.models import NERModelWrapper
from textattackner.datasets import NERDataset


class NonNamedEntityConstraint(Constraint):
    """
        A constraint that prevents named entities from
        being replaced
    """
    def __init__(self):
        super().__init__(compare_against_original=True)

    def _check_constraint(self, transformed_text, original_text):
        transformed_entities = self._entity_tokens(transformed_text)
        original_entities = self._entity_tokens(original_text)

        # Make sure the text contains the same named
        # entities in the same order
        return original_entities == transformed_entities

    def _entity_tokens(self, attacked_text):
        text_entities = []

        tokens = attacked_text.text.split(" ")
        ground_truth = attacked_text.attack_attrs["ground_truth"]

        for token, label in zip(tokens, ground_truth):
            if label > 0:
                text_entities.append(token)

        return text_entities
