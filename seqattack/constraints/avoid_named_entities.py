from textattack.constraints import Constraint

from seqattack.models import NERModelWrapper
from seqattack.datasets import NERDataset

from textattack.shared.attacked_text import AttackedText

from seqattack.utils import elements_diff


class AvoidNamedEntityConstraint(Constraint):
    """
        This constraint makes sure that the altered words are
        not recognized as named entities 
    """
    def __init__(self, ner_model_wrapper):
        super().__init__(compare_against_original=False)
        self.ner_model_wrapper = ner_model_wrapper

    def _check_constraint(self, transformed_text, original_text):
        # Predict named entities for the original and transformed text
        insertion_index = transformed_text._ground_truth_inserted_index(
            transformed_text
        )

        transformed_ground_truth = transformed_text.attack_attrs["ground_truth"]

        transformed_preds = self.ner_model_wrapper([
            transformed_text.text
        ])[0]

        transformed_preds = self.ner_model_wrapper.process_raw_output(
            transformed_preds,
            transformed_text.text
        ).tolist()

        if insertion_index is not None:
            return transformed_preds[insertion_index] == 0
        else:
            diff_indices = transformed_text.all_words_diff(original_text)

            for idx in list(diff_indices):
                if transformed_preds[idx] > 0:
                    # Introduced entity via entity insertion (e.g. phone --> Belgium)
                    return False
                elif transformed_preds[idx] == 0 and transformed_ground_truth[idx] > 0:
                    # Removed entity (e.g. Belgium --> phone)
                    return False

            return True
