from textattack.constraints import Constraint
from textattackner.models.exceptions import PredictionError


class SkipModelErrors(Constraint):
    """
        A constraint that rejects texts which cause an error in the model
        inference (e.g. texts that have more tokens than out predictions)
    """
    def __init__(self, model_wrapper):
        super().__init__(compare_against_original=True)
        self.model_wrapper = model_wrapper

    def _check_constraint(self, transformed_text, original_text):
        try:
            _ = self.model_wrapper([
                transformed_text.text
            ], raise_excs=True)

            return True
        except PredictionError as ex:
            print(f"Rejected attacked text '{transformed_text.text}' due to prediction errors: {ex}")

            return False
