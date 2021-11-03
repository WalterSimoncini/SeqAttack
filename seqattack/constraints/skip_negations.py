from textattack.constraints import Constraint


class SkipNegations(Constraint):
    """
        A constraint that rejects texts that contain negations,
        namely "do not", any word ending in "n't" and "does not"
    """
    def __init__(self):
        super().__init__(compare_against_original=True)

    def _check_constraint(self, transformed_text, original_text):
        targets = [
            "do not",
            "n't",
            "does not"
        ]

        for target in targets:
            if target in transformed_text.text:
                return False

        return True
