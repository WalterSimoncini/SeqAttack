from textattack.constraints import Constraint
from seqattack.utils import is_ascii


class SkipNonASCII(Constraint):
    """
        A constraint that rejects texts with non ASCII characters
    """
    def __init__(self):
        super().__init__(compare_against_original=True)

    def _check_constraint(self, transformed_text, original_text):
        return is_ascii(transformed_text.text)
