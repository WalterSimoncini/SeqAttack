import copy

from textattack.shared.attacked_text import AttackedText
from textattack.shared.utils.strings import words_from_text

from .sequence import pad_sequence


class NERAttackedText(AttackedText):
    """
        Custom attacked text class that keeps track of a sequence
        ground truth
    """
    def __init__(self, text_input, attack_attrs=None, ground_truth=None):
        super().__init__(text_input, attack_attrs=attack_attrs)

        if "ground_truth" not in self.attack_attrs:
            self.attack_attrs["ground_truth"] = ground_truth

    def insert_text_after_word_index(self, index, text):
        return self._update_ground_truth_insert(
            super().insert_text_after_word_index(index, text)
        )

    def insert_text_before_word_index(self, index, text):
        return self._update_ground_truth_insert(
            super().insert_text_before_word_index(index, text)
        )

    def _update_ground_truth_insert(self, attacked_text):
        ground_truth_idx = self._ground_truth_inserted_index(attacked_text)

        # Insert a no-entity token in the ground truth to match the modified text
        updated_truth = copy.deepcopy(attacked_text.attack_attrs["ground_truth"])
        updated_truth.insert(ground_truth_idx, 0)

        attacked_text.attack_attrs["ground_truth"] = updated_truth

        return attacked_text

    def _ground_truth_inserted_index(self, attacked_text):
        """
            Returns the index of the last inserted
            word in the input attacked_text.
        """
        if "previous_attacked_text" not in attacked_text.attack_attrs:
            return None

        new_tokens = attacked_text.text.split(" ")
        old_tokens = attacked_text.attack_attrs["previous_attacked_text"].text.split(" ")

        if len(old_tokens) == len(new_tokens):
            return None

        # Pad the old tokens sequence with None values
        # to match the length of new_tokens
        old_tokens = pad_sequence(old_tokens, len(new_tokens), filler=None)

        for i, (new, old) in enumerate(zip(new_tokens, old_tokens)):
            if new != old:
                return i

        return None

    def generate_new_attacked_text(self, new_words):
        new_attacked_text = super().generate_new_attacked_text(new_words)

        return NERAttackedText(
            text_input=new_attacked_text.text,
            attack_attrs=new_attacked_text.attack_attrs,
            ground_truth=self.attack_attrs.get("ground_truth", None))

    def strip_remove_double_spaces(self):
        """
            Removes double spaces in text and strips
            whitespace at both ends, which may lead
            to errors downstream
        """
        self._text_input["text"] = " ".join(self._text_input["text"].split())
