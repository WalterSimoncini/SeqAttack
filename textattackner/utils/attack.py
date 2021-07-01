from textattack.shared import Attack


class NERAttack(Attack):
    def _get_transformations_uncached(self, current_text, original_text=None, **kwargs):
        transformed_texts = super()._get_transformations_uncached(
            current_text,
            original_text=original_text,
            **kwargs)

        # Remove multiple spaces from samples
        for transformed in transformed_texts:
            transformed.strip_remove_double_spaces()

        return transformed_texts
