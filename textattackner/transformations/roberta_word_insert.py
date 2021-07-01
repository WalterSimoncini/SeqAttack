from textattack.transformations import WordInsertionMaskedLM


class RoBERTaWordInsertionMaskedLM(WordInsertionMaskedLM):
    def _get_transformations(self, current_text, indices_to_modify):
        indices_to_modify = list(indices_to_modify)
        new_words = self._get_new_words(current_text, indices_to_modify)

        transformed_texts = []

        for i in range(len(new_words)):
            index_to_modify = indices_to_modify[i]
            word_at_index = current_text.words[index_to_modify]

            for word in new_words[i]:
                if word != word_at_index:
                    transformed_texts.append(
                        current_text.insert_text_before_word_index(
                            # Fix RoBERTa BPE artifacts
                            index_to_modify, word.replace("Ġ", "")
                        )
                    )

        return transformed_texts
