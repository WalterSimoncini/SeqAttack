import string
import numpy as np

from textattackner.datasets import NERDataset
from textattack.search_methods import GreedyWordSwapWIR
from textattackner.utils import get_tokens


class NERGreedyWordSwapWIR(GreedyWordSwapWIR):
    """
        This search strategy works in the same fashion as
        its superclass, with the addition that entity tokens
        (identified using the provided dataset) are not considered
        for the search (a.k.a. they are immutable)
    """
    def __init__(self, dataset: NERDataset, tokenizer, wir_method="unk", max_subtokens=3):
        # max_subtokens should be kept under 4 to avoid a huge search space
        super().__init__(wir_method=wir_method)

        self.tokenizer = tokenizer
        self.dataset = dataset.to_dict()
        self.max_subtokens = max_subtokens

    def _get_index_order(self, initial_text):
        index_order, search_over = None, None

        if self.wir_method == "delete":
            # If the WIR method is delete make sure we do not leave
            # double spaces
            leave_one_texts = [
                initial_text.delete_word_at_index(i)
                for i in range(len(initial_text.words))
            ]

            for text in leave_one_texts:
                # Remove double spaces and trailing/leading whitespaces
                text.strip_remove_double_spaces()

            leave_one_results, search_over = self.get_goal_results(
                leave_one_texts
            )

            index_scores = np.array(
                [result.score for result in leave_one_results]
            )

            index_order = (-index_scores).argsort()
        else:
            # index order refers to the word index
            index_order, search_over = super()._get_index_order(initial_text)

        # Filter out words which have too many subtokens. This is required
        # to keep the search space reasonably small
        index_order = self._filter_subtokenized_words(
            initial_text,
            index_order)

        return index_order, search_over

    def _filter_subtokenized_words(self, initial_text, candidate_words_indices):
        """
            Given an AttackedText and a list of candidate words (for
            replacement) filters out the words that are split by the
            tokenizer in more than max_subtokens sub-tokens
        """
        candidate_words = [w for i, w in enumerate(initial_text.words) if i in candidate_words_indices]
        # Tokenize each word and remov the [CLS] and [SEP] start/end tokens
        candidate_words_tokens = [get_tokens(w, self.tokenizer)[1:-1] for w in candidate_words]
        
        filtered_indices = []

        for tokens, index in zip(candidate_words_tokens, candidate_words_indices):
            if len(tokens) <= self.max_subtokens:
                filtered_indices.append(index)

        return filtered_indices

    def _get_entity_indices(self, initial_text):
        """
            Returns a list of indices representing the entity
            token in a given text. The text must exists in the
            dataset this class was initialized with
        """
        entities_indices = []

        tokenized_text = initial_text.text.split(" ")
        ground_truth_labels = self.dataset[initial_text.text]

        # Keep track of how many punctuation symbols were skipped
        puntuaction_delta = 0

        for i, label in enumerate(ground_truth_labels):
            # Select any non-zero labels, which correspond to entities
            if label > 0:
                entities_indices.append(i - puntuaction_delta)
            elif tokenized_text[i] in string.punctuation:
                # Skip punctuation symbols.
                # FIXME: We should process tokens in the same way the
                # `words` property does on AttackedText instances
                puntuaction_delta += 1

        return entities_indices
