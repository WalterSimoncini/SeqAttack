"""

TextFooler (Is BERT Really Robust?)
===================================================
A Strong Baseline for Natural Language Attack on Text Classification and Entailment)

"""

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding

from textattack.attack_recipes import AttackRecipe

from textattackner.search import NERGreedyWordSwapWIR
from textattackner.utils import postprocess_ner_output
from textattackner.constraints import NonNamedEntityConstraint, SkipNonASCII, \
    SkipNegations

from textattackner.utils.attack import NERAttack
from .seqattack_recipe import SeqAttackRecipe


class NERTextFoolerJin2019(SeqAttackRecipe):
    """
        Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019).

        Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment.

        https://arxiv.org/abs/1907.11932
    """

    @staticmethod
    def build(
            model,
            tokenizer,
            dataset,
            goal_function_class,
            max_candidates=50,
            additional_constraints=[],
            query_budget=2500,
            use_cache=False,
            attack_timeout=30,
            **kwargs):
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=max_candidates)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [
            RepeatModification(), 
            StopwordModification(stopwords=stopwords),
            SkipNonASCII(),
            SkipNegations()]

        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )

        constraints.append(use_constraint)
        constraints.extend(additional_constraints)

        #
        # Goal is untargeted classification
        #
        goal_function = goal_function_class(
            model,
            tokenizer=tokenizer,
            use_cache=use_cache,
            query_budget=query_budget,
            ner_postprocess_func=postprocess_ner_output,
            label_names=dataset.label_names)

        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = NERGreedyWordSwapWIR(
            dataset=dataset,
            tokenizer=tokenizer,
            wir_method="delete")

        return NERAttack(
            goal_function,
            constraints,
            transformation,
            search_method,
            attack_timeout=attack_timeout,)
