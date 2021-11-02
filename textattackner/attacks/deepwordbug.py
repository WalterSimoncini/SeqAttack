"""

DeepWordBug
========================================
(Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)

"""

from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)

from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from textattackner.models import NERModelWrapper
from textattackner.search import NERGreedyWordSwapWIR

from textattackner.utils import postprocess_ner_output
from textattackner.utils.attack import NERAttack
from textattackner.constraints import SkipNegations
from .seqattack_recipe import SeqAttackRecipe

from textattackner.constraints import (
    SkipModelErrors,
    AvoidNamedEntityConstraint,
    NonNamedEntityConstraint
)


class NERDeepWordBugGao2018(SeqAttackRecipe):
    """
        Gao, Lanchantin, Soffa, Qi.

        Black-box Generation of Adversarial Text Sequences to Evade Deep
        Learning Classifiers.

        https://arxiv.org/abs/1801.04354
    """
    @staticmethod
    def build(
            model,
            tokenizer,
            dataset,
            goal_function_class,
            max_edit_distance=30,
            use_cache=True,
            query_budget=512,
            additional_constraints=[],
            use_all_transformations=True,
            **kwargs):
        #
        # Swap characters out from words. Choose the best of four potential transformations.
        #
        if use_all_transformations:
            # We propose four similar methods:
            transformation = CompositeTransformation(
                [
                    # (1) Swap: Swap two adjacent letters in the word.
                    WordSwapNeighboringCharacterSwap(),
                    # (2) Substitution: Substitute a letter in the word with a random letter.
                    WordSwapRandomCharacterSubstitution(),
                    # (3) Deletion: Delete a random letter from the word.
                    WordSwapRandomCharacterDeletion(),
                    # (4) Insertion: Insert a random letter in the word.
                    WordSwapRandomCharacterInsertion(),
                ]
            )
        else:
            # We use the Combined Score and the Substitution Transformer to generate
            # adversarial samples, with the maximum edit distance difference of 30
            # (ϵ = 30).
            transformation = WordSwapRandomCharacterSubstitution()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [
            RepeatModification(),
            StopwordModification(),
            SkipNegations()
        ]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant
        #
        constraints.append(LevenshteinEditDistance(max_edit_distance))
        # Add extra constraints
        constraints.extend(additional_constraints)
        #
        # Goal is untargeted classification in the original paper
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
            wir_method="unk"
        )

        return NERAttack(
            goal_function,
            constraints,
            transformation,
            search_method)

    @staticmethod
    def get_ner_constraints(model_name, **kwargs):
        preserve_named_entities = kwargs.get("preserve_named_entities", False)

        constraints_model_wrapper = NERModelWrapper.load_huggingface_model(
            model_name=model_name
        )[1]

        constraints = [
            SkipModelErrors(model_wrapper=constraints_model_wrapper)
        ]

        if preserve_named_entities:
            constraints.append(AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper))
            constraints.append(NonNamedEntityConstraint())

        return constraints
