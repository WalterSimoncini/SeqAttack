"""
MORPHEUS2020
===============
(It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)
"""
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)

from textattack.shared.attack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.transformations import WordSwapInflections

from textattackner.constraints import NonNamedEntityConstraint, SkipNegations
from textattackner.goal_functions import UntargetedNERGoalFunction

from textattackner.search import GreedySearchNER
from textattackner.utils import postprocess_ner_output

from textattackner.utils.attack import NERAttack
from .seqattack_recipe import SeqAttackRecipe


class MorpheusTan2020NER(SeqAttackRecipe):
    """
        Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher.
        It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations
        https://www.aclweb.org/anthology/2020.acl-main.263/
    """
    @staticmethod
    def build(
            model,
            tokenizer,
            dataset,
            goal_function_class,
            use_cache=True,
            query_budget=512,
            additional_constraints=[],
            attack_timeout=30,
            **kwargs):
        goal_function = goal_function_class(
            model,
            tokenizer,
            use_cache=use_cache,
            query_budget=query_budget,
            ner_postprocess_func=postprocess_ner_output,
            label_names=dataset.label_names)

        # Swap words with their inflections
        transformation = WordSwapInflections()

        # The POS mapping has some compatibility issues with the POS
        # output of AttackedText(s). Add these mappings to patch the
        # issue
        transformation._enptb_to_universal["ADJ"] = "ADJ"
        transformation._enptb_to_universal["NOUN"] = "NOUN"
        transformation._enptb_to_universal["VERB"] = "VERB"

        constraints = [
            # Do not modify already changed words
            RepeatModification(),
            # Do not modify stopwords
            StopwordModification(),
            SkipNegations()]

        constraints.extend(additional_constraints)

        # Greedily swap words (see pseudocode, Algorithm 1 of the paper).
        search_method = GreedySearchNER()

        return NERAttack(
            goal_function,
            constraints,
            transformation,
            search_method,
            attack_timeout=attack_timeout)
