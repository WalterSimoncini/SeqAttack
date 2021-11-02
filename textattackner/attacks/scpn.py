"""
BAE (BAE: BERT-Based Adversarial Examples)
============================================

"""
from textattackner.constraints import SkipNonASCII, SkipNegations
from textattackner.search import GreedySearchNER
from textattackner.utils import postprocess_ner_output

from textattack.attack_recipes import AttackRecipe
from textattackner.utils.attack import NERAttack
from textattackner.transformations import ParaphraseTransformation
from .seqattack_recipe import SeqAttackRecipe


class NERSCPNParaphrase(SeqAttackRecipe):
    """
        Adversarial Example Generation with Syntactically Controlled
        Paraphrase Networks.

        Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer.
        NAACL-HLT 2018.

        `[pdf] <https://www.aclweb.org/anthology/N18-1170.pdf>`__
        `[code] <https://github.com/miyyer/scpn>`__
    """
    @staticmethod
    def build(
            model,
            tokenizer,
            dataset,
            goal_function_class,
            additional_constraints=[],
            query_budget=2500,
            use_cache=False,
            **kwargs):
        transformation = ParaphraseTransformation()

        constraints = [
            # Only skip non-ASCII characters
            SkipNonASCII(),
            SkipNegations(),
        ]

        # Add user-provided constraints
        constraints.extend(additional_constraints)

        # FIXME: we might want to limit this to the
        # strict untargeted / targeted goal functions
        # since I <-> B swaps are easy to generate
        goal_function = goal_function_class(
            model,
            tokenizer=tokenizer,
            use_cache=use_cache,
            query_budget=query_budget,
            ner_postprocess_func=postprocess_ner_output,
            label_names=dataset.label_names)

        search_method = GreedySearchNER()

        return NERAttack(
            goal_function,
            constraints,
            transformation,
            search_method)

    @staticmethod
    def get_ner_constraints(model_name, **kwargs):
        return []
