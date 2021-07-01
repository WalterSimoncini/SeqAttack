import transformers

from textattack.shared.attack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.transformations import WordSwapMaskedLM, CompositeTransformation
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification

from textattackner.utils import postprocess_ner_output
from textattackner.transformations import RoBERTaWordInsertionMaskedLM
from textattackner.constraints import NonNamedEntityConstraint, SkipNonASCII, SkipNegations
from textattackner.search import GreedySearchNER
from textattackner.utils.attack import NERAttack


class NERCLARE(AttackRecipe):
    """
        Li, Zhang, Peng, Chen, Brockett, Sun, Dolan.

        "Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)

        https://arxiv.org/abs/2009.07502

        This method uses greedy search with replace, merge, and insertion transformations that leverage a
        pretrained language model. It also uses USE similarity constraint.
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
        use_cache=False):

        shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained("distilroberta-base")
        shared_tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

        transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-4,
                ),
                RoBERTaWordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                )
            ]
        )

        constraints = [
            # Do not modify already changed words
            RepeatModification(),
            # Do not modify stopwords
            StopwordModification(),
            SkipNonASCII(),
            SkipNegations(),
            # Avoid modifying ground truth named entities
            NonNamedEntityConstraint()
        ]

        constraints.extend(additional_constraints)

        use_constraint = UniversalSentenceEncoder(
            threshold=0.7,
            metric="cosine",
            compare_against_original=True,
            # The original implementation uses a window of 15 and skips
            # samples shorter than that window
            window_size=None)

        constraints.append(use_constraint)

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
