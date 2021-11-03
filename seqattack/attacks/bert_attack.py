from textattack.transformations import WordSwapMaskedLM
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification

from seqattack.constraints import SkipNonASCII, SkipNegations
from seqattack.search import NERGreedyWordSwapWIR
from seqattack.utils import postprocess_ner_output
from seqattack.utils.attack import NERAttack
from .seqattack_recipe import SeqAttackRecipe


class BertAttackNER(SeqAttackRecipe):
    """
        Li, L.., Ma, R., Guo, Q., Xiangyang, X., Xipeng, Q. (2020).
        BERT-ATTACK: Adversarial Attack Against BERT Using BERT

        https://arxiv.org/abs/2004.09984

        This is "attack mode" 1 from the paper, BAE-R, word replacement.
        This code is heavily based on: (refer to it for detailed documentation)

        https://textattack.readthedocs.io/en/latest/_modules/textattack/attack_recipes/bert_attack_li_2020.html#BERTAttackLi2020
    """
    @staticmethod
    def build(
            model,
            tokenizer,
            dataset,
            goal_function_class,
            max_perturbed_percent=0.4,
            max_candidates=48,
            additional_constraints=[],
            query_budget=2500,
            use_cache=False,
            max_entities_mispredicted=0.8,
            attack_timeout=30,
            **kwargs):
        transformation = WordSwapMaskedLM(
            method="bert-attack",
            max_candidates=max_candidates)

        constraints = [
            # Do not modify already changed words
            RepeatModification(),
            # Do not modify stopwords
            StopwordModification(),
            SkipNonASCII(),
            SkipNegations(),
            MaxWordsPerturbed(max_percent=max_perturbed_percent)]

        constraints.extend(additional_constraints)

        use_constraint = UniversalSentenceEncoder(
            threshold=0.2,
            metric="cosine",
            compare_against_original=True,
            window_size=None)
        constraints.append(use_constraint)

        goal_function = goal_function_class(
            model,
            tokenizer=tokenizer,
            use_cache=use_cache,
            query_budget=query_budget,
            ner_postprocess_func=postprocess_ner_output,
            label_names=dataset.label_names)

        # Select words with the most influence on output logits
        # search_method = GreedyWordSwapWIR(wir_method="unk")
        search_method = NERGreedyWordSwapWIR(
            dataset=dataset,
            tokenizer=tokenizer,
            wir_method="unk")

        return NERAttack(
            goal_function,
            constraints,
            transformation,
            search_method,
            attack_timeout=attack_timeout,
            max_entities_mispredicted=max_entities_mispredicted)
