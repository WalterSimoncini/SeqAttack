from seqattack.models import NERModelWrapper
from textattack.attack_recipes import AttackRecipe
from seqattack.constraints import (
    SkipModelErrors,
    AvoidNamedEntityConstraint,
    NonNamedEntityConstraint
)


class SeqAttackRecipe(AttackRecipe):
    @staticmethod
    def get_ner_constraints(model_name, **kwargs):
        constraints_model_wrapper = NERModelWrapper.load_huggingface_model(
            model_name=model_name
        )[1]

        return [
            SkipModelErrors(model_wrapper=constraints_model_wrapper),
            AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper),
            # Avoid modifying ground truth named entities
            NonNamedEntityConstraint()
        ]
