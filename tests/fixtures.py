import pytest

from textattackner.models import NERModelWrapper
from transformers import AutoTokenizer, AutoModelForTokenClassification

from textattackner.utils import postprocess_ner_output
from textattackner.constraints import AvoidNamedEntityConstraint, \
    SkipModelErrors


@pytest.fixture(scope="module")
def ner_model_wrapper():
    return NERModelWrapper(
        model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"),
        tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER"),
        postprocess_func=postprocess_ner_output)


@pytest.fixture(scope="module")
def ner_tokenizer():
    return AutoTokenizer.from_pretrained("dslim/bert-base-NER")


@pytest.fixture(scope="module")
def avoid_named_entity_constraint(ner_model_wrapper):
    return AvoidNamedEntityConstraint(
        ner_model_wrapper=ner_model_wrapper)


@pytest.fixture(scope="module")
def skip_model_errors(ner_model_wrapper):
    return SkipModelErrors(
        model_wrapper=ner_model_wrapper)
