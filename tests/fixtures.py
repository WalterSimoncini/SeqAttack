import pytest

from transformers import AutoTokenizer
from seqattack.models import NERModelWrapper

from seqattack.constraints import AvoidNamedEntityConstraint


@pytest.fixture(scope="module")
def ner_model_wrapper():
    return NERModelWrapper.load_huggingface_model("dslim/bert-base-NER")[1]


@pytest.fixture(scope="module")
def ner_tokenizer():
    return AutoTokenizer.from_pretrained("dslim/bert-base-NER")


@pytest.fixture(scope="module")
def conll2003_labels():
    return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


@pytest.fixture()
def avoid_named_entity_constraint(ner_model_wrapper):
    return AvoidNamedEntityConstraint(ner_model_wrapper=ner_model_wrapper)
