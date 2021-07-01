import pytest

from utils import get_conll2003_labels

from textattack.shared import AttackedText

from tests.fixtures import ner_model_wrapper, ner_tokenizer

from textattackner.utils import postprocess_ner_output
from textattackner.goal_functions.untargeted_ner_strict import StrictUntargetedNERGoalFunction


@pytest.fixture(scope="module")
def strict_untargeted_ner_goal_function(ner_model_wrapper, ner_tokenizer):
    return StrictUntargetedNERGoalFunction(
        model_wrapper=ner_model_wrapper,
        tokenizer=ner_tokenizer,
        min_percent_entities_mispredicted=0.5,
        ner_postprocess_func=postprocess_ner_output,
        label_names=get_conll2003_labels())


def test_score_per_token(strict_untargeted_ner_goal_function):
    # Same class and numeric label, 0.9 confidence --> 0.1 expected
    assert round(strict_untargeted_ner_goal_function._score_per_token(
        1, "I-PER",
        {
            1: 0.9
        },
        1, "I-PER"), 2) == 0.1

    # Same class, different numeric label, 0.1 ground truth confidence
    assert round(strict_untargeted_ner_goal_function._score_per_token(
        1, "B-PER",
        {
            1: 0.2,
            2: 0.7,
            3: 0.1
        },
        2, "I-PER"), 2) == 0.3

    # Different numeric label and class, any confidence. 1 expected
    assert round(strict_untargeted_ner_goal_function._score_per_token(
        1, "I-PER",
        {
            1: 0.1,
            3: 0.9
        },
        3, "I-LOC"), 2) == 1
