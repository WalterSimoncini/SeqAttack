import pytest
import torch

from utils import get_conll2003_labels

from textattackner.utils.ner_attacked_text import NERAttackedText

from tests.fixtures import ner_model_wrapper, ner_tokenizer

from textattackner.utils import postprocess_ner_output
from textattackner.goal_functions import TargetedNERGoalFunction


@pytest.fixture(scope="module")
def targeted_ner_goal_function(ner_model_wrapper, ner_tokenizer):
    return TargetedNERGoalFunction(
        model_wrapper=ner_model_wrapper,
        tokenizer=ner_tokenizer,
        min_percent_entities_mispredicted=0.5,
        ner_postprocess_func=postprocess_ner_output,
        label_names=get_conll2003_labels())


def test_get_score_confidence(targeted_ner_goal_function):
    attacked_text = NERAttackedText(
        "The Netherlands is nice",
        ground_truth=[1, 1, 0, 0])

    ground_truth = torch.tensor([1, 1, 0, 0])
    targeted_ner_goal_function.set_min_percent_entities_mispredicted(0.5)

    model_ground_truth = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Invalid misprediction - class change but not to "O"!
    model_invalid_misprediction = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Partial misprediction, 50% weight on O for first entity-token
    model_invalid_misprediction_probs = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0, 0, 0.5, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Partial misprediction, 60% weight on O for first entity-token
    model_valid_misprediction_probs = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.6, 0, 0, 0.4, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # One entity token disappeared
    model_valid_misprediction = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    targeted_ner_goal_function.init_attack_example(
        attacked_text,
        ground_truth)

    # Same sample, full confidence. so 0% difference expected
    assert round(targeted_ner_goal_function._get_score_confidence(
        model_ground_truth,
        attacked_text
    ), 2) == 0

    # Goal should be false, we're testing against the ground truth
    assert targeted_ner_goal_function._is_goal_complete(
        model_ground_truth,
        attacked_text
    ) == False

    # Same sample, full confidence but wrong non-O class so 0% difference expected
    assert round(targeted_ner_goal_function._get_score_confidence(
        model_invalid_misprediction,
        attacked_text
    ), 2) == 0
    
    # Goal should be false, wrong destination class
    assert targeted_ner_goal_function._is_goal_complete(
        model_invalid_misprediction,
        attacked_text
    ) == False

    # Same sample, 50% confidence on O for one class, thus we expect 25% score (50 / 2)
    assert round(targeted_ner_goal_function._get_score_confidence(
        model_invalid_misprediction_probs,
        attacked_text
    ), 2) == 0.25

    # Goal should be true (50% probability to O)
    assert targeted_ner_goal_function._is_goal_complete(
        model_invalid_misprediction_probs,
        attacked_text
    ) == True

    # Same sample, 60% confidence on O for one class, thus we expect 30% score (60 / 2)
    assert round(targeted_ner_goal_function._get_score_confidence(
        model_valid_misprediction_probs,
        attacked_text
    ), 2) == 0.30
    
    # Goal should be true (O has an higher confidence than the class token)
    assert targeted_ner_goal_function._is_goal_complete(
        model_valid_misprediction_probs,
        attacked_text
    ) == True

    # "Removed" 1/2 entity tokens, 50% score expected
    assert round(targeted_ner_goal_function._get_score_confidence(
        model_valid_misprediction,
        attacked_text
    ), 2) == 0.5

    # 50% misprediction score, so the goal should be complete
    assert targeted_ner_goal_function._is_goal_complete(
        model_valid_misprediction,
        attacked_text
    ) == True
