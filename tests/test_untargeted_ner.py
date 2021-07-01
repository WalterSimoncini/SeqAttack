import json
import torch
import pytest

from utils import get_conll2003_labels

from textattackner.utils.ner_attacked_text import NERAttackedText

from tests.utils import numeric_string_to_tensor
from tests.fixtures import ner_model_wrapper, ner_tokenizer

from textattackner.utils import postprocess_ner_output
from textattackner.goal_functions import UntargetedNERGoalFunction


@pytest.fixture(scope="module")
def untargeted_ner_goal_function(ner_model_wrapper, ner_tokenizer):
    return UntargetedNERGoalFunction(
        model_wrapper=ner_model_wrapper,
        tokenizer=ner_tokenizer,
        min_percent_entities_mispredicted=0.5,
        ner_postprocess_func=postprocess_ner_output,
        label_names=get_conll2003_labels())


def test_ner_goal_function_get_score_confidence_example(untargeted_ner_goal_function):
    json_data = json.loads(
        open("tests/mock/attacked.json").read()
    )

    ground_truth = numeric_string_to_tensor(json_data["ground_truth"])
    attacked_text = NERAttackedText(
        json_data["attacked_sample"],
        ground_truth=[int(x) for x in ground_truth])

    perturbed_pred = numeric_string_to_tensor(json_data["perturbed_pred"])

    perturbed_model_pred = torch.tensor(json_data["raw_pred"])

    untargeted_ner_goal_function.init_attack_example(
        attacked_text,
        ground_truth)

    # Only one entity is changed in the sample, thus we are
    # happy with a 25% misprediction
    untargeted_ner_goal_function.set_min_percent_entities_mispredicted(0.25)

    assert untargeted_ner_goal_function._is_goal_complete(
        perturbed_model_pred,
        attacked_text
    ) == True

    assert round(untargeted_ner_goal_function._get_score_confidence(
        perturbed_model_pred,
        attacked_text
    ), 2) == 0.28


def test_ner_goal_function_get_score_confidence(untargeted_ner_goal_function):
    attacked_text = NERAttackedText(
        "The Netherlands is nice",
        ground_truth=[1, 1, 0, 0])
    ground_truth = torch.tensor([1, 1, 0, 0])
    untargeted_ner_goal_function.set_min_percent_entities_mispredicted(0.5)

    model_ground_truth = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    model_valid_misprediction = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    model_confidence_misprediction = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.5, 0, 0, 0, 0, 0, 0, 0.5],
        [0, 0.5, 0, 0, 0, 0, 0, 0, 0.5],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    untargeted_ner_goal_function.init_attack_example(
        attacked_text,
        ground_truth)

    # Same sample, full confidence. so 0% difference expected
    assert round(untargeted_ner_goal_function._get_score_confidence(
        model_ground_truth,
        attacked_text
    ), 2) == 0

    # One misprediction, full confidence. so 50% difference expected
    assert round(untargeted_ner_goal_function._get_score_confidence(
        model_valid_misprediction,
        attacked_text
    ), 2) == 0.50

    # The 50% misprediction should result in a successful attack
    assert untargeted_ner_goal_function._is_goal_complete(
        model_valid_misprediction,
        attacked_text
    ) == True

    # No misprediction, two entities with 50% confidence,
    # so we expect a 50% score
    assert round(untargeted_ner_goal_function._get_score_confidence(
        model_confidence_misprediction,
        attacked_text
    ), 2) == 0.50

    # But since no entity is actually mispredicted the
    # goal should NOT be complete
    assert untargeted_ner_goal_function._is_goal_complete(
        model_confidence_misprediction,
        attacked_text
    ) == False


def test_ner_goal_function_get_score_labels(untargeted_ner_goal_function):
    attacked_text = NERAttackedText(
        "5. Jacqui Cooper ( Australia ) 156.52",
        ground_truth=[0, 3, 4, 0, 7, 0, 0])

    ground_truth = torch.tensor([0, 3, 4, 0, 7, 0, 0])
    valid_misprediction = torch.tensor([0, 1, 1, 0, 7, 0, 0])

    untargeted_ner_goal_function.init_attack_example(
        attacked_text,
        ground_truth)

    # Same sample, so 0% difference expected
    assert round(untargeted_ner_goal_function._get_score_labels(
        ground_truth,
        attacked_text
    ), 2) == 0

    # 66% altered tensor, _get_score should return 0.67 (considering rounding)
    assert round(untargeted_ner_goal_function._get_score_labels(
        valid_misprediction,
        attacked_text
    ), 2) == 0.67

    # Make sure that if an entity is introduced in a
    # no-entities sample the returned score will be 1
    ground_truth_empty = torch.tensor([0, 0, 0, 0, 0, 0, 0])
    add_entity_misprediction = torch.tensor([0, 1, 0, 0, 0, 0, 0])

    untargeted_ner_goal_function.init_attack_example(
        attacked_text,
        ground_truth_empty)

    assert round(untargeted_ner_goal_function._get_score_labels(
        add_entity_misprediction,
        attacked_text
    ), 2) == 1
