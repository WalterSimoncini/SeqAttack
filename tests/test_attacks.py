import torch
import pytest

from tests.fixtures import ner_model_wrapper, ner_tokenizer, \
    avoid_named_entity_constraint, skip_model_errors

from utils import launch_attack_conll2003, get_conll2003_labels

from textattackner.datasets import NERDataset
from textattackner.attacks import NERTextFoolerJin2019
from textattackner.goal_functions import UntargetedNERGoalFunction


def test_textfooler(
        ner_model_wrapper,
        ner_tokenizer,
        avoid_named_entity_constraint,
        skip_model_errors):
    """
        Regression test for textfooler. As long as no exceptions
        are raised this test is considered to be passing
    """
    additional_constraints = [
        skip_model_errors,
        avoid_named_entity_constraint
    ]

    dataset = NERDataset([(
        # Sample from CoNLL2003
        "-- Frankfurt Newsroom , +49 69 756525",
        torch.tensor([0, 5, 6, 0, 0, 0, 0])
    )], label_names=get_conll2003_labels())

    attack = NERTextFoolerJin2019.build(
        ner_model_wrapper,
        ner_tokenizer,
        dataset,
        UntargetedNERGoalFunction,
        additional_constraints=additional_constraints,
        use_cache=True,
        query_budget=1024,
        max_candidates=50)

    launch_attack_conll2003(
        {
            "output": "/dev/null",
            "dataset": dataset,
            "model": ner_model_wrapper,
            "max_p_mispredicted": 0.8,
            "attack_timeout": 120
        },
        attack,
        {},
        raise_exceptions=True
    )
