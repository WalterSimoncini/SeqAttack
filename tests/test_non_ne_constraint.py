import pytest

from textattackner.utils.ner_attacked_text import NERAttackedText
from textattackner.constraints import NonNamedEntityConstraint


@pytest.fixture
def non_named_entity_constraint():
    return NonNamedEntityConstraint()


def test_check_constraint(non_named_entity_constraint):
    original_text = NERAttackedText(
        "Europe has announced a new budget for education",
        ground_truth=[1, 0, 0, 0, 0, 0, 0, 0]
    )

    perturbed_no_entities = original_text.insert_text_before_word_index(7, "tertiary")
    perturbed_replace_entity = original_text.replace_word_at_index(0, "it")
    perturbed_replace_non_entity = original_text.replace_word_at_index(2, "proposed")

    assert non_named_entity_constraint._check_constraint(
        perturbed_no_entities,
        original_text) == True
    
    assert non_named_entity_constraint._check_constraint(
        perturbed_replace_entity,
        original_text) == False

    assert non_named_entity_constraint._check_constraint(
        perturbed_replace_non_entity,
        original_text) == True
