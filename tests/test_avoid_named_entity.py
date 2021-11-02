from textattackner.utils.ner_attacked_text import NERAttackedText
from tests.fixtures import (
    # This fixture is not used directly, but if removed tests won't run
    ner_model_wrapper,
    avoid_named_entity_constraint
)


def test_avoid_named_entities(avoid_named_entity_constraint):
    sample_1 = NERAttackedText(
        "PHOENIX 3 14 .176 10 1/2",
        ground_truth=[3, 0, 0, 0, 0, 0])

    sample_1_change = sample_1.replace_word_at_index(2, "24")
    sample_1_add = sample_1.replace_word_at_index(4, "Fairfax")
    sample_1_remove = sample_1.replace_word_at_index(0, "phone")

    # Samples with token insertion
    sample_insert_entity = sample_1.insert_text_before_word_index(5, "Canada")
    sample_insert_normal = sample_1.insert_text_before_word_index(5, "74")

    # Check that samples introducing a named entity are not allowed
    assert avoid_named_entity_constraint._check_constraint(
        sample_1_add,
        sample_1
    ) == False

    assert avoid_named_entity_constraint._check_constraint(
        sample_insert_entity,
        sample_1
    ) == False

    # Check that samples removing a named entity are not allowed
    assert avoid_named_entity_constraint._check_constraint(
        sample_1_remove,
        sample_1
    ) == False

    # Check that text which does not change the prediction is allowed
    assert avoid_named_entity_constraint._check_constraint(
        sample_1_change,
        sample_1
    ) == True

    assert avoid_named_entity_constraint._check_constraint(
        sample_insert_normal,
        sample_1
    ) == True

    # Check that samples which change an entity prediction are allowed
    sample_2 = NERAttackedText(
        "Wimbledon 16 9 4 3 29 17 31",
        ground_truth=[3, 0, 0, 0, 0, 0, 0, 0])

    # Swapping 16 with 'par' changed the prediction of 'Wimbledon'
    # from B-ORG to B-MISC. This is specific to the model used in this file
    sample_2_change = sample_2.replace_word_at_index(1, "par")

    assert avoid_named_entity_constraint._check_constraint(
        sample_2_change,
        sample_2
    ) == True
