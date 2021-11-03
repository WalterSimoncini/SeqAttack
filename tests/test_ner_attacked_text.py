import pytest

from seqattack.utils.ner_attacked_text import NERAttackedText


def test_ground_truth_inserted_index():
    """
        Verifies that the function returns
        the correct last inserted token index
    """
    attacked_text = NERAttackedText(
        "Germany published the new bond rates",
        ground_truth=[1, 0, 2, 0, 0, 0])

    insert_middle = attacked_text.insert_text_after_word_index(1, "today")
    insert_beginning = attacked_text.insert_text_before_word_index(0, "Today")
    insert_end = attacked_text.insert_text_after_word_index(5, "specifications")

    replace_word = attacked_text.replace_word_at_index(2, "Lorem")

    assert insert_middle._ground_truth_inserted_index(
        insert_middle) == 2

    assert insert_beginning._ground_truth_inserted_index(
        insert_beginning) == 0

    assert insert_end._ground_truth_inserted_index(
        insert_end) == 6

    # Make sure that if no token was inserted the returned index is None
    assert attacked_text._ground_truth_inserted_index(attacked_text) is None
    assert replace_word._ground_truth_inserted_index(replace_word) is None

    # Make sure the ground truths were updated correctly
    assert insert_middle.attack_attrs["ground_truth"] == [1, 0, 0, 2, 0, 0, 0]
    assert insert_beginning.attack_attrs["ground_truth"] == [0, 1, 0, 2, 0, 0, 0]
    assert insert_end.attack_attrs["ground_truth"] == [1, 0, 2, 0, 0, 0, 0]
