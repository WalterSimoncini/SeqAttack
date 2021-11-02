from textattackner.utils import elements_diff, pad_sequence


def test_elements_diff():
    assert elements_diff(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ) == set()

    assert elements_diff(
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 3, 0]
    ) == set([2, 4])


def test_pad_sequence():
    original = [0, 1, 3, 0, 4]

    # Same length as sequence, no change expected
    assert pad_sequence(original, len(original)) == original

    # 2 pads needed
    expected = [0, 1, 3, 0, 4, 47, 47]
    assert pad_sequence(original, len(original) + 2, filler=47) == expected

    # Longer than required, so no padding needed
    assert pad_sequence(original, len(original) - 2, filler=47) == original
