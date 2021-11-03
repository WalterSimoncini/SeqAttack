import torch


def remap_negations(dataset):
    """
        Given a dataset maps negations to the form
        root-n ' t (e.g. do not --> don ' t,
        does n't --> doesn ' t)
    """
    out_dataset = []

    for sample_idx in range(len(dataset)):
        sample_text, sample_truth = dataset[sample_idx]

        out_dataset.append(
            remap_negations_single(sample_text, sample_truth)
        )

    return out_dataset


def remap_negations_single(text, truth_labels, str_labels=False):
    def fix_negation(i, words, labels):
        current_word = words[i]

        words[i] = f"{current_word}n"
        words[i + 1] = "'"
        words.insert(i + 2, "t")

        labels[i] = 0
        labels[i + 1] = 0
        labels.insert(i + 2, 0)

    if "do not" in text or "does not" in text or "n't" in text:
        if str_labels:
            words, labels = text.split(" "), truth_labels
        else:
            words, labels = text.split(" "), [int(x) for x in truth_labels]

        for i in range(len(words) - 1):
            current_word, next_word = words[i], words[i + 1]
            start_word_match = current_word in ["do"]

            if start_word_match and (next_word == "not"):
                fix_negation(i, words, labels)
            elif next_word == "n't":
                fix_negation(i, words, labels)

        return (
            " ".join(words),
            labels if str_labels else torch.tensor(labels)
        )

    return (
        text,
        truth_labels if str_labels else torch.tensor(list(truth_labels))
    )
