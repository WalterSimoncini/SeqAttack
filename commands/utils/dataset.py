import json

from .preprocess import remap_negations_single


def load_attacked_dataset(path):
    json_data = json.loads(
        open(path).read())

    samples_list = json_data

    if type(json_data) == dict:
        # Extract samples list from newer formats of the datasets
        samples_list = json_data["attacked_examples"]

    return json_data["config"], samples_list


def extract_attacked_dataset(attacked_dataset):
    """
        Creates a copy of the original dataset by replacing original
        samples with their adversarial counterpart
    """
    out_attacked_dataset = []

    for sample in attacked_dataset:
        ner_labels = sample.get("ground_truth_labels", [])
        input_text = sample["original_text"]

        if sample["status"] == "Successful":
            input_text = sample["perturbed_text"]

            if "final_ground_truth_labels" in sample:
                ner_labels = sample["final_ground_truth_labels"]

        input_text, ner_labels = remap_negations_single(
            input_text,
            ner_labels,
            str_labels=True
        )

        out_attacked_dataset.append((
            input_text, ner_labels
        ))

    return out_attacked_dataset
