import json

from seqeval.scheme import IOB2
from seqeval.metrics import classification_report

from .preprocess import remap_negations_single


def load_attacked_dataset(path):
    json_data = json.loads(
        open(path).read())

    samples_list = json_data

    if type(json_data) == dict:
        # Extract samples list from newer formats of the datasets
        samples_list = json_data["attacked_samples"]

    return json_data, samples_list


def extract_attacked_dataset(attacked_dataset):
    """
        Creates a copy of the original dataset by replacing original
        samples with their adversarial counterpart
    """
    out_attacked_dataset = []

    for sample in attacked_dataset:
        ner_labels = sample.get("original_labels", [])
        input_text = sample["attacked_sample"]

        if sample["status"] == "SuccessfulAttackResult":
            input_text = sample["perturbed_text"]

            if "final_truth_labels" in sample:
                ner_labels = sample["final_truth_labels"]

        input_text, ner_labels = remap_negations_single(
            input_text,
            ner_labels,
            str_labels=True,
            no_entity_label="O"
        )

        out_attacked_dataset.append((
            input_text, ner_labels
        ))

    return out_attacked_dataset


def calculate_metrics(model, dataset, label_names, mode=None):
    """
        Given a dataset in the format (text_input, ground_truth_labels)
        and a model this function calculates the model prediction for
        each text samples and uses seqeval to calculate the precision,
        recall and F1 metrics
    """
    ground_truths, model_predictions = [], []

    for (input_text, ground_truth_labels) in dataset:
        predicted_labels = predict_labels(
            model,
            input_text,
            label_names)

        ground_truths.append(ground_truth_labels)
        model_predictions.append(predicted_labels)

    classification_dict = classification_report(
        ground_truths,
        model_predictions,
        scheme=IOB2,
        mode=mode,
        output_dict=True)

    classification_str = classification_report(
        ground_truths,
        model_predictions,
        scheme=IOB2,
        mode=mode,
        output_dict=False)

    return classification_str, serialize_metrics_dict(classification_dict)


def serialize_metrics_dict(metrics_dict):
    """
        Converts the values of the dictionary output of seqeval
        from numpy int64/float64 to floats
    """
    out_dict = {}

    for top_k in metrics_dict.keys():
        out_dict[top_k] = {}

        for sub_k in metrics_dict[top_k].keys():
            out_dict[top_k][sub_k] = float(metrics_dict[top_k][sub_k])

    return out_dict


def predict_labels(model, sample: str, dataset_labels: list):
    """
        Predicts a single textual sample and returns a list of
        classes, one per token
    """
    prediction = model([sample])[0]
    prediction = model.process_raw_output(prediction, sample).tolist()

    return prediction_to_labels(prediction, dataset_labels)


def prediction_to_labels(prediction, labels):
    return [labels[x] for x in list(prediction)]
