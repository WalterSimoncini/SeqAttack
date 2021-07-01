import json
import click

from utils import extract_attacked_dataset, \
    get_conll2003_labels, calculate_metrics, \
    load_attacked_dataset, remap_negations_single

from textattackner.models import NERModelWrapper
from textattackner.utils import postprocess_ner_output

from utils import calculate_metrics

from transformers import AutoTokenizer, AutoModelForTokenClassification


@click.group()
@click.option("--output-filename", required=True, type=str)
@click.option("--attacked-dataset-filename", type=str, required=True)
@click.pass_context
def metrics_runner(
        ctx,
        output_filename,
        attacked_dataset_filename):
    """
        Global setup
    """
    # ensure that ctx.attack exists and is a dict
    ctx.ensure_object(dict)

    # Load the dataset
    input_json, input_dataset = load_attacked_dataset(
        attacked_dataset_filename
    )

    attacked_dataset = extract_attacked_dataset(input_dataset)

    original_dataset = [
        remap_negations_single(
            sample["attacked_sample"],
            sample.get("original_labels", []),
            str_labels=True,
            no_entity_label="O"
        ) for sample in input_dataset
    ]

    ctx.obj = {
        "output_filename": output_filename,
        "attacked_dataset": attacked_dataset,
        "original_dataset": original_dataset,
        "dataset_json": input_json
    }


@metrics_runner.command()
@click.option("--mode", type=str, default=None)
@click.pass_context
def get_metrics(ctx, mode):
    model = NERModelWrapper(
        AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"),
        AutoTokenizer.from_pretrained("dslim/bert-base-NER"),
        postprocess_func=postprocess_ner_output)

    original_str, original_metrics = calculate_metrics(
        model,
        ctx.obj["original_dataset"],
        get_conll2003_labels(),
        mode=mode)

    attacked_str, attacked_metrics = calculate_metrics(
        model,
        ctx.obj["attacked_dataset"],
        get_conll2003_labels(),
        mode=mode)

    print("Original metrics: \n")
    print(original_str)

    print("Attacked metrics: \n")
    print(attacked_str)

    ctx.obj["dataset_json"]["metrics"] = {
        "original": original_metrics,
        "attacked": attacked_metrics
    }

    with open(ctx.obj["output_filename"], "w") as out_dataset_file:
        out_dataset_file.write(
            json.dumps(ctx.obj["dataset_json"])
        )


@metrics_runner.command()
@click.option("--ground-truth-field", type=str, default="ground_truth")
@click.option("--labels-field", type=str, default="original_labels")
@click.pass_context
def add_labels(ctx, ground_truth_field, labels_field):
    """
        Adds CoNLL2003 class labels to an attacked dataset
    """
    label_names = get_conll2003_labels()

    for sample in ctx.obj["dataset_json"]["attacked_samples"]:
        if labels_field not in sample:
            if ground_truth_field not in sample:
                continue

            truth = sample[ground_truth_field]

            if type(truth) == str:
                truth = [int(x) for x in truth.split(" ")]

            sample[labels_field] = [
                label_names[x] for x in truth
            ]

    with open(ctx.obj["output_filename"], "w") as out_dataset_file:
        out_dataset_file.write(
            json.dumps(ctx.obj["dataset_json"])
        )


if __name__ == '__main__':
    metrics_runner(obj={})
