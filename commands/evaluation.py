import json
import click

from commands.utils import (
    calculate_metrics,
    remap_negations,
    remap_negations_single
)

from commands.utils import (
    load_attacked_dataset,
    extract_attacked_dataset
)

from textattackner.models import NERModelWrapper
from textattackner.datasets import NERHuggingFaceDataset


@click.command()
@click.option("--model", required=True, type=str)
@click.option("--dataset-config", default=None, type=str)
@click.option("--mode", default=None, type=str)
@click.pass_context
def evaluate(ctx, model, dataset_config, mode):
    _, model = NERModelWrapper.load_huggingface_model(model)
    dataset = NERHuggingFaceDataset.from_config_file(dataset_config)

    # Negations must be remapped to avoid prediction errors
    dataset.dataset = remap_negations(dataset.dataset)
    labelled_dataset = [(sample[0], [dataset.label_names[x] for x in sample[1]]) for sample in dataset]

    original_str, _ = calculate_metrics(
        model,
        labelled_dataset,
        dataset.label_names,
        mode=mode)

    print()
    print(original_str)


@click.command()
@click.option("--model", type=str)
@click.option("--attacked-dataset", default=None, type=str)
@click.option("--output-filename", required=False, type=str)
@click.option("--mode", default=None, type=str)
@click.pass_context
def evaluate_attacked(ctx, model, attacked_dataset, output_filename, mode):
    _, model = NERModelWrapper.load_huggingface_model(model)
    config, input_dataset = load_attacked_dataset(
        attacked_dataset
    )

    attacked_dataset = extract_attacked_dataset(input_dataset)
    original_dataset = [
        remap_negations_single(
            sample["original_text"],
            sample.get("ground_truth_labels", []),
            str_labels=True
        ) for sample in input_dataset
    ]

    original_str, original_metrics = calculate_metrics(model, original_dataset, config["labels"], mode=mode)
    attacked_str, attacked_metrics = calculate_metrics(model, attacked_dataset, config["labels"], mode=mode)

    print("Original metrics: \n")
    print(original_str)

    print("Attacked metrics: \n")
    print(attacked_str)

    if output_filename is not None:
        with open(output_filename, "w") as out_file:
            out_file.write(json.dumps({
                "original": original_metrics,
                "attacked": attacked_metrics
            }))
