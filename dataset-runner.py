import json
import random
import click

from utils import extract_attacked_dataset, \
    get_conll2003, calculate_metrics, \
    load_attacked_dataset

from textattackner.models import NERModelWrapper
from textattackner.utils import postprocess_ner_output

from utils.model import get_ner_model
from utils import calculate_metrics, calculate_metrics, remap_negations

from transformers import AutoTokenizer, AutoModelForTokenClassification

from textattackner.utils.ner_attacked_text import NERAttackedText
from textattackner.goal_functions import UntargetedNERGoalFunction


@click.group()
@click.option("--dataset", required=True, type=str)
@click.pass_context
def dataset_runner(ctx, dataset):
    """
        Global setup
    """
    # ensure that ctx.attack exists and is a dict
    ctx.ensure_object(dict)
    ctx.obj["dataset_name"] = dataset


@dataset_runner.command()
@click.option("--random-seed", default=567, type=int)
@click.option("--split", required=True, type=str)
@click.option("--max-samples", required=True, type=int)
@click.option("--max-initial-score", required=True, type=float)
@click.option("--output-filename", required=True, type=str)
@click.pass_context
def pick_samples(
        ctx,
        random_seed,
        split,
        max_samples,
        max_initial_score,
        output_filename):
    """
        Extract a subset of samples from CoNLL2003 with maximum initial
        misprediction score
    """
    # Set random seed
    random.seed(random_seed)

    dataset_loaders = {
        "conll2003": lambda: get_conll2003(max_samples=None, split=split, shuffle=True)[0]
    }

    dataset_name = ctx.obj["dataset_name"]
    assert dataset_name in list(dataset_loaders.keys())

    dataset = dataset_loaders[dataset_name]()

    # Handle negations
    dataset.dataset = remap_negations(dataset.dataset)

    # Load the model
    tokenizer, model = get_ner_model()

    # Select only samples with an initial score < 0.5
    goal_function = UntargetedNERGoalFunction(
        model,
        tokenizer=tokenizer,
        use_cache=True,
        ner_postprocess_func=postprocess_ner_output,
        label_names=dataset.label_names
    )

    score_filtered_dataset = []

    for sample, ground_truth in dataset.dataset:
        list_ground_truth = [int(x) for x in ground_truth]

        attacked_text = NERAttackedText(
            sample,
            attack_attrs={
                "label_names": dataset.label_names
            },
            ground_truth=list_ground_truth)

        try:
            goal_function.init_attack_example(
                attacked_text,
                ground_truth)

            model_raw = model([sample])[0]
            model_preds = model.process_raw_output(
                model_raw,
                attacked_text.text
            )

            initial_score = goal_function._get_score_labels(
                model_preds,
                attacked_text)
        except Exception as ex:
            print(f"Error scoring {sample}: {ex}")

        if initial_score <= max_initial_score:
            score_filtered_dataset.append(
                (sample, list_ground_truth)
            )

            if len(score_filtered_dataset) > max_samples:
                # Terminate when we found enough samples
                break

    with open(output_filename, "w") as out_file:
        out_file.write(json.dumps({
            "meta": {
                "max_samples": max_samples,
                "max_initial_score": max_initial_score,
                "dataset": dataset_name,
                "random_seed": random_seed,
                "split": split,
                "dataset_labels": dataset.label_names
            },
            "samples": score_filtered_dataset
        }))


if __name__ == '__main__':
    dataset_runner(obj={})
