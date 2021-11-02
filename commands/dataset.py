import json
import click
import numpy as np
from tqdm import tqdm

from commands.utils import remap_negations

from textattackner.models import NERModelWrapper
from textattackner.datasets import NERHuggingFaceDataset
from textattackner.goal_functions import get_goal_function
from textattackner.utils import postprocess_ner_output
from textattackner.utils.ner_attacked_text import NERAttackedText


@click.command()
@click.option("--model", required=True, type=str)
@click.option("--dataset-config", default=None, type=str)
@click.option("--max-samples", required=True, type=int)
@click.option("--max-initial-score", required=True, type=float)
@click.option("--output-filename", required=True, type=str)
@click.option("--goal-function", default="untargeted")
@click.pass_context
def pick_samples(
        ctx,
        model,
        dataset_config,
        max_samples,
        max_initial_score,
        output_filename,
        goal_function):
    """
        Extracts a subset of samples from a dataset with a
        given maximum initial misprediction score
    """
    tokenizer, model = NERModelWrapper.load_huggingface_model(model)
    dataset = NERHuggingFaceDataset.from_config_file(dataset_config)
    
    goal_function_cls = get_goal_function(goal_function)
    goal_function = goal_function_cls(
        model_wrapper=model,
        tokenizer=tokenizer,
        use_cache=True,
        ner_postprocess_func=postprocess_ner_output,
        label_names=dataset.label_names
    )

    # Negations must be remapped to avoid prediction errors
    dataset.dataset = remap_negations(dataset.dataset)

    progress_bar = tqdm(total=max_samples)
    selected_samples, initial_scores = [], []

    for sample, ground_truth in dataset.dataset:
        if len(selected_samples) >= max_samples:
            break

        attacked_text = NERAttackedText(
            sample,
            attack_attrs={"label_names": dataset.label_names},
            ground_truth=ground_truth.tolist())

        try:
            goal_function.init_attack_example(attacked_text, ground_truth)

            model_raw = model([sample])[0]
            model_preds = model.process_raw_output(model_raw, attacked_text.text)

            initial_score = goal_function._get_score_labels(model_preds, attacked_text)
        except Exception as ex:
            print(f"Error scoring {sample}: {ex}")

        if initial_score <= max_initial_score:
            progress_bar.update(1)
            initial_scores.append(initial_score)
            selected_samples.append((sample, ground_truth.tolist()))

    # Save samples
    with open(output_filename, "w") as out_file:
        out_file.write(json.dumps({
            "meta": {
                "max_samples": max_samples,
                "max_initial_score": max_initial_score,
                "dataset": dataset.name,
                "split": dataset.split,
                "dataset_labels": dataset.label_names
            },
            "samples": selected_samples
        }))

    print(f"Selected {len(selected_samples)} from {dataset.name}")
    print(f"Average misprediction score: {np.array(initial_scores).mean()}")
