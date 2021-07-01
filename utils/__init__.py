import copy
import time
import json
import torch
import signal
import random
import numpy as np
from tqdm import tqdm

from pprint import pprint

from pygit2 import Repository
from pygit2 import GIT_SORT_TOPOLOGICAL

from datasets import load_dataset

from textattackner.datasets import NERDataset
from textattack.shared.attacked_text import AttackedText
from textattackner.utils.ner_attacked_text import NERAttackedText

from .metrics import *
from .preprocess import *

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.attack_results import FailedAttackResult, SuccessfulAttackResult


def create_output_hook(args, output_filename):
    repo_info = get_repo_info()

    args_dict = args
    args_dict = {**args_dict, **repo_info}

    def hook(attacked_samples):
        with open(output_filename, "w") as out_file:
            out_file.write(json.dumps({
                "meta": args_dict,
                "attacked_samples": attacked_samples
            }))

    return hook


def conll2003_to_textattack(ner_sample):
    """
        Given a sample from CoNLL2003 (https://huggingface.co/datasets/conll2003)
        converts it in a (input_text, ground_truth_labels) tuple
    """
    return (" ".join(ner_sample["tokens"]), torch.tensor(ner_sample["ner_tags"]))


def get_conll2003_labels():
    return [
        "O",
        "B-MISC",
        "I-MISC",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC"
    ]


def get_conll2003(max_samples=10, split="test", shuffle=True, remap=False) -> NERDataset:
    """
        Loads the chosen split pf the CoNLL2003 dataset and returns it
        as a NERDataset instance
    """
    dataset = load_dataset('conll2003', None, split=split)
    dataset = [conll2003_to_textattack(sample) for sample in dataset]

    if shuffle:
        random.shuffle(dataset)

    if max_samples is not None:
        dataset = dataset[:max_samples]

    # Skip samples without entities
    dataset = [x for x in dataset if x[1].sum() > 0]

    labels = get_conll2003_labels()

    if remap:
        dataset = [(s[0], remap_sample(s[1])) for s in dataset]

    dataset = NERDataset(dataset, label_names=labels)

    return dataset, labels


def load_custom_dataset(dataset_path) -> NERDataset:
    """
        Loads a custom dataset in the format
        {
            "meta" {
                "dataset_labels": ["O", "B-PER", ...]
            },
            "samples": [
                ["Lorem dolor ...", [0, 0, ...]]
            ]
        }
    """
    json_data = json.loads(open(dataset_path).read())

    labels = json_data["meta"]["dataset_labels"]
    dataset = [
        (x[0], torch.tensor(x[1]))
        for x in json_data["samples"]
    ]

    return NERDataset(dataset, label_names=labels), labels


def remap_sample(sample):
    """
        Converts the CoNLL2003 class labels from huggingface to
        the labels used in dslim/bert-base-NER

        Model labels: https://huggingface.co/dslim/bert-base-NER/blob/main/config.json
        Dataset labels: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py
    """
    mapping = {
        0: 0,
        1: 3,
        2: 4,
        3: 5,
        4: 6,
        5: 7,
        6: 8,
        7: 1,
        8: 2
    }

    return torch.tensor([mapping[int(x)] for x in sample])


def tensor_to_string(tensor):
    """
        Converts a 1xN tensor to a list of strings
    """
    stringified_tensor = [str(x) for x in tensor.tolist()]
    return " ".join(stringified_tensor)


def run_attack(
        attack,
        samples_count,
        dataset,
        dataset_name,
        model,
        max_percent_mispredictions=0.5,
        results_hook=None,
        timeout_seconds=60,
        raise_exceptions=False):
    print("")
    print("*****************************************")
    print(f"Starting Attack on {dataset_name}")
    print("*****************************************")
    print("")

    def timeout_hook(signum, frame):
        raise TimeoutError("Attack time expired")

    out_dataset = []

    for i in tqdm(range(len(dataset))):
        sample, ground_truth = dataset[i]

        model_raw = model([sample])[0]
        model_pred = model.process_raw_output(model_raw, sample)

        if len(model_pred) == 0:
            # The sample can't be correctly predicted by the model
            continue

        model_pred = tensor_to_string(model_pred)

        ground_truth_labels = tensor_to_string(ground_truth)
        ground_truth_labels_readable = [
            dataset.label_names[i] for i in ground_truth
        ]

        sample_record = {
            "attacked_sample": sample,
            "ground_truth": ground_truth_labels,
            "model_pred": model_pred
        }

        current_success_result = None

        def success_hook(result):
            nonlocal current_success_result
            current_success_result = result

        try:
            attacked_text = NERAttackedText(
                sample,
                attack_attrs={
                    "label_names": dataset.label_names
                },
                ground_truth=[int(x) for x in ground_truth])

            attack.goal_function.min_percent_entities_mispredicted = max_percent_mispredictions
            goal_function_result, _ = attack.goal_function.init_attack_example(
                attacked_text,
                ground_truth)

            if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
                sample_record["status"] = "SkippedAttackResult"
                sample_record["original_labels"] = ground_truth_labels_readable
            else:
                # The initial (original) misprediction score
                initial_score = attack.goal_function._get_score(
                    model_raw,
                    attacked_text)

                # Set the timeout for attacking a single sample
                signal.signal(signal.SIGALRM, timeout_hook)
                signal.alarm(timeout_seconds)

                result = run_attack_single(
                    attacked_text,
                    ground_truth,
                    attack,
                    max_percent_mispredictions,
                    initial_score,
                    success_callback=success_hook)

                add_result_to_sample(
                    sample_record,
                    result,
                    ground_truth_labels_readable)

                # Cancel the timeout
                signal.alarm(0)
        except Exception as ex:
            # If the attack crashes skip the sample
            # and restart the attack
            print(f"Could not attack sample: {ex}")

            if current_success_result is not None:
                # Even in the case of an overall failure
                # we might have a sample that causes a misprediction
                # in some entities
                add_result_to_sample(
                    sample_record,
                    current_success_result,
                    ground_truth_labels_readable)
            else:
                sample_record["status"] = "ErrorAttackResult"

            sample_record["exception_message"] = str(ex)

            if raise_exceptions:
                raise ex

            signal.alarm(0)

        out_dataset.append(sample_record)

        if results_hook is not None:
            results_hook(out_dataset)

    return out_dataset


def add_result_to_sample(sample_record, result, ground_truth_labels_readable):
    sample_record["status"] = str(type(result)).split(".")[-1].replace("'>", "")
    sample_record["num_queries"] = result.num_queries
    sample_record["raw_pred"] = result.perturbed_result.unprocessed_raw_output.tolist()

    sample_record["original_labels"] = ground_truth_labels_readable

    sample_record["original_score"] = result.original_result.score
    sample_record["perturbed_score"] = result.perturbed_result.score

    sample_record["perturbed_labels"] = result.perturbed_result._processed_output[0]
    sample_record["perturbed_pred"] = tensor_to_string(result.perturbed_result.raw_output)
    sample_record["perturbed_text"] = result.perturbed_result.attacked_text.text
    sample_record["final_ground_truth"] = result.perturbed_result.attacked_text.attack_attrs["ground_truth"]


def run_attack_single(
        attacked_text,
        ground_truth,
        attack,
        max_percent_mispredictions,
        initial_score,
        success_callback=None):
    """
        This function attacks a single sample trying to induce a misprediction
        of up to max_percent_mispredictions percent of entities in the attacked
        text.

        The worst-case prediction found will be returned or
        a FailedAttackResult.

        Samples which have an incorrect prediction to start with
         (SKIPPED samples) should not be passed to this function

        :param success_callback:    is a function called every time a new
                                    successful best result is found
    """
    assert initial_score >= 0 and initial_score <= 1, "The initial score must be in the interval [0, 1]"

    goal_function_result, _ = attack.goal_function.init_attack_example(
        attacked_text,
        ground_truth)

    # Try to cause a misprediction of up to max_percent_mispredictions by
    # stepping from the rounded initial score onward to the upper bound by
    # a factor of 0.1
    mispredictions_target_percents = list(np.arange(
        max(round(initial_score, 1), 0.1),
        max_percent_mispredictions, 0.1))
    mispredictions_target_percents = mispredictions_target_percents + [max_percent_mispredictions]

    best_result, best_score = None, None

    for target_misprediction in mispredictions_target_percents:
        if best_score is not None and best_score >= target_misprediction:
            # If we already induced a misprediction with a score higher than
            # target_misprediction skip this iteration
            continue

        # Update the goal function's misprediction target
        attack.goal_function.set_min_percent_entities_mispredicted(target_misprediction)

        result = attack.attack_one(goal_function_result)

        if type(result) == SuccessfulAttackResult:
            # Make sure we are improving on the score
            if best_result is None or score_result(result, attack.goal_function) > best_score:
                best_result = result
                best_score = score_result(result, attack.goal_function)

                if success_callback is not None:
                    success_callback(best_result)

            if best_score >= max_percent_mispredictions:
                # If we obtain a score as high (or higher) than our
                # target terminate and return the result
                return best_result
        elif type(result) == FailedAttackResult:
            # The attack cannot progress, so either return
            # the currrent best or the current (failed) result
            if best_result is not None:
                return best_result
            else:
                return result

    return best_result


def score_result(result, goal_function):
    perturbed_text = result.perturbed_result.attacked_text
    perturbed_prediction = result.perturbed_result.unprocessed_raw_output

    return goal_function._get_score(
        perturbed_prediction,
        perturbed_text)


def get_repo_info():
    try:
        repo = Repository(".")
        commit = list(repo.walk(repo.head.target, GIT_SORT_TOPOLOGICAL))[0]

        return {
            "git_head": repo.head.name,
            "git_commit_hex": commit.hex,
            "git_commit_message": commit.message.replace("\n", "")
        }
    except Exception as ex:
        print(f"Could not get the repository info: {ex}")

        return {}


def launch_attack_conll2003(
        attack_dict,
        attack,
        attack_meta,
        raise_exceptions=False):
    print("Attack meta: ")
    pprint(attack_meta)

    save_hook = create_output_hook(
        attack_meta,
        attack_dict["output"])

    # Run the attack!
    run_attack(
        attack,
        len(attack_dict["dataset"]),
        attack_dict["dataset"],
        "conll-2003",
        attack_dict["model"],
        max_percent_mispredictions=attack_dict["max_p_mispredicted"],
        results_hook=save_hook,
        timeout_seconds=attack_dict["attack_timeout"],
        raise_exceptions=raise_exceptions)
