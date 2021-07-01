import os
import json
import argparse

from utils.model import get_ner_model
from utils import get_conll2003_labels

from textattack.shared.attacked_text import AttackedText

from textattackner.goal_functions import UntargetedNERGoalFunction
from textattackner.utils import postprocess_ner_output


parser = argparse.ArgumentParser(description="Cherry picks samples from multiple datasets according to a scoring function")
parser.add_argument("--directory", type=str, required=True, help="Directory containing the attacked datasets")
parser.add_argument("--output", type=str, required=True, help="Output dataset path")

args = parser.parse_args()

target_files = [fn for fn in os.listdir(args.directory) if ".json" in fn]

datasets = [json.loads(
    open(os.path.join(args.directory, fn)).read()
) for fn in target_files]

samples_dict = {}

for dataset in datasets:
    for sample in dataset["attacked_samples"]:
        in_sample = sample["attacked_sample"]

        if in_sample not in samples_dict:
            samples_dict[in_sample] = []

        samples_dict[in_sample].append(sample)

# Load model and tokenizer
tokenizer, model = get_ner_model()
goal_function = UntargetedNERGoalFunction(
    model_wrapper=model,
    tokenizer=tokenizer,
    min_percent_entities_mispredicted=0.5,
    ner_postprocess_func=postprocess_ner_output,
    label_names=get_conll2003_labels())

# Pick best samples for each dict key
picked_samples = []

for k in samples_dict.keys():
    # Default
    best_score, best_sample = None, samples_dict[k][0]

    for sample in samples_dict[k]:
        if "raw_pred" not in sample:
            continue

        ground_truth = [
            int(x) for x in sample["ground_truth"].split(" ")
        ]

        goal_function.init_attack_example(
            AttackedText(sample["attacked_sample"]),
            ground_truth)

        sample_score = goal_function._get_score(
            sample["raw_pred"],
            AttackedText(sample["perturbed_text"]))

        if best_score is None or sample_score > best_score:
            best_score, best_sample = sample_score, sample
    
    picked_samples.append(best_sample)

# Save output file
with open(args.output, "w") as output_file:
    output_file.write(json.dumps({
        "attacked_samples": picked_samples
    }))
