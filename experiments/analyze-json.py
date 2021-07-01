import json
import argparse

from colored import fg, bg, attr

from attack_result import AttackResult
from analysis_utils import highlight_sequences_diff, print_attack_summary


parser = argparse.ArgumentParser(description="Process a JSON TextAttack output")
parser.add_argument("--filename", type=str, help="The input file")

args = parser.parse_args()

attacked_samples = json.loads(open(args.filename).read())["attacked_samples"]
samples_dict = {
    AttackResult.SUCCESS.value: [],
    AttackResult.FAILED.value: [],
    AttackResult.ERROR.value: [],
    AttackResult.SKIPPED.value: []
}

for sample in attacked_samples:
    attack_status = AttackResult.from_textattack_class(sample["status"])
    samples_dict[attack_status.value].append(sample)

    # Show the details of successful attacks
    if attack_status == AttackResult.SUCCESS:
        print("------------------------------------")

        diff_labels, orig_labels = highlight_sequences_diff(
            sample["original_labels"],
            sample["perturbed_labels"],
            seq_joiner="\t"
        )

        diff_sample, orig_sample = highlight_sequences_diff(
            sample["attacked_sample"].split(" "),
            sample["perturbed_text"].split(" ")
        )

        model_queries = sample["num_queries"]

        print(f"Original labels: {orig_sample}")
        print(f"Attacked labels: {diff_sample}")
        
        print("\n")

        print(f"Original labels: {orig_labels}")
        print(f"Attacked labels: {diff_labels}")

print_attack_summary(samples_dict)
