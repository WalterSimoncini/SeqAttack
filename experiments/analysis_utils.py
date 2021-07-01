from colored import fg, bg, attr

from attack_result import AttackResult


def highlight_sequences_diff(original_seq, modified_seq, seq_joiner=" "):
    out_sequence_mod = []
    out_sequence_original = []

    for or_token, mod_token in zip(original_seq, modified_seq):
        if or_token != mod_token:
            out_sequence_mod.append(f"{fg(1)}{mod_token}{attr(0)}")
            out_sequence_original.append(f"{fg(111)}{or_token}{attr(0)}")
        else:
            out_sequence_mod.append(mod_token)
            out_sequence_original.append(or_token)

    return seq_joiner.join(out_sequence_mod), seq_joiner.join(out_sequence_original)


def preprocess_text_sample(sample):
    return list(filter(lambda x: x.strip() != "", sample.split("\n")))


def process_attacked_text_sample(sample):
    lines = preprocess_text_sample(sample)

    # Remove TF errors
    lines = [line for line in lines if "tensorflow.org/guide" not in line]

    # Successful samples have 6 lines of text
    successful = len(lines) == 6

    if successful:
        original_labels, attacked_labels = lines[3].split(" --> ")

        return True, {
            "sample": lines[0].replace("Attacking sample: ", ""),
            "ground_truth": lines[1].replace("The ground truth labels are:\t", ""),
            "model_pred": lines[2].replace("The model prediction is:\t", ""),
            "labels": {
                "original": original_labels.split(", "),
                "attacked": attacked_labels.split(", ")
            },
            "post_attack_sample": lines[-1]
        }

    return False, None


def print_attack_summary(samples_dict):
    print("\nSummary statistics:")

    total_samples = 0

    for k in samples_dict.keys():
        total_samples += len(samples_dict[k])

    successful_samples = len(samples_dict[AttackResult.SUCCESS.value])
    failed_samples = len(samples_dict[AttackResult.FAILED.value])
    errors_count = len(samples_dict[AttackResult.ERROR.value])

    print(f"    ▶ Successfully attacked {successful_samples}/{total_samples} samples")
    print(f"    ▶ Failed to attack {failed_samples}/{total_samples} samples")

    if AttackResult.SKIPPED.value in samples_dict:
        skipped_samples = len(samples_dict[AttackResult.SKIPPED.value])
        print(f"    ▶ Skipped {skipped_samples}/{total_samples} samples")

    print(f"    ▶ Could not attack {errors_count}/{total_samples} samples due to errors\n")
    print(f"Original words/labels are in {fg(111)}BLUE{attr(0)} and the attacked ones are in {fg(1)}RED{attr(0)}")
