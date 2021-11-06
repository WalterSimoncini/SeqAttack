import json


class AttackRunner():
    def __init__(
            self,
            attack,
            dataset,
            output_filename: str,
            attack_args: dict = None) -> None:
        self.attack = attack
        self.dataset = dataset
        self.output_filename = output_filename
        self.attack_args = attack_args

    def run(self):
        attack_results = []
        attack_iterator = self.attack.attack_dataset(self.dataset)

        print("*****************************************")
        print(f"Starting attack on {self.dataset.name}")
        print("*****************************************")

        for sample, ground_truth in self.dataset:
            sample_labels = self.__prediction_to_labels(
                ground_truth,
                self.dataset.label_names
            )

            sample_labels = " ".join(sample_labels)

            print("-----------------------------------------")
            print(f"Attacking sample: {sample}")
            print(f"Labels: {sample_labels}")

            result = next(attack_iterator)

            print()
            print(f"Result: {self.__result_status(result)}")
            print()

            perturbed_text = result.perturbed_result.attacked_text.text
            perturbed_labels = self.__prediction_to_labels(
                result.perturbed_result.raw_output.tolist(),
                self.dataset.label_names
            )

            perturbed_labels = " ".join(perturbed_labels)

            print(f"Perturbed sample: {perturbed_text}")
            print(f"Labels: {perturbed_labels}")

            attack_results.append(self.attack_result_to_json(result))
            self.save_results(attack_results)

    def attack_result_to_json(self, result):
        original_text = result.original_result.attacked_text
        perturbed_text = result.perturbed_result.attacked_text

        labels_map = original_text.attack_attrs.get("label_names", None)

        original_pred = result.original_result.raw_output.tolist()
        perturbed_pred = result.perturbed_result.raw_output.tolist()

        try:
            perturbed_words = perturbed_text.words_diff_ratio(original_text)
        except (ZeroDivisionError, AssertionError):
            # Assign a difference of zero in case the input text has no words
            # or the two texts have a different words count
            perturbed_words = 0

        data = {
            "original_text": original_text.text,
            "ground_truth": original_text.attack_attrs["ground_truth"],
            "original_pred": original_pred,
            "status": self.__result_status(result),
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "perturbed_pred": perturbed_pred,
            "perturbed_text": perturbed_text.text,
            "final_ground_truth": perturbed_text.attack_attrs["ground_truth"],
            # FIXME: This might be an issue with CLARE (and insertions in general)
            "percent_words_perturbed": perturbed_words,
            "num_queries": result.num_queries
        }

        if labels_map is not None:
            labelled_data = {
                "ground_truth_labels": self.__prediction_to_labels(original_text.attack_attrs["ground_truth"], labels_map),
                "original_pred_labels": self.__prediction_to_labels(original_pred, labels_map),
                "perturbed_pred_labels": self.__prediction_to_labels(perturbed_pred, labels_map),
                "final_ground_truth_labels": self.__prediction_to_labels(perturbed_text.attack_attrs["ground_truth"], labels_map)
            }

            data = {**data, **labelled_data}

        return data

    def __result_status(self, result):
        return str(type(result)).split(".")[-1].replace("'>", "").replace("AttackResult", "")

    def __prediction_to_labels(self, pred, labels_map):
        return [labels_map[x] for x in pred]

    def save_results(self, attack_results):
        with open(self.output_filename, "w") as out_file:
            recipe_metadata = self.attack_args["recipe_metadata"]
            recipe_metadata["additional_constraints"] = [
                str(const) for const in recipe_metadata["additional_constraints"]
            ]

            out_file.write(json.dumps({
                "config": {
                    "goal_function": self.attack.goal_function.name,
                    "model": self.attack_args["model"].name,
                    "tokenizer": str(self.attack_args["tokenizer"]),
                    "dataset": self.attack_args["dataset"].name,
                    "cache": self.attack_args["use_cache"],
                    "query_budget": self.attack_args["query_budget"],
                    "random_seed": self.attack_args["random_seed"],
                    "examples_count": self.attack_args["num_examples"],
                    "max_entities_mispredicted": self.attack_args["max_entities_mispredicted"],
                    "attack_timeout": self.attack_args["attack_timeout"],
                    "split": self.attack_args["dataset"].split,
                    "recipe": self.attack_args["recipe"],
                    "recipe_args": recipe_metadata
                },
                "attacked_examples": attack_results
            }))
