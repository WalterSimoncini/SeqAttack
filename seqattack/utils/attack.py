import signal
import numpy as np

from textattack.shared import Attack
from seqattack.utils.ner_attacked_text import NERAttackedText
from textattack.goal_function_results import GoalFunctionResultStatus

from textattack.attack_results import (
    FailedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)


class NERAttack(Attack):
    def __init__(
            self,
            goal_function=None,
            constraints=[],
            transformation=None,
            search_method=None,
            transformation_cache_size=2**15,
            constraint_cache_size=2**15,
            max_entities_mispredicted=0.8,
            search_step=0.1,
            attack_timeout=30):
        super().__init__(
            goal_function=goal_function,
            constraints=constraints,
            transformation=transformation,
            search_method=search_method,
            transformation_cache_size=transformation_cache_size,
            constraint_cache_size=constraint_cache_size
        )

        self.search_step = search_step
        self.attack_timeout = attack_timeout
        self.max_entities_mispredicted = max_entities_mispredicted

    def _get_transformations_uncached(self, current_text, original_text=None, **kwargs):
        transformed_texts = super()._get_transformations_uncached(
            current_text,
            original_text=original_text,
            **kwargs)

        # Remove multiple spaces from samples
        for transformed in transformed_texts:
            transformed.strip_remove_double_spaces()

        return transformed_texts

    def timeout_hook(self, signum, frame):
        raise TimeoutError("Attack time expired")

    def attack_dataset(self, dataset, indices=None):
        # FIXME: Same as superclass
        examples = self._get_examples_from_dataset(dataset, indices=indices)

        for goal_function_result in examples:
            if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
                yield SkippedAttackResult(goal_function_result)
            else:
                result = self.attack_one(goal_function_result)
                yield result

    def attack_one(self, initial_result):
        attacked_text = self.goal_function.initial_attacked_text
        # The initial (original) misprediction score
        initial_score = self.goal_function._get_score(
            attacked_text.attack_attrs["model_raw"],
            attacked_text)

        best_result, best_score = None, 0
        # List of misprediction targets [0.1, 0.2, ...]
        target_scores = np.arange(
            max(initial_score, self.search_step),
            self.max_entities_mispredicted + self.search_step,
            self.search_step
        )

        try:
            # Set the timeout for attacking a single sample
            signal.signal(signal.SIGALRM, self.timeout_hook)
            signal.alarm(self.attack_timeout)

            for target in target_scores:
                # FIXME: To speed up the search use the current best result
                # FIXME: This code is better suited to be in a search method

                # Check if we can reach a mispredicion of target %
                if target < best_score:
                    # If we already obtained a sample with a score higher
                    # than the current target skip this iteration
                    continue

                self.goal_function.min_percent_entities_mispredicted = target

                result = super().attack_one(initial_result)

                current_score = self.goal_function._get_score(
                    result.perturbed_result.unprocessed_raw_output,
                    result.perturbed_result.attacked_text
                )

                if type(result) == SuccessfulAttackResult:
                    if best_result is None or current_score > best_score:
                        best_result, best_score = result, current_score
                elif type(result) == FailedAttackResult:
                    if best_result is None:
                        best_result, best_score = result, current_score

                    # The attack failed, nothing else we can do
                    break

            # Cancel the timeout
            signal.alarm(0)

            return best_result
        except Exception as ex:
           # FIXME: Handle timeouts etc.
           print(f"Could not attack sample: {ex}")

           return FailedAttackResult(
               initial_result,
               initial_result
           )

    def _get_examples_from_dataset(self, dataset, indices=None):
        # FIXME: indices is currently ignored
        for example, ground_truth in dataset:
            model_raw, _, valid_prediction = self.__is_example_valid(example, ground_truth)

            if not valid_prediction:
                continue

            attacked_text = NERAttackedText(
                example,
                attack_attrs={
                    "label_names": dataset.label_names,
                    "model_raw": model_raw
                },
                # FIXME: is this needed?
                ground_truth=[int(x) for x in ground_truth]
            )

            # If the original prediction mispredicts more entities than
            # max_entities_mispredicted then we skip the example
            self.goal_function.min_percent_entities_mispredicted = self.max_entities_mispredicted
            goal_function_result, _ = self.goal_function.init_attack_example(
                attacked_text,
                ground_truth
            )

            yield goal_function_result

    def __is_example_valid(self, sample, ground_truth):
        """Checks whether the model can correctly predict the sample or not"""
        model_raw = self.goal_function.model([sample])[0]
        model_pred = self.goal_function.model.process_raw_output(model_raw, sample)

        return model_raw, model_pred, len(model_pred) > 0
