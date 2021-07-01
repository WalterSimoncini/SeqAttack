from textattack.shared.utils import color_from_output, color_text
from textattack.goal_function_results import GoalFunctionResult


class NERGoalFunctionResult(GoalFunctionResult):
    """
        Represents the result of a NER goal function.
    """
    def __init__(
        self,
        attacked_text,
        raw_output,
        output,
        goal_status,
        score,
        num_queries,
        ground_truth_output,
        unprocessed_raw_output
    ):
        super().__init__(
            attacked_text=attacked_text,
            raw_output=raw_output,
            output=output,
            goal_status=goal_status,
            score=score,
            num_queries=num_queries,
            ground_truth_output=ground_truth_output)

        self.unprocessed_raw_output = unprocessed_raw_output

    @property
    def _processed_output(self):
        """
            Takes a model output (like `1`) and returns the class labeled output
            (like `positive`), if possible. Also returns the associated color.
        """
        if self.attacked_text.attack_attrs.get("label_names"):
            token_labels = [
                self.attacked_text.attack_attrs["label_names"][x] for x in self.raw_output
            ]

            token_colors = [
                color_from_output(label, class_id) for (label, class_id) in zip(token_labels, self.raw_output)
            ]

            return token_labels, token_colors
        else:
            raise Exception("The dataset has no labels!")

    def get_text_color_input(self):
        """
            A string representing the color this result's changed portion should
            be if it represents the original input.
        """
        return "red"

    def get_text_color_perturbed(self):
        """
            A string representing the color this result's changed portion should
            be if it represents the perturbed input.
        """
        return "blue"

    def get_colored_output(self, color_method=None):
        """
            Returns a string representation of this result's output, colored
            according to `color_method`.
        """
        colored_labels = []
        token_labels, token_colors = self._processed_output

        for label, color in zip(token_labels, token_colors):
            colored_output = color_text(label, color=color, method=color_method)
            colored_labels.append(colored_output)

        return ", ".join(colored_labels)
