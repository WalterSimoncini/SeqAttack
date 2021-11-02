from textattackner.utils import diff_elements_count, tensor_mask

from .untargeted_ner import UntargetedNERGoalFunction


class StrictUntargetedNERGoalFunction(UntargetedNERGoalFunction):
    """
        Goal function that determines whether an attack on
        named entity recognition was successful. We consider an attack
        to be successful when at least one named entity token is mispredicted
        to any other class (switching between I-CLS and B-CLS is NOT allowed)
    """
    def _score_per_token(self, pred, pred_label, conf, truth, truth_label):
        """
            Returns 1 in case of a class swap and 1 - pred_conf in case of adherence
            to the ground truth
        """
        pred_label = pred_label.replace("I-", "").replace("B-", "")
        truth_label = truth_label.replace("I-", "").replace("B-", "")

        if pred_label == truth_label:
            return 1 - conf[truth]
        else:
            return 1

    def _get_score_labels(self, model_output, attacked_text):
        """
            Returns the percentage of mispredicted labels classes
            (e.g. ORG instead of PER) in the attacked text
        """
        mapped_ground_truth = self._preprocess_ground_truth(attacked_text)
        named_entities_ground_truth = tensor_mask(mapped_ground_truth)

        if named_entities_ground_truth.sum() == 0:
            # No entities = nothing we can do here.
            # Return the maximum score
            return 1

        pred_token_labels = [self.label_names[x] for x in model_output]
        truth_token_labels = [self.label_names[x] for x in mapped_ground_truth]

        pred_token_labels = [self.class_for_label(x) for x in pred_token_labels]
        truth_token_labels = [self.class_for_label(x) for x in truth_token_labels]

        mispredicted_tokens_count = diff_elements_count(
            truth_token_labels,
            pred_token_labels
        )

        if mispredicted_tokens_count > 0:
            # We have some mispredictions
            # Return the percentage of mispredicted entities
            # FIXME: maybe we can change it with recall/precision?
            return (mispredicted_tokens_count / named_entities_ground_truth.sum()).item()

        return 0

    @property
    def name(self):
        return "Strict untargeted NER goal function"
