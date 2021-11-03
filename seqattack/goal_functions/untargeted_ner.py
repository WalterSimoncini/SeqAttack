from seqattack.utils import diff_elements_count, tensor_mask

from .ner import NERGoalFunction


class UntargetedNERGoalFunction(NERGoalFunction):
    """
        Goal function that determines whether an attack on
        named entity recognition was successful. We consider an attack
        to be successful when at least one named entity token is mispredicted
        to any other class (switching between I-CLS and B-CLS IS allowed)
    """
    def _is_goal_complete(self, model_output, attacked_text):
        preds, _, _, _ = self._preprocess_model_output(
            model_output,
            attacked_text)

        return self._get_score_labels(preds, attacked_text) >= self.min_percent_entities_mispredicted

    def _get_score(self, model_output, attacked_text):
        return self._get_score_confidence(
            model_output,
            attacked_text)

    def _get_score_confidence(self, model_output, attacked_text):
        total_score = 0

        preds, _, named_entities_mask, all_labels_confidences = self._preprocess_model_output(
            model_output,
            attacked_text)

        mapped_ground_truth = self._preprocess_ground_truth(attacked_text)

        pred_token_labels = [self.label_names[x] for x in preds]
        truth_token_labels = [self.label_names[x] for x in mapped_ground_truth]

        for pred, pred_label, conf, truth, truth_label in zip(preds, pred_token_labels, all_labels_confidences, mapped_ground_truth, truth_token_labels):
            total_score += self._score_per_token(int(pred), pred_label, conf, int(truth), truth_label)

        if named_entities_mask.sum() == 0:
            # Always return 1 if the input sample has no
            # named entities (nothing to do here)
            return 1
        else:
            return float(total_score / named_entities_mask.sum())

    def _score_per_token(self, pred, pred_label, conf, truth, truth_label):
        if pred == truth:
            return 1 - conf[pred]
        elif truth == 0:
            # A named entity was introduced. No score added
            return 0
        else:
            return 1

    def _get_score_labels(self, model_output, attacked_text):
        mapped_ground_truth = self._preprocess_ground_truth(attacked_text)
        named_entities_ground_truth = tensor_mask(mapped_ground_truth)

        if named_entities_ground_truth.sum() == 0:
            # No entities = nothing we can do here.
            # Return the maximum score
            return 1

        total_score = 0

        for truth, model_out in zip(mapped_ground_truth, model_output):
            if truth > 0 and truth != model_out:
                total_score += 1

        # Return the percentage of mispredicted entities
        return (total_score / named_entities_ground_truth.sum()).item()

    @property
    def name(self):
        return "Untargeted NER goal function"
