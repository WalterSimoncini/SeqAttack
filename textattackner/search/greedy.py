"""
Greedy Search (with fix)
=================
"""
import numpy as np

from textattack.search_methods import GreedySearch
from textattack.goal_function_results import GoalFunctionResultStatus


class GreedySearchNER(GreedySearch):
    def _perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result

            results, search_over = self.get_goal_results(potential_next_beam)

            if results is None or len(results) == 0:
                # Results may be empty if the query budget was previously
                # exceeded. In this case return the initial result
                return initial_result

            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result
