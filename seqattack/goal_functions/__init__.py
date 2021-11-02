from .ner_goal_function_result import NERGoalFunctionResult
from .untargeted_ner import UntargetedNERGoalFunction
from .targeted_ner import TargetedNERGoalFunction
from .untargeted_ner_strict import StrictUntargetedNERGoalFunction


def get_goal_function(goal_function):
    goal_functions = {
        "untargeted": UntargetedNERGoalFunction,
        "untargeted-strict": StrictUntargetedNERGoalFunction,
        "targeted": TargetedNERGoalFunction,
    }

    if goal_function not in goal_functions.keys():
        raise ValueError(f"Invalid goal function {goal_function}. Valid values are {list(goal_functions.keys())}")

    return goal_functions[goal_function]
