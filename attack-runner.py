import sys
import click
import random

from textattackner.attacks import (
    NERCLARE,
    BertAttackNER,
    MorpheusTan2020NER,
    NERDeepWordBugGao2018,
    NERTextFoolerJin2019,
    NERBAEGarg2019,
    NERSCPNParaphrase
)

from textattackner.goal_functions import \
    TargetedNERGoalFunction, UntargetedNERGoalFunction
from textattackner.goal_functions.untargeted_ner_strict import \
    StrictUntargetedNERGoalFunction

from textattackner.constraints import SkipModelErrors, \
    AvoidNamedEntityConstraint, NonNamedEntityConstraint

from utils.model import get_ner_model
from utils import get_conll2003, load_custom_dataset, \
    launch_attack_conll2003, remap_negations


@click.group()
@click.option("--output", required=True, type=str)
@click.option("--random-seed", default=567)
@click.option("--samples-count", default=256)
@click.option("--samples-offset", default=0, type=int)
@click.option("--max-percent-entities-mispredicted", default=0.8)
@click.option("--cache/--no-cache", default=False)
@click.option("--goal-function", default="untargeted")
@click.option("--max-queries", default=512)
@click.option("--attack-timeout", default=60)
@click.option("--custom-dataset-path", default=None)
@click.option("--dataset-split", default="test")
@click.option("--model-name", default="dslim/bert-base-NER")
@click.option("--remap/--no-remap", default=False)
@click.pass_context
def attack_runner(
        ctx,
        output,
        random_seed,
        samples_count,
        samples_offset,
        max_percent_entities_mispredicted,
        cache,
        goal_function,
        max_queries,
        attack_timeout,
        custom_dataset_path,
        dataset_split,
        model_name,
        remap):
    """
        Global setup needed for all attacks
    """
    goal_function_mapping = {
        "untargeted": UntargetedNERGoalFunction,
        "untargeted-strict": StrictUntargetedNERGoalFunction,
        "targeted": TargetedNERGoalFunction,
    }

    if goal_function not in goal_function_mapping.keys():
        print(f"Goal function '{goal_function}' does not exist. Valid values are {list(goal_function_mapping.keys())}")
        sys.exit(-1)

    # Set random seed
    random.seed(random_seed)

    # Load the dataset
    if custom_dataset_path:
        dataset, _ = load_custom_dataset(custom_dataset_path)
    else:
        dataset, _ = get_conll2003(
            max_samples=samples_count,
            split=dataset_split,
            remap=remap)

    if samples_offset > 0:
        dataset.dataset = dataset.dataset[samples_offset:]

    # Convert negations to a suitable form
    dataset.dataset = remap_negations(dataset.dataset)

    # Load model and tokenizer
    tokenizer, model = get_ner_model(model_name=model_name)

    # ensure that ctx.attack exists and is a dict
    ctx.ensure_object(dict)
    ctx.obj = {
        "output": output,
        "goal_function_cls": goal_function_mapping[goal_function],
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "cache": cache,
        "max_queries": max_queries,
        "random_seed": random_seed,
        "samples_count": samples_count,
        "max_p_mispredicted": max_percent_entities_mispredicted,
        "attack_timeout": attack_timeout,
        "split": None if custom_dataset_path else dataset_split,
        "model_name": model_name,
        "remap": remap
    }


@attack_runner.command()
@click.option("--max-perturbed-percent", default=0.4)
@click.option("--max-candidates", default=48)
@click.pass_context
def bert_attack(ctx, max_perturbed_percent, max_candidates):
    constraints_model_wrapper = get_ner_model(
        model_name=ctx.obj["model_name"]
    )[1]

    additional_constraints = [
        SkipModelErrors(model_wrapper=constraints_model_wrapper),
        AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper)
    ]

    attack = BertAttackNER.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=additional_constraints,
        max_perturbed_percent=max_perturbed_percent,
        max_candidates=max_candidates,
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"])

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "max_perturbed_percent": max_perturbed_percent,
            "max_candidates": max_candidates,
            "script_name": bert_attack.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


@attack_runner.command()
@click.option("--max-candidates", default=50)
@click.pass_context
def clare(ctx, max_candidates):
    constraints_model_wrapper = get_ner_model(
        model_name=ctx.obj["model_name"]
    )[1]

    additional_constraints = [
        SkipModelErrors(model_wrapper=constraints_model_wrapper),
        AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper)
    ]

    attack = NERCLARE.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=additional_constraints,
        max_candidates=max_candidates,
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"])

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "max_candidates": max_candidates,
            "script_name": clare.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


@attack_runner.command()
@click.pass_context
def scpn(ctx):
    attack = NERSCPNParaphrase.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=[],
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"])

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "script_name": scpn.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


@attack_runner.command()
@click.option("--constraint/--no-constraint", default=True)
@click.option("--max-edit-distance", default=50)
@click.pass_context
def deepwordbug(ctx, constraint, max_edit_distance):
    constraints_model = get_ner_model(
        model_name=ctx.obj["model_name"]
    )[1]

    additional_constraints = [
        SkipModelErrors(model_wrapper=constraints_model)
    ]

    if constraint:
        additional_constraints.extend([
            AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model),
            NonNamedEntityConstraint()
        ])

    attack = NERDeepWordBugGao2018.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=additional_constraints,
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"],
        max_edit_distance=max_edit_distance)

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "max_edit_distance": max_edit_distance,
            "script_name": deepwordbug.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


@attack_runner.command()
@click.pass_context
def morpheus(ctx):
    constraints_model_wrapper = get_ner_model(
        model_name=ctx.obj["model_name"]
    )[1]

    additional_constraints = [
        SkipModelErrors(model_wrapper=constraints_model_wrapper),
        AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper)
    ]

    attack = MorpheusTan2020NER.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=additional_constraints,
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"])

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "script_name": morpheus.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


@attack_runner.command()
@click.option("--max-candidates", default=50)
@click.pass_context
def textfooler(ctx, max_candidates):
    constraints_model_wrapper = get_ner_model(
        model_name=ctx.obj["model_name"]
    )[1]

    additional_constraints = [
        SkipModelErrors(model_wrapper=constraints_model_wrapper),
        AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper)
    ]

    attack = NERTextFoolerJin2019.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=additional_constraints,
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"],
        max_candidates=max_candidates)

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "max_candidates": max_candidates,
            "script_name": textfooler.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


@attack_runner.command()
@click.option("--max-candidates", default=50)
@click.pass_context
def bae(ctx, max_candidates):
    constraints_model_wrapper = get_ner_model(
        model_name=ctx.obj["model_name"]
    )[1]

    additional_constraints = [
        SkipModelErrors(model_wrapper=constraints_model_wrapper),
        AvoidNamedEntityConstraint(ner_model_wrapper=constraints_model_wrapper)
    ]

    attack = NERBAEGarg2019.build(
        ctx.obj["model"],
        ctx.obj["tokenizer"],
        ctx.obj["dataset"],
        ctx.obj["goal_function_cls"],
        additional_constraints=additional_constraints,
        use_cache=ctx.obj["cache"],
        query_budget=ctx.obj["max_queries"],
        max_candidates=max_candidates)

    all_attack_params = preprocess_args_dict(
        ctx.obj,
        {
            "script_name": bae.name
        }
    )

    launch_attack_conll2003(ctx.obj, attack, all_attack_params)


def preprocess_args_dict(global_params, local_params):
    args_dict = {**global_params, **local_params}

    args_dict["tokenizer"] = str(args_dict["tokenizer"])
    args_dict["model"] = str(args_dict["model"])
    args_dict["dataset"] = str(args_dict["dataset"])
    args_dict["goal_function_cls"] = str(args_dict["goal_function_cls"])

    return args_dict


if __name__ == '__main__':
    attack_runner(obj={})
