import click

from utils import get_conll2003, \
    calculate_metrics, get_conll2003_labels, \
    remap_negations

from textattackner.models import NERModelWrapper
from textattackner.utils import postprocess_ner_output

from transformers import AutoTokenizer, AutoModelForTokenClassification


@click.group()
@click.pass_context
def cli(ctx):
    """
        Global setup
    """
    # ensure that ctx.attack exists and is a dict
    ctx.ensure_object(dict)


@cli.command()
@click.option("--model", required=True, type=str)
@click.option("--tokenizer", required=True, type=str)
@click.option("--mode", default=None, type=str)
@click.pass_context
def eval(ctx, model, tokenizer, mode):
    model = NERModelWrapper(
        AutoModelForTokenClassification.from_pretrained(model),
        AutoTokenizer.from_pretrained(tokenizer),
        postprocess_func=postprocess_ner_output)

    dataset, dataset_labels = get_conll2003(
        max_samples=None,
        split="test")

    dataset.dataset = remap_negations(dataset.dataset)

    dataset = [
        (sample[0], [dataset_labels[x] for x in sample[1]]) for sample in dataset.dataset
    ]

    original_str, _ = calculate_metrics(
        model,
        dataset,
        get_conll2003_labels(),
        mode=mode)

    print(original_str)


if __name__ == '__main__':
    cli(obj={})
