import click

from commands import (
    attack,
    evaluate_attacked,
    evaluate,
    pick_samples
)


@click.group()
@click.pass_context
def cli(ctx):
    pass


if __name__ == '__main__':
    cli.add_command(attack)
    cli.add_command(evaluate)
    cli.add_command(pick_samples)
    cli.add_command(evaluate_attacked)

    cli(obj={})
