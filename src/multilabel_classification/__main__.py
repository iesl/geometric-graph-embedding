import click

from .generate.__main__ import main as generate
# from .metrics.__main__ import main as metrics
# from .training.__main__ import train


@click.group()
def main():
    """Scripts to generate graphs, train and evaluate graph representations"""
    pass


main.add_command(generate, "generate")
# main.add_command(train, "train")
# main.add_command(metrics, "metrics")
