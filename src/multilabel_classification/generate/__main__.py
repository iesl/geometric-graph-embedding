import functools
import random

import click

from . import write_dataset


@click.group()
def main():
    """Multilabel classifiation dataset generation commands"""
    pass


def _common_options(func):
    """Common options used in all subcommands"""

    @main.command(context_settings=dict(show_default=True))
    @click.option(
        "--indir",
        type=click.Path(writeable=False),
        help="path to dataset dir",
    )
    @click.option(
        "--dataset_name",
        type=str,
        help="e.g. cellcycle_FUN"
    )
    @click.option(
        "--num_labels",
        default=500,
        type=int,
        help="number of classification labels"
    )
    @click.option(
        "--outdir",
        default="data/mlc/",
        type=click.Path(writable=True),
        help="location to save output",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@_common_options
def mlc_dataset(outdir, **mlc_config):
    """Writes out a pickled mlc dataset"""
    write_dataset(outdir, **mlc_config)
