import click

# graph_modeling is the only task that needs to generate graphs, so treat graph_modeling.generate as top-level group
from .graph_modeling.generate.__main__ import main as generate
from .train_eval.__main__ import train, eval


@click.group()
def main():
    """Scripts to generate graphs, train and evaluate graph representations"""
    pass


main.add_command(generate, "generate")
main.add_command(train, "train")
main.add_command(eval, "eval")

if __name__ == "__main__":
    main()
