import click


class IntOrPercent(click.ParamType):
    name = "click_union"

    def convert(self, value, param, ctx):
        try:
            float_value = float(value)
            if 0 <= float_value <= 1:
                return float_value
            elif float_value == int(float_value):
                return int(float_value)
            else:
                self.fail(
                    f"expected float between [0,1] or int, got {float_value}",
                    param,
                    ctx,
                )
        except TypeError:
            self.fail(
                "expected string for int() or float() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or float", param, ctx)


@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--data_path",
    type=click.Path(),
    help="directory or file with graph data (eg. data/graph/some_tree)",
    required=True,
)
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "tbox",
            "gumbel_box",
            "hard_box",
            "order_embeddings",
            "partial_order_embeddings",
            "vector_sim",
            "vector_dist",
            "bilinear_vector",
            "complex_vector",
            "lorentzian_distance",
            "lorentzian_score",
            "lorentzian",
            "hyperbolic_entailment_cones",
        ],
        case_sensitive=False,
    ),
    default="tbox",
    help="model architecture to use",
)
@click.option(
    "--dim", type=int, default=4, help="dimension for embedding space",
)
@click.option(
    "--log_batch_size",
    type=int,
    default=10,
    help="batch size for training will be 2**LOG_BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--log_eval_batch_size",
    type=int,
    default=15,
    help="batch size for eval will be 2**LOG_EVAL_BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--learning_rate", type=float, default=0.01, help="learning rate",
)
@click.option(
    "--epochs", type=int, default=1_000, help="maximum number of epochs to train"
)
@click.option(
    "--patience",
    type=int,
    default=11,
    help="number of log_intervals without decreased loss before stopping training",
)
@click.option(
    "--log_interval",
    type=IntOrPercent(),
    default=0.1,
    help="interval or percentage (as float in [0,1]) of examples to train between logging training metrics",
)
@click.option(
    "--eval / --no_eval",
    default=True,
    help="whether or not to evaluate the model at the end of training",
)
@click.option(
    "--cuda / --no_cuda", default=True, help="enable/disable CUDA (eg. no nVidia GPU)",
)
@click.option(
    "--save_prediction / --no_save_prediction",
    default=False,
    help="enable/disable saving predicted adjacency matrix",
)
@click.option(
    "--seed", type=int, help="seed for random number generator",
)
@click.option(
    "--wandb / --no_wandb",
    default=False,
    help="enable/disable logging to Weights and Biases",
)
@click.option(
    "--output_dir",
    type=str,
    default=None,
    help="output directory for recording current hyper-parameters and results",
)
@click.option(
    "--save_model / --no_save_model",
    type=bool,
    default=False,
    help="whether or not to save the model to disk",
)
def train(**config):
    """Train a graph embedding representation"""
    from .train import training

    training(config)
