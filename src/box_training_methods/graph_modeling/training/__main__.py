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
    "--negatives_permutation_option",
    type=click.Choice(["none", "head", "tail"], case_sensitive=False),
    default="none",
    help="whether to use permuted negatives during training, and if so whether to permute head or tail",
)
@click.option(
    "--undirected / --directed",
    default=None,
    help="whether to train using an undirected or directed graph (default is model dependent)",
    show_default=False,
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
    "--negative_weight", type=float, default=0.9, help="weight of negative loss",
)
@click.option(
    "--margin",
    type=float,
    default=1.0,
    help="margin for MaxMarginWithLogitsNegativeSamplingLoss or BCEWithDistancesNegativeSamplingLoss (unused otherwise)",
)
@click.option(
    "--negative_ratio",
    type=int,
    default=128,
    help="number of negative samples for each positive",
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
    "--vector_separate_io / --vector_no_separate_io",
    default=True,
    help="enable/disable using separate input/output representations for vector / bilinear vector model",
)
@click.option(
    "--vector_use_bias / --vector_no_use_bias",
    default=False,
    help="enable/disable using bias term in vector / bilinear",
)
@click.option(
    "--lorentzian_alpha",
    type=float,
    default=5.0,
    help="penalty for distance, where higher alpha emphasises distance as a determining factor in edge direction more",
)
@click.option(
    "--lorentzian_beta",
    type=float,
    default=1.0,
    help="-1/curvature of the space, if beta is higher the space is less curved / more euclidean",
)
@click.option(
    "--hyperbolic_entailment_cones_relative_cone_aperture_scale",
    type=float,
    default=1.0,
    help="float in (0,1) representing relative scale of cone apertures with respect to radius (K = relative_cone_aperature_scale * eps_bound / (1 - eps_bound^2))",
)
@click.option(
    "--hyperbolic_entailment_cones_eps_bound",
    type=float,
    default=0.1,
    help="restrict vectors to be parameterized in an annulus from eps to 1-eps",
)
@click.option(
    "--constrain_deltas_fn",
    type=click.Choice(["sqr", "exp", "softplus", "proj"]),
    default="sqr",
    help="which function to apply to width parameters of hard_box in order to make them positive, or use projected gradient descent (clipping in forward method)"
)
@click.option(
    "--box_intersection_temp",
    type=float,
    default=0.01,
    help="temperature of intersection calculation (hyperparameter for gumbel_box, initialized value for tbox)",
)
@click.option(
    "--box_volume_temp",
    type=float,
    default=1.0,
    help="temperature of volume calculation (hyperparameter for gumbel_box, initialized value for tbox)",
)
@click.option(
    "--tbox_temperature_type",
    type=click.Choice(["global", "per_dim", "per_entity", "per_entity_per_dim"]),
    default="per_entity_per_dim",
    help="type of learned temperatures (for tbox model)",
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
