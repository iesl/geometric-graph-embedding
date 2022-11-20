"""
Scripts to generate graphs, train and evaluate graph representations
"""
import fastentrypoints
from setuptools import find_packages, setup

setup(
    name="graph_modeling",
    version="0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    description="Scripts to generate graphs, train and evaluate graph representations",
    install_requires=[
        "Click>=7.1.2",
        "networkx",
        "scipy",
        "scikit-learn",
        "numpy",
        "xopen",
        "toml",
        "torch",
        "pandas",
        "loguru",
        "tqdm",
        "wandb",  # TODO: break out this dependency
    ],
    extras_require={
        "price_generation": ["graph_tool"],
        "test": ["pytest", "hypothesis"],
    },
    entry_points={"console_scripts": ["graph_modeling = graph_modeling.__main__:main",
                                      "multilabel_classification = multilabel_classification.__main__:main"]},
)
