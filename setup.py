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
        "networkx==2.8.8",
        "scipy",
        "scikit-learn",
        "numpy",
        "xopen",
        "toml",
        "torch==1.11.0", # breaks with 
        "pandas",
        "loguru",
        "tqdm",
        "wandb",  # TODO: break out this dependency
    ],
    extras_require={
        "price_generation": ["graph_tool"],
        "test": ["pytest", "hypothesis"],
    },
    entry_points={"console_scripts": ["box_training_methods = box_training_methods.__main__:main"]},
)
