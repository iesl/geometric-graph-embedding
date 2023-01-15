# Copyright 2021 The Geometric Graph Embedding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
        "networkx~=2.6.3",
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
    entry_points={"console_scripts": ["graph_modeling = graph_modeling.__main__:main"]},
)
