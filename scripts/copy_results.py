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

import os
import argparse
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Submit multiple sweeps")
parser.add_argument("--src", type=str, default="")
parser.add_argument("--tgt", type=str, default="")
parser.add_argument("--model_type", type=str)
parser.add_argument("--dim", type=int, help="dimension ")
config = parser.parse_args()
if config.model_type not in [
    "pure_gumbel",
    "per_entity_learned_temp_box",
    "per_dim_learned_temp_box",
    "global_learned_temp_box",
    "box",
    "oe",
    "poe",
    "vector",
    "vector_dist",
    "bilinear_vector",
    "complex_vector",
    "lorentzian_distance",
]:
    raise Exception(f"model type {config.model_type} does not exist")

if config.model_type not in [
    "box",
    "pure_gumbel",
    "per_entity_learned_temp_box",
    "per_dim_learned_temp_box",
    "global_learned_temp_box",
    "vector",
    "complex_vector",
    "vector_dist",
]:
    config.dim = config.dim * 2

for path in tqdm(
    Path(config.src).glob(f"**/results/{config.model_type}_{config.dim}/*")
):
    tgt_path = config.tgt + str(path).split(config.src)[1]
    tgt_dir = os.path.dirname(tgt_path)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    os.system(f"cp {str(path)} {tgt_path}")
