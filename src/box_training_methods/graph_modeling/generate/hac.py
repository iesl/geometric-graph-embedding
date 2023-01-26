import networkx as nx
import numpy as np
from loguru import logger
from numpy.random import default_rng
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm

__all__ = [
    "generate",
]


def from_linkage_matrix(Z):
    edges = []  # parents
    for i in range(Z.shape[0]):
        edges.append((Z.shape[0] + 1 + i, int(Z[i, 0])))
        edges.append((Z.shape[0] + 1 + i, int(Z[i, 1])))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def load_text(infile):
    vecs = []
    lbls = []
    with open(infile, "r") as f:
        for i, line in enumerate(f):
            splits = line.strip().split("\t")
            lbls.append(splits[1])
            vecs.append([float(x) for x in splits[2:]])
    vecs = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    num_zero = np.sum(norms == 0)
    logger.info("Loaded vectors, and unit norming. %s vectors had 0 norm.", num_zero)
    norms[norms == 0] = 1.0
    vecs /= norms
    return np.arange(vecs.shape[0]), lbls, vecs


def generate(log_num_nodes: int, seed: int, vector_file: str, **kwargs) -> nx.DiGraph:

    pids, labels, X = load_text(vector_file)

    # select a subset of log_num_nodes
    r = default_rng(seed)
    idx = np.arange(X.shape[0])
    r.shuffle(idx)
    idx = idx[: 2 ** log_num_nodes]
    X = X[idx, :]

    def dot(XA, XB):
        return np.matmul(XA, XB.T)

    def batched_cdist(XA, XB, batch_size=1000, use_tqdm=True):
        res = np.zeros((XA.shape[0], XB.shape[0]), dtype=np.float32)
        if use_tqdm:
            for i in tqdm(range(0, XA.shape[0], batch_size), "cdist"):
                for j in range(0, XB.shape[0], batch_size):
                    istart = i
                    jstart = j
                    iend = min(XA.shape[0], i + batch_size)
                    jend = min(XB.shape[0], j + batch_size)
                    res[istart:iend, jstart:jend] = dot(
                        XA[istart:iend], XB[jstart:jend]
                    )
        else:
            for i in range(0, XA.shape[0], batch_size):
                for j in range(0, XB.shape[0], batch_size):
                    istart = i
                    jstart = j
                    iend = min(XA.shape[0], i + batch_size)
                    jend = min(XB.shape[0], j + batch_size)
                    res[istart:iend, jstart:jend] = dot(
                        XA[istart:iend], XB[jstart:jend]
                    )
        res = res + 1
        res = np.maximum(res, 0.0)
        return res

    Xdist = batched_cdist(X, X)
    Xdist = 0.5 * (Xdist + Xdist.T)
    Xdist = 2 - Xdist
    np.fill_diagonal(Xdist, 0.0)
    Xdist = np.maximum(Xdist, 0.0)
    from scipy.spatial.distance import squareform

    Xdist = squareform(Xdist)

    Z = linkage(Xdist, method="average")

    # build a coo matrix

    G = from_linkage_matrix(Z)
    return G
