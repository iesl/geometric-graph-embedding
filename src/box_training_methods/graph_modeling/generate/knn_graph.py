import networkx as nx
import numpy as np
from loguru import logger
from numpy.random import default_rng
from tqdm import tqdm

__all__ = [
    "generate",
]


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


def generate(
    log_num_nodes: int, seed: int, vector_file: str, k: int, **kwargs
) -> nx.DiGraph:

    pids, labels, X = load_text(vector_file)

    # select a subset of log_num_nodes
    r = default_rng(seed)
    idx = np.arange(X.shape[0])
    r.shuffle(idx)
    idx = idx[: 2 ** log_num_nodes]
    X = X[idx, :]

    def dot(XA, XB):
        return np.matmul(XA, XB.T)

    def batched_knn(XA, XB, K, batch_size=1000, offset=0):
        K = np.minimum(K, XB.shape[0])
        res_i = np.zeros((XA.shape[0], K), dtype=np.int32)
        res = np.zeros((XA.shape[0], K), dtype=np.int32)
        resd = np.zeros((XA.shape[0], K), dtype=np.float32)
        for i in tqdm([x for x in range(0, XA.shape[0], batch_size)]):
            istart = i
            iend = min(XA.shape[0], i + batch_size)
            r = np.zeros((iend - istart, XB.shape[0]), dtype=np.float32)
            for j in range(0, XB.shape[0], batch_size):
                jstart = j
                jend = min(XB.shape[0], j + batch_size)
                r[:, jstart:jend] = dot(XA[istart:iend], XB[jstart:jend])
            np.put(
                r,
                np.arange(iend - istart) * r.shape[1] + np.arange(istart, iend),
                np.inf,
            )
            res[istart:iend, :] = np.argpartition(r, -K, axis=1)[:, -K:]
            resd[istart:iend, :] = r[
                np.arange(iend - istart)[:, None], res[istart:iend, :]
            ]
            res_i[istart:iend, :] = (
                np.repeat(np.expand_dims(np.arange(istart, iend), 1), K, axis=1)
                + offset
            )

        from scipy.sparse import coo_matrix

        row = res_i.flatten()
        col = res.flatten()
        d = resd.flatten()
        c = coo_matrix(
            (d, (row, col)), dtype=np.float32, shape=(XB.shape[0], XB.shape[0])
        )
        return c

    G = batched_knn(X, X, k)
    edges = []
    for i in range(G.row.shape[0]):
        edges.append((G.col[i], G.row[i]))

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G
