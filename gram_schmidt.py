import numpy as np


def projection(v, u):
    denominator = np.dot(u, u)

    if abs(denominator) < 1e-12:
        return np.zeros_like(v)

    return (np.dot(v, u) / denominator) * u


def gram_schmidt(a, tol=1e-10):
    a = np.array(a, dtype=float)

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    vectors = []

    for i in range(a.shape[1]):
        v = a[:, i].copy()

        for u in vectors:
            v = v - projection(v, u)

        if np.linalg.norm(v) > tol:
            vectors.append(v)

    if len(vectors) == 0:
        return np.empty((a.shape[0], 0))

    return np.column_stack(vectors)


def normalize(vectors, tol=1e-10):
    vectors = np.array(vectors, dtype=float)

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")

    q = []

    for i in range(vectors.shape[1]):
        norm = np.linalg.norm(vectors[:, i])

        if norm > tol:
            q.append(vectors[:, i] / norm)

    if len(q) == 0:
        return np.empty((vectors.shape[0], 0))

    return np.column_stack(q)
