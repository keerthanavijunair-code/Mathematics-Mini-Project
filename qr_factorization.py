import numpy as np

from gram_schmidt import gram_schmidt, normalize


def qr_factorization(a, tol=1e-10):
    a = np.array(a, dtype=float)

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    if a.shape[1] == 0:
        return np.empty((a.shape[0], 0)), np.empty((0, 0))

    u = gram_schmidt(a, tol)
    q = normalize(u, tol)
    r = q.T @ a

    return q, r


def verify_qr(a, q, r, tol=1e-10):
    return np.allclose(a, q @ r, atol=tol)


def verify_orthonormal(q, tol=1e-10):
    if q.shape[1] == 0:
        return True

    identity = np.eye(q.shape[1])

    return np.allclose(q.T @ q, identity, atol=tol)


def back_substitution(r, c, tol=1e-10):
    r = np.array(r, dtype=float)
    c = np.array(c, dtype=float).reshape(-1)

    if r.ndim != 2:
        raise ValueError("R must be a 2D matrix")

    if r.shape[0] != r.shape[1]:
        raise ValueError("R must be square for back substitution")

    if r.shape[0] != len(c):
        raise ValueError("R and c dimensions do not match")

    n = len(c)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(r[i, i]) < tol:
            raise ValueError("R is singular")

        total = np.dot(r[i, i + 1:], x[i + 1:])
        x[i] = (c[i] - total) / r[i, i]

    return x


def qr_solve(a, b, tol=1e-10):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    q, r = qr_factorization(a, tol)

    if q.shape[1] != r.shape[0]:
        raise ValueError("invalid QR dimensions")

    c = q.T @ b

    if r.shape[0] != r.shape[1]:
        raise ValueError(
            "QR solve requires a square R. "
            "For a rectangular least-squares system, use the "
            "least-squares stage."
        )

    x = back_substitution(r, c, tol)

    return x, q, r
