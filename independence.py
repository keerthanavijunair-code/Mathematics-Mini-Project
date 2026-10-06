import numpy as np

from diagnostics import rank


def independent_columns(a, tol=1e-10):
    a = np.array(a, dtype=float)

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    selected = []
    current_rank = 0

    for i in range(a.shape[1]):
        test_columns = selected + [i]
        test_matrix = a[:, test_columns]
        new_rank = rank(test_matrix, tol)

        if new_rank > current_rank:
            selected.append(i)
            current_rank = new_rank

    redundant = [i for i in range(a.shape[1]) if i not in selected]

    return selected, redundant


def get_basis(a, tol=1e-10):
    selected, redundant = independent_columns(a, tol)

    if len(selected) == 0:
        basis = np.empty((a.shape[0], 0))
    else:
        basis = np.array(a, dtype=float)[:, selected]

    return basis, selected, redundant
