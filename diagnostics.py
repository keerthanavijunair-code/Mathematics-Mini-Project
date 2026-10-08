import numpy as np


def rref(a, tol=1e-10):
    a = np.array(a, dtype=float).copy()

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    rows, cols = a.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        max_row = pivot_row + np.argmax(np.abs(a[pivot_row:, col]))

        if abs(a[max_row, col]) < tol:
            continue

        if max_row != pivot_row:
            a[[pivot_row, max_row]] = a[[max_row, pivot_row]]

        a[pivot_row] /= a[pivot_row, col]

        for row in range(rows):
            if row != pivot_row:
                a[row] -= a[row, col] * a[pivot_row]

        pivot_row += 1

    a[np.abs(a) < tol] = 0
    return a


def rank(a, tol=1e-10):
    r = rref(a, tol)
    return np.sum(np.any(np.abs(r) > tol, axis=1))


def nullity(a, tol=1e-10):
    a = np.array(a, dtype=float)

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    return a.shape[1] - rank(a, tol)


def check_consistency(a, b, tol=1e-10):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    if a.shape[0] != len(b):
        raise ValueError("number of rows in A must match length of b")

    rank_a = rank(a, tol)
    rank_augmented = rank(np.column_stack((a, b)), tol)

    consistency = (
        "consistent" if rank_a == rank_augmented
        else "inconsistent"
    )

    return rank_a, rank_augmented, consistency


def system_type(a, b, tol=1e-10):
    a = np.array(a, dtype=float)
    rank_a, rank_augmented, consistency = check_consistency(a, b, tol)

    if consistency == "inconsistent":
        return "no solution"

    if rank_a == a.shape[1]:
        return "unique solution"

    return "infinitely many solutions"


def diagnose(a, b, tol=1e-10):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    if a.shape[0] != len(b):
        raise ValueError("number of rows in A must match length of b")

    rows, cols = a.shape
    rank_a, rank_augmented, consistency = check_consistency(a, b, tol)

    if rows > cols:
        shape_type = "overdetermined"
    elif rows < cols:
        shape_type = "underdetermined"
    else:
        shape_type = "square"

    if consistency == "inconsistent":
        solution_type = "no solution"
    elif rank_a == cols:
        solution_type = "unique solution"
    else:
        solution_type = "infinitely many solutions"

    return {
        "rows": rows,
        "columns": cols,
        "rank": rank_a,
        "nullity": cols - rank_a,
        "rank_augmented": rank_augmented,
        "consistency": consistency,
        "system_type": shape_type,
        "solution_type": solution_type
    }