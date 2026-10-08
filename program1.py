import numpy as np
import time


# ---------------------------------------------------------
# 1. GAUSSIAN ELIMINATION
# ---------------------------------------------------------
def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian Elimination.

    Steps:
    1. Convert A into an upper triangular matrix.
    2. Use back substitution to find the values of x.
    """

    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(b)

    # Forward elimination
    for i in range(n):

        # Partial pivoting:
        # Choose the row with the largest pivot value.
        max_row = i + np.argmax(np.abs(A[i:, i]))

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        # Make all elements below the pivot zero
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]

            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            b[i] - np.dot(A[i, i + 1:], x[i + 1:])
        ) / A[i, i]

    return x


# ---------------------------------------------------------
# 2. RREF
# ---------------------------------------------------------
def rref(A):
    """
    Converts a matrix into Reduced Row Echelon Form.

    Each pivot is made equal to 1 and all other
    entries in the pivot column are made zero.
    """

    A = A.astype(float).copy()

    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):

        if pivot_row >= rows:
            break

        # Find the best pivot row
        max_row = pivot_row + np.argmax(
            np.abs(A[pivot_row:, col])
        )

        # Skip the column if there is no pivot
        if abs(A[max_row, col]) < 1e-12:
            continue

        # Swap rows if required
        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

        # Make the pivot equal to 1
        A[pivot_row] = (
            A[pivot_row] / A[pivot_row, col]
        )

        # Make all other values in this column zero
        for row in range(rows):

            if row != pivot_row:
                factor = A[row, col]
                A[row] = A[row] - factor * A[pivot_row]

        pivot_row += 1

    return A


def rref_solve(A, b):
    """
    Solves Ax = b by applying RREF to the augmented matrix [A | b].
    """

    augmented = np.column_stack((A, b))

    reduced = rref(augmented)

    # Last column contains the solution
    return reduced[:, -1]


# ---------------------------------------------------------
# 3. LU DECOMPOSITION
# ---------------------------------------------------------
def lu_decomposition(A):
    """
    Decomposes A into:

        A = L @ U

    L = Lower triangular matrix
    U = Upper triangular matrix
    """

    A = A.astype(float).copy()

    n = A.shape[0]

    # Start L as the identity matrix
    L = np.eye(n)

    # U initially contains A
    U = A.copy()

    # Eliminate elements below each pivot
    for i in range(n):

        for j in range(i + 1, n):

            factor = U[j, i] / U[i, i]

            # Store the elimination factor in L
            L[j, i] = factor

            # Eliminate the element below the pivot
            U[j, i:] = (
                U[j, i:] - factor * U[i, i:]
            )

    return L, U


# ---------------------------------------------------------
# 4. FORWARD SUBSTITUTION
# ---------------------------------------------------------
def forward_substitution(L, b):
    """
    Solves Ly = b.

    Since L is lower triangular, the values of y
    can be calculated from top to bottom.
    """

    n = len(b)

    y = np.zeros(n)

    for i in range(n):

        y[i] = (
            b[i] - np.dot(L[i, :i], y[:i])
        ) / L[i, i]

    return y


# ---------------------------------------------------------
# 5. BACK SUBSTITUTION
# ---------------------------------------------------------
def back_substitution(U, y):
    """
    Solves Ux = y.

    Since U is upper triangular, the values of x
    are calculated from bottom to top.
    """

    n = len(y)

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        x[i] = (
            y[i] - np.dot(U[i, i + 1:], x[i + 1:])
        ) / U[i, i]

    return x


# ---------------------------------------------------------
# 6. LU SOLVER
# ---------------------------------------------------------
def lu_solve(A, b):
    """
    Solves Ax = b using LU decomposition.

    First:
        A = LU

    Then:
        Ly = b
        Ux = y
    """

    L, U = lu_decomposition(A)

    # Solve Ly = b
    y = forward_substitution(L, b)

    # Solve Ux = y
    x = back_substitution(U, y)

    return x, L, U


# =========================================================
# MAIN PROGRAM
# =========================================================

# ---------------------------------------------------------
# Problem:
#
# A manufacturing unit produces four products:
# P1, P2, P3 and P4.
#
# The four resource constraints give the system Ax = b.
# ---------------------------------------------------------

A = np.array([
    [2, 1, 3, 1],
    [1, 2, 1, 2],
    [3, 1, 1, 2],
    [1, 3, 2, 1]
], dtype=float)

b = np.array(
    [25, 20, 26, 25],
    dtype=float
)


# ---------------------------------------------------------
# Solve using Gaussian Elimination
# ---------------------------------------------------------

start = time.perf_counter()

x_gaussian = gaussian_elimination(A, b)

gaussian_time = time.perf_counter() - start


# ---------------------------------------------------------
# Solve using RREF
# ---------------------------------------------------------

start = time.perf_counter()

x_rref = rref_solve(A, b)

rref_time = time.perf_counter() - start


# ---------------------------------------------------------
# Solve using LU Decomposition
# ---------------------------------------------------------

start = time.perf_counter()

x_lu, L, U = lu_solve(A, b)

lu_time = time.perf_counter() - start


# ---------------------------------------------------------
# Solve using NumPy
# Used as the reference solution for comparison.
# ---------------------------------------------------------

start = time.perf_counter()

x_numpy = np.linalg.solve(A, b)

numpy_time = time.perf_counter() - start


# =========================================================
# RESULTS
# =========================================================

print("\n========== LINEAR SYSTEM ==========")

print("\nMatrix A:")
print(A)

print("\nVector b:")
print(b)


print("\n========== SOLUTIONS ==========")

print("\nGaussian Elimination:")
print(x_gaussian)

print("\nRREF:")
print(x_rref)

print("\nLU Decomposition:")
print(x_lu)

print("\nNumPy:")
print(x_numpy)


# =========================================================
# ERROR COMPARISON
# =========================================================

print("\n========== ERROR COMPARISON ==========")

print(
    "Gaussian Error:",
    np.linalg.norm(x_gaussian - x_numpy)
)

print(
    "RREF Error:",
    np.linalg.norm(x_rref - x_numpy)
)

print(
    "LU Error:",
    np.linalg.norm(x_lu - x_numpy)
)


# =========================================================
# EXECUTION TIME
# =========================================================

print("\n========== EXECUTION TIMES ==========")

print(
    "Gaussian:",
    gaussian_time,
    "seconds"
)

print(
    "RREF:",
    rref_time,
    "seconds"
)

print(
    "LU:",
    lu_time,
    "seconds"
)

print(
    "NumPy:",
    numpy_time,
    "seconds"
)


# =========================================================
# LU VERIFICATION
# =========================================================

print("\n========== LU VERIFICATION ==========")

print("\nL:")
print(L)

print("\nU:")
print(U)

print("\nL @ U:")
print(L @ U)

print("\nOriginal A:")
print(A)