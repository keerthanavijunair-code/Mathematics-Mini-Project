import numpy as np

from program import (
    gaussian_elimination,
    rref_solve,
    lu_solve
)

from diagnostics import diagnose, rref

from independence import get_basis

from gram_schmidt import (
    gram_schmidt,
    normalize
)

from qr_factorization import (
    qr_factorization,
    verify_qr,
    verify_orthonormal,
    qr_solve
)

from least_squares import (
    solve_least_squares,
    solve_least_squares_qr,
    plot_regression_and_residuals
)


# =========================================================
# COMMON DATASET
# =========================================================

A = np.array([
    [2, 1, 3, 1],
    [1, 2, 1, 2],
    [3, 1, 1, 2],
    [1, 3, 2, 1]
], dtype=float)

b = np.array([
    25,
    20,
    26,
    25
], dtype=float)


# =========================================================
# HELPER FUNCTION
# =========================================================

def print_matrix(name, matrix):
    print(f"\n{name}:")
    print(np.round(matrix, 6))


# =========================================================
# STAGE 1
# DIRECT METHODS
# =========================================================

def direct_methods():

    print("\n")
    print("=" * 65)
    print("STAGE 1: DIRECT METHODS")
    print("=" * 65)

    print("\nProblem:")
    print("Solve the manufacturing system Ax = b.")

    print_matrix("Matrix A", A)
    print_matrix("Vector b", b)

    # -----------------------------------------------------
    # Gaussian Elimination
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("GAUSSIAN ELIMINATION")
    print("----------------------------------------")

    x_gaussian = gaussian_elimination(A, b)

    print_matrix("Solution x", x_gaussian)

    # -----------------------------------------------------
    # RREF
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("RREF")
    print("----------------------------------------")

    x_rref = rref_solve(A, b)

    print_matrix("Solution x", x_rref)

    print_matrix(
        "RREF of [A | b]",
        rref(np.column_stack((A, b)))
    )

    # -----------------------------------------------------
    # LU Decomposition
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("LU DECOMPOSITION")
    print("----------------------------------------")

    x_lu, L, U = lu_solve(A, b)

    print_matrix("L", L)
    print_matrix("U", U)

    print_matrix("Solution x", x_lu)

    print("\nLU Verification:")
    print("L @ U = A:", np.allclose(L @ U, A))

    # -----------------------------------------------------
    # NumPy Reference
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("NUMPY REFERENCE")
    print("----------------------------------------")

    x_numpy = np.linalg.solve(A, b)

    print_matrix("NumPy solution", x_numpy)

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("SOLUTION COMPARISON")
    print("----------------------------------------")

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

    print("\nExpected solution:")
    print("[5, 4, 3, 2]")


# =========================================================
# STAGE 2
# MATRIX STRUCTURE AND QR
# =========================================================

def matrix_structure():

    print("\n")
    print("=" * 65)
    print("STAGE 2: MATRIX STRUCTURE AND ORTHOGONALIZATION")
    print("=" * 65)

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("MATRIX DIAGNOSTICS")
    print("----------------------------------------")

    results = diagnose(A, b)

    print(
        "Matrix size:",
        results["rows"],
        "x",
        results["columns"]
    )

    print(
        "Rank(A):",
        results["rank"]
    )

    print(
        "Nullity(A):",
        results["nullity"]
    )

    print(
        "Rank([A|b]):",
        results["rank_augmented"]
    )

    print(
        "Consistency:",
        results["consistency"]
    )

    print(
        "System type:",
        results["system_type"]
    )

    print(
        "Solution type:",
        results["solution_type"]
    )

    # -----------------------------------------------------
    # Linear Independence
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("LINEAR INDEPENDENCE")
    print("----------------------------------------")

    basis, selected, redundant = get_basis(A)

    print(
        "Independent columns:",
        [i + 1 for i in selected]
    )

    print(
        "Redundant columns:",
        [i + 1 for i in redundant]
    )

    print_matrix(
        "Basis formed from independent columns",
        basis
    )

    # -----------------------------------------------------
    # Gram-Schmidt
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("GRAM-SCHMIDT")
    print("----------------------------------------")

    orthogonal_basis = gram_schmidt(basis)

    orthonormal_basis = normalize(
        orthogonal_basis
    )

    print_matrix(
        "Orthogonal basis",
        orthogonal_basis
    )

    print_matrix(
        "Orthonormal basis Q",
        orthonormal_basis
    )

    # -----------------------------------------------------
    # QR Factorization
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("QR FACTORIZATION")
    print("----------------------------------------")

    Q, R = qr_factorization(A)

    print_matrix("Q", Q)
    print_matrix("R", R)

    print("\nQR Verification:")
    print(
        "A = QR:",
        verify_qr(A, Q, R)
    )

    print(
        "Q^T Q = I:",
        verify_orthonormal(Q)
    )

    print_matrix(
        "Q @ R",
        Q @ R
    )

    # -----------------------------------------------------
    # QR-Based Exact Solution
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("QR-BASED SOLUTION")
    print("----------------------------------------")

    x_qr, Q_solve, R_solve = qr_solve(A, b)

    print_matrix(
        "Solution x",
        x_qr
    )

    print(
        "QR solution:",
        x_qr
    )


# =========================================================
# STAGE 3
# LEAST SQUARES APPLICATION
# =========================================================

def least_squares_stage():

    print("\n")
    print("=" * 65)
    print("STAGE 3: LEAST SQUARES APPLICATION")
    print("=" * 65)

    print(
        "\nGenerating a real-world style dataset "
        "with measurement noise..."
    )

    np.random.seed(42)

    x_raw = np.linspace(1, 10, 20)

    noise = np.random.normal(
        0,
        1.2,
        size=x_raw.shape
    )

    y_raw = (
        2.5 * x_raw
        + 5.0
        + noise
    )

    # Design matrix:
    #
    # y = mx + c
    #
    # A = [x  1]

    A_ls = np.column_stack(
        (
            x_raw,
            np.ones_like(x_raw)
        )
    )

    b_ls = y_raw

    print(
        "\nMatrix A dimensions:",
        A_ls.shape[0],
        "equations x",
        A_ls.shape[1],
        "unknowns"
    )

    print_matrix(
        "First 5 rows of A",
        A_ls[:5]
    )

    print_matrix(
        "First 5 values of b",
        b_ls[:5]
    )

    # -----------------------------------------------------
    # Normal Equations
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("NORMAL EQUATIONS")
    print("----------------------------------------")

    x_hat, b_hat, residuals, residual_norm = (
        solve_least_squares(
            A_ls,
            b_ls
        )
    )

    print(
        "Estimated slope:",
        x_hat[0]
    )

    print(
        "Estimated intercept:",
        x_hat[1]
    )

    print_matrix(
        "Projected vector b_hat",
        b_hat
    )

    print(
        "\nResidual norm ||b - A*x_hat||:",
        residual_norm
    )

    # -----------------------------------------------------
    # QR-Based Least Squares
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("QR-BASED LEAST SQUARES")
    print("----------------------------------------")

    Q_ls, R_ls = qr_factorization(A_ls)

    (
        x_hat_qr,
        b_hat_qr,
        residuals_qr,
        residual_norm_qr
    ) = solve_least_squares_qr(
        Q_ls,
        R_ls,
        b_ls
    )

    print(
        "QR estimated slope:",
        x_hat_qr[0]
    )

    print(
        "QR estimated intercept:",
        x_hat_qr[1]
    )

    print(
        "QR residual norm:",
        residual_norm_qr
    )

    print(
        "\nNormal Equations = QR:",
        np.allclose(
            x_hat,
            x_hat_qr
        )
    )

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("VISUALIZATION")
    print("----------------------------------------")

    print(
        "Generating regression and residual plots..."
    )

    plot_regression_and_residuals(
        x_raw,
        b_ls,
        b_hat,
        residuals
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 65)
    print("PES UNIVERSITY")
    print("LINEAR ALGEBRA MINI-PROJECT")
    print("=" * 65)

    direct_methods()

    matrix_structure()

    least_squares_stage()

    print("\n")
    print("=" * 65)
    print("PROJECT EXECUTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()