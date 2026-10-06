import numpy as np

from diagnostics import diagnose, rref
from independence import get_basis
from gram_schmidt import gram_schmidt, normalize
from qr_factorization import (
    qr_factorization,
    verify_qr,
    verify_orthonormal,
    qr_solve
)


def print_matrix(name, matrix):
    print(f"\n{name}:")
    print(np.round(matrix, 6))


def main():
    # Test case.
    # Replace only A and b when your team gets the real dataset.

    a = np.array([
        [2, 1, 1],
        [4, 2, 2],
        [1, 1, 2]
    ], dtype=float)

    b = np.array([
        5,
        10,
        6
    ], dtype=float)

    print("========================================")
    print("       TEAMMATE 2")
    print("MATRIX DIAGNOSTICS AND ORTHOGONALIZATION")
    print("========================================")

    print_matrix("A", a)
    print_matrix("b", b)

    print("\n----------------------------------------")
    print("MATRIX DIAGNOSTICS")
    print("----------------------------------------")

    results = diagnose(a, b)

    print("matrix size:", results["rows"], "x", results["columns"])
    print("rank(A):", results["rank"])
    print("nullity(A):", results["nullity"])
    print("rank([A|b]):", results["rank_augmented"])
    print("consistency:", results["consistency"])
    print("system type:", results["system_type"])
    print("solution type:", results["solution_type"])

    print_matrix("RREF(A)", rref(a))

    print("\n----------------------------------------")
    print("LINEAR INDEPENDENCE")
    print("----------------------------------------")

    basis, selected, redundant = get_basis(a)

    print("independent columns:", [i + 1 for i in selected])
    print("redundant columns:", [i + 1 for i in redundant])

    print_matrix("Selected basis", basis)

    print("\n----------------------------------------")
    print("GRAM-SCHMIDT")
    print("----------------------------------------")

    u = gram_schmidt(basis)
    q = normalize(u)

    print_matrix("Orthogonal basis", u)
    print_matrix("Orthonormal basis Q", q)

    print("\n----------------------------------------")
    print("QR FACTORIZATION")
    print("----------------------------------------")

    q, r = qr_factorization(basis)

    print_matrix("Q", q)
    print_matrix("R", r)

    print("\nQR verification:")
    print("A = QR:", verify_qr(basis, q, r))

    print("\nOrthonormal verification:")
    print("Q^T Q = I:", verify_orthonormal(q))

    print_matrix("QR", q @ r)

    print("\n----------------------------------------")
    print("QR-BASED SOLUTION")
    print("----------------------------------------")

    if (
        basis.shape[0] == basis.shape[1]
        and basis.shape[1] == len(b)
    ):
        try:
            x, q_solve, r_solve = qr_solve(basis, b)
            print_matrix("Q", q_solve)
            print_matrix("R", r_solve)
            print_matrix("x", x)
        except ValueError as e:
            print("QR solve skipped:", e)
    else:
        print(
            "QR solve skipped because the selected basis is not "
            "a square system matching b."
        )
        print(
            "For rectangular systems, Teammate 3 can use the "
            "least-squares stage."
        )

    print("\n========================================")
    print("END OF TEAMMATE 2")
    print("========================================")


if __name__ == "__main__":
    main()
