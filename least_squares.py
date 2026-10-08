import numpy as np
import matplotlib.pyplot as plt


def solve_least_squares(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    AtA = A.T @ A
    Atb = A.T @ b

    x_hat = np.linalg.solve(AtA, Atb)
    b_hat = A @ x_hat
    residuals = b - b_hat
    residual_norm = np.linalg.norm(residuals)

    return x_hat, b_hat, residuals, residual_norm


def solve_least_squares_qr(Q, R, b):
    b = np.asarray(b, dtype=float)

    Qtb = Q.T @ b
    x_hat = np.linalg.solve(R, Qtb)
    b_hat = Q @ Qtb
    residuals = b - b_hat
    residual_norm = np.linalg.norm(residuals)

    return x_hat, b_hat, residuals, residual_norm


def plot_regression_and_residuals(
    x_vals,
    b,
    b_hat,
    residuals,
    filename="least_squares_demo.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        x_vals,
        b,
        color="dodgerblue",
        edgecolor="k",
        label="Actual Data ($b$)",
        zorder=3
    )

    axes[0].plot(
        x_vals,
        b_hat,
        color="crimson",
        linewidth=2,
        label=r"Least Squares Line ($\hat{b}$)"
    )

    axes[0].set_title("Orthogonal Projection onto Col(A)")
    axes[0].set_xlabel("Independent Variable")
    axes[0].set_ylabel("Response")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].stem(
        x_vals,
        residuals,
        linefmt="firebrick",
        markerfmt="ro",
        basefmt="k-"
    )

    axes[1].set_title(r"Residual Errors ($e = b - \hat{b}$)")
    axes[1].set_xlabel("Data Points")
    axes[1].set_ylabel("Error Margin")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()