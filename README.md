# Linear Algebra Mini Project

## Application of Linear Algebra Techniques to a Manufacturing Resource Allocation Problem

---

## Problem Statement

A manufacturing unit produces four products: **P1, P2, P3, and P4**.

The resource constraints are represented as a system of linear equations:

**Ax = b**

The project applies different linear algebra techniques to solve and analyze this system.

---

## Dataset

### Coefficient Matrix A

| P1 | P2 | P3 | P4 |
| -: | -: | -: | -: |
|  2 |  1 |  3 |  1 |
|  1 |  2 |  1 |  2 |
|  3 |  1 |  1 |  2 |
|  1 |  3 |  2 |  1 |

### Resource Vector b

| Value |
| ----: |
|    25 |
|    20 |
|    26 |
|    25 |

---

## Methods

The project explores the following techniques:

* **Gaussian Elimination**
* **Reduced Row Echelon Form (RREF)**
* **LU Decomposition**
* **Rank and Nullity**
* **Linear Independence**
* **Gram-Schmidt Orthogonalization**
* **QR Factorization**
* **Least Squares**
* **Normal Equations**
* **QR-Based Least Squares**

---

## Project Stages

### Stage 1: Direct Methods

The system is solved using:

* Gaussian Elimination
* RREF
* LU Decomposition
* NumPy reference solution

The solutions obtained from the different methods are compared using numerical error.

### Stage 2: Matrix Structure and Orthogonalization

The coefficient matrix is analyzed using:

* Rank
* Nullity
* Consistency
* Linear independence
* Basis extraction
* Gram-Schmidt Orthogonalization
* QR Factorization

The QR factorization is verified by checking:

* **A = QR**
* **QᵀQ = I**

### Stage 3: Least Squares Application

A synthetic dataset with measurement noise is generated to demonstrate a practical least-squares application.

The project:

* Constructs a design matrix for a linear model
* Estimates slope and intercept using normal equations
* Solves the same least-squares problem using QR factorization
* Compares the two solutions
* Calculates residual errors
* Generates regression and residual plots

---

## Objectives

The main objectives of the project are to:

1. Solve the given system using different direct methods.
2. Analyze the structure and properties of the coefficient matrix.
3. Examine linear independence, rank, and nullity.
4. Construct an orthonormal basis using Gram-Schmidt Orthogonalization.
5. Factorize the matrix using QR decomposition.
6. Apply least-squares methods to a noisy dataset.
7. Compare and verify the obtained solutions.

---

## Final Solution

The system has the solution:

**x = [5, 4, 3, 2]**

---

## Project Structure

```text
Mathematics-Mini-Project/
│
├── main.py
├── program.py
├── diagnostics.py
├── independence.py
├── gram_schmidt.py
├── qr_factorization.py
├── least_squares.py
└── README.md
```

---

## How to Run

Activate the virtual environment and run:

```bash
python main.py
```

The program executes all three stages and generates the regression and residual plots for the least-squares application.
