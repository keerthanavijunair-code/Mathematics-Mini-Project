"""
Linear Algebra Mini-Project - desktop application (PySide6)
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import matplotlib

matplotlib.use("QtAgg")  # must be set before least_squares imports pyplot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

# ---- your existing backend (unchanged) --------------------------------
from program import gaussian_elimination, rref_solve, lu_solve
from diagnostics import diagnose, rref
from independence import get_basis
from gram_schmidt import gram_schmidt, normalize
from qr_factorization import (
    qr_factorization, verify_qr, verify_orthonormal, qr_solve,
)
from least_squares import solve_least_squares, solve_least_squares_qr

# ---- sample system from main.py ---------------------------------------
SAMPLE_A = np.array([[2, 1, 3, 1],
                     [1, 2, 1, 2],
                     [3, 1, 1, 2],
                     [1, 3, 2, 1]], dtype=float)
SAMPLE_B = np.array([25, 20, 26, 25], dtype=float)

# =======================================================================
# THEMES
# =======================================================================

THEMES = {
    "light": dict(
        bg="#f3f5f9", surface="#ffffff", surface2="#eaeef5", alt="#f6f8fc",
        border="#d8dee9", text="#18233b", muted="#5d6a84",
        accent="#1f3a68", accent_hover="#2c4f8c", on_accent="#ffffff",
        mark="#e8892b", ok="#1b7f5c", bad="#c0392b",
        side="#14213d", side_hover="#1f3157", side_text="#b8c4dd",
        pt="#1f3a68", line="#c0392b",
    ),
    "dark": dict(
        bg="#0e1525", surface="#161f37", surface2="#1f2b4a", alt="#1a2542",
        border="#2a3858", text="#e7ecf6", muted="#94a1bf",
        accent="#7c9ceb", accent_hover="#97b1f0", on_accent="#0e1525",
        mark="#f0a04b", ok="#4cc79a", bad="#f08072",
        side="#0a101f", side_hover="#17223c", side_text="#9aa8c6",
        pt="#7c9ceb", line="#f08072",
    ),
}


def stylesheet(t):
    return f"""
    * {{ font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", "Noto Sans", Arial, sans-serif;
         font-size: 13px; }}
    QMainWindow, QWidget#root {{ background: {t['bg']}; }}
    QLabel {{ color: {t['text']}; background: transparent; }}
    QScrollArea {{ background: transparent; border: none; }}

    QFrame#sidebar {{ background: {t['side']}; }}
    QLabel#brand {{ color: #ffffff; font-size: 17px; font-weight: 700; }}
    QLabel#brandSub {{ color: {t['side_text']}; font-size: 12px; }}
    QLabel#footer {{ color: {t['side_text']}; font-size: 11px; }}
    QPushButton[nav="true"] {{
        background: transparent; color: {t['side_text']}; text-align: left;
        border: none; border-left: 3px solid transparent; border-radius: 0;
        padding: 11px 18px; font-size: 13px; }}
    QPushButton[nav="true"]:hover {{ background: {t['side_hover']}; color: #ffffff; }}
    QPushButton[nav="true"]:checked {{
        background: {t['side_hover']}; color: #ffffff;
        border-left: 3px solid {t['mark']}; font-weight: 600; }}
    QPushButton[side="true"] {{
        background: transparent; color: {t['side_text']}; text-align: left;
        border: 1px solid {t['side_hover']}; border-radius: 6px; padding: 8px 12px; }}
    QPushButton[side="true"]:hover {{ background: {t['side_hover']}; color: #ffffff; }}

    QLabel#h1 {{ font-size: 24px; font-weight: 700; }}
    QLabel#sub {{ color: {t['muted']}; font-size: 13px; }}

    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; }}
    QLabel#cardTitle {{ font-size: 15px; font-weight: 600; }}
    QLabel#note {{ color: {t['muted']}; }}
    QLabel#caption {{ color: {t['muted']}; font-size: 12px; font-weight: 600; margin-top: 4px; }}
    QLabel#ok {{ color: {t['ok']}; font-weight: 600; }}
    QLabel#bad {{ color: {t['bad']}; font-weight: 600; }}
    QLabel#empty {{ color: {t['muted']}; font-size: 14px; padding: 40px 0; }}

    QFrame#viva {{ background: {t['surface2']}; border: none;
                   border-left: 3px solid {t['mark']}; border-radius: 6px; }}
    QLabel#vivaHead {{ font-weight: 700; }}
    QLabel#vivaText {{ color: {t['muted']}; }}

    QPushButton {{ background: {t['surface']}; color: {t['text']};
        border: 1px solid {t['border']}; border-radius: 6px; padding: 7px 16px; }}
    QPushButton:hover {{ background: {t['surface2']}; }}
    QPushButton#primary {{ background: {t['accent']}; color: {t['on_accent']};
        border: none; font-weight: 600; padding: 8px 20px; }}
    QPushButton#primary:hover {{ background: {t['accent_hover']}; }}

    QSpinBox, QDoubleSpinBox {{ background: {t['surface']}; color: {t['text']};
        border: 1px solid {t['border']}; border-radius: 5px; padding: 4px 6px; min-width: 70px; }}

    QTableWidget {{ background: {t['surface']}; alternate-background-color: {t['alt']};
        color: {t['text']}; border: 1px solid {t['border']}; border-radius: 6px;
        gridline-color: {t['border']}; selection-background-color: {t['surface2']};
        selection-color: {t['text']}; }}
    QHeaderView::section {{ background: {t['surface2']}; color: {t['muted']};
        border: none; border-right: 1px solid {t['border']};
        border-bottom: 1px solid {t['border']}; padding: 4px 10px; font-weight: 600; }}
    QTableCornerButton::section {{ background: {t['surface2']}; border: none; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 5px; min-height: 30px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 5px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

    QStatusBar {{ background: {t['surface']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
    QMessageBox {{ background: {t['surface']}; }}
    QToolTip {{ background: {t['surface2']}; color: {t['text']}; border: 1px solid {t['border']}; }}
    """


# =======================================================================
# SMALL HELPERS
# =======================================================================

def fmt(v):
    """Format a number for display."""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if round(v, 4) == 0:
        v = 0.0
    return f"{v:.4f}"


def vec_str(x):
    return "[" + ", ".join(fmt(v) for v in np.ravel(x)) + "]"


def truthy(result):
    """verify_* helpers may return a bool or a tuple whose first item is a bool."""
    if isinstance(result, (tuple, list)):
        result = result[0]
    return bool(np.all(result))


def require_finite(x, method):
    """Backend solvers return nan/inf (instead of raising) on a zero pivot."""
    if not np.all(np.isfinite(x)):
        raise ArithmeticError(
            f"{method} hit a zero pivot, so the result is not a number. "
            "The matrix is probably singular (or, for LU, needs row swaps, "
            "which this LU does not do)."
        )


def fit_table(t, max_h=300, max_w=980):
    """Size a table to its content (with a cap, after which it scrolls)."""
    t.resizeColumnsToContents()
    total = 0
    for j in range(t.columnCount()):
        w = max(t.columnWidth(j) + 14, 76)
        t.setColumnWidth(j, w)
        total += w
    row_h = 30
    t.verticalHeader().setDefaultSectionSize(row_h)
    vw = 0
    if not t.verticalHeader().isHidden():
        vw = max(t.verticalHeader().sizeHint().width(), 36)
    w = vw + total + 4
    h = t.horizontalHeader().sizeHint().height() + row_h * t.rowCount() + 6
    if w > max_w:
        w, h = max_w, h + 16
    if h > max_h:
        h, w = max_h, w + 16
    t.setFixedSize(w, h)


def make_table(rows, col_labels=None, row_labels=None, vheader=True):
    n = len(rows)
    m = len(rows[0]) if n else 0
    t = QTableWidget(n, m)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionMode(QTableWidget.NoSelection)
    t.setFocusPolicy(Qt.NoFocus)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    mono.setPointSize(10)
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(mono)
            t.setItem(i, j, item)
    t.setHorizontalHeaderLabels(col_labels or [str(j + 1) for j in range(m)])
    if vheader:
        t.setVerticalHeaderLabels(row_labels or [str(i + 1) for i in range(n)])
    else:
        t.verticalHeader().setVisible(False)
    fit_table(t)
    return t


# =======================================================================
# REUSABLE WIDGETS
# =======================================================================

class Card(QFrame):
    """A titled result block. Also records a plain-text copy for the report."""

    def __init__(self, title, note=None):
        super().__init__()
        self.setObjectName("card")
        self.failed = False
        self.report = [title, "-" * len(title)]
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(18, 14, 18, 16)
        self.lay.setSpacing(8)
        head = QLabel(title)
        head.setObjectName("cardTitle")
        self.lay.addWidget(head)
        if note:
            n = QLabel(note)
            n.setObjectName("note")
            n.setWordWrap(True)
            self.lay.addWidget(n)

    def add_left(self, widget):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addStretch()
        self.lay.addLayout(row)

    def add_text(self, text, name=None):
        lab = QLabel(text)
        if name:
            lab.setObjectName(name)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lay.addWidget(lab)
        self.report.append(text)

    def add_status(self, text, ok):
        self.add_text(("\u2713 " if ok else "\u2717 ") + text, "ok" if ok else "bad")

    def add_matrix(self, name, M, rows=None, cols=None):
        M = np.asarray(M, dtype=float)
        if M.ndim == 0:
            M = M.reshape(1, 1)
        if M.ndim == 1:
            M = M.reshape(-1, 1)
        cap = QLabel(f"{name}   ({M.shape[0]} \u00d7 {M.shape[1]})")
        cap.setObjectName("caption")
        self.lay.addWidget(cap)
        self.report.append(f"{name}:")
        if M.size == 0:
            self.add_text("(empty)", "note")
            return
        rows_txt = [[fmt(v) for v in r] for r in M]
        self.add_left(make_table(rows_txt, cols, rows))
        self.report.append(np.array2string(np.round(M, 4), precision=4, suppress_small=True))
        self.report.append("")

    def add_table(self, headers, rows):
        self.add_left(make_table(rows, headers, vheader=False))
        self.report.append(" | ".join(headers))
        self.report.extend(" | ".join(r) for r in rows)
        self.report.append("")

    def fail(self, exc):
        self.failed = True
        self.add_text(f"Could not compute this step: {exc}", "bad")


class VivaStrip(QFrame):
    """Concept -> Purpose -> Outcome, the structure examiners ask for."""

    def __init__(self, concept, purpose, outcome):
        super().__init__()
        self.setObjectName("viva")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(24)
        for head, text in (("Concept", concept), ("Purpose", purpose), ("Outcome", outcome)):
            col = QVBoxLayout()
            col.setSpacing(2)
            h = QLabel(head)
            h.setObjectName("vivaHead")
            b = QLabel(text)
            b.setObjectName("vivaText")
            b.setWordWrap(True)
            col.addWidget(h)
            col.addWidget(b)
            col.addStretch()
            lay.addLayout(col, 1)


# =======================================================================
# PAGES
# =======================================================================

class Page(QWidget):
    title = ""
    subtitle = ""
    run_label = None          # None = page has no Run button
    empty_text = None
    viva = ("", "", "")

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.cards = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 24, 30, 12)
        outer.setSpacing(14)

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        h1 = QLabel(self.title)
        h1.setObjectName("h1")
        sub = QLabel(self.subtitle)
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        titles.addWidget(h1)
        titles.addWidget(sub)
        head.addLayout(titles, 1)
        self.head_row = head
        if self.run_label:
            btn = QPushButton(self.run_label)
            btn.setObjectName("primary")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda: self.run())
            head.addWidget(btn, 0, Qt.AlignTop)
        outer.addLayout(head)

        outer.addWidget(VivaStrip(*self.viva))

        self.extra = QVBoxLayout()          # optional controls above results
        outer.addLayout(self.extra)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setAutoFillBackground(False)
        self.scroll.viewport().setAutoFillBackground(False)
        self.body = QVBoxLayout(body)
        self.body.setContentsMargins(0, 0, 8, 12)
        self.body.setSpacing(14)
        self.body.addStretch()
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        self.show_empty()

    # -- layout helpers ------------------------------------------------
    def show_empty(self):
        if self.empty_text:
            lab = QLabel(self.empty_text)
            lab.setObjectName("empty")
            lab.setAlignment(Qt.AlignCenter)
            lab.setWordWrap(True)
            self.body.insertWidget(0, lab)

    def clear(self):
        while self.body.count() > 1:
            item = self.body.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.cards.clear()

    def add_card(self, card):
        self.body.insertWidget(self.body.count() - 1, card)
        self.cards.append(card)

    def section(self, title, note, fn):
        """Build one card; a failure in one step never blocks the others."""
        card = Card(title, note)
        try:
            fn(card)
        except Exception as exc:       # noqa: BLE001 - shown to the user
            card.fail(exc)
        self.add_card(card)

    # -- running -------------------------------------------------------
    def compute(self):
        raise NotImplementedError

    def run(self, interactive=True):
        self.clear()
        try:
            self.compute()
        except Exception as exc:       # noqa: BLE001
            self.show_empty()
            if interactive:
                QMessageBox.warning(self, "Cannot run this stage", str(exc))
            self.win.statusBar().showMessage(f"{self.title}: {exc}")
            return str(exc)
        bad = sum(c.failed for c in self.cards)
        msg = f"{self.title}: finished"
        if bad:
            msg += f" ({bad} step{'s' if bad > 1 else ''} could not run)"
        self.win.statusBar().showMessage(msg)
        return None

    def report_text(self):
        return "\n".join("\n".join(c.report) + "\n" for c in self.cards)


# ---- Page 1: data -----------------------------------------------------

class DataPage(Page):
    title = "Data and matrix"
    subtitle = "Enter the linear system Ax = b. Every other stage reads from this page."
    viva = (
        "Matrix representation of a system of linear equations.",
        "Each row is one equation and each column is one unknown, so the whole "
        "problem becomes the single object Ax = b.",
        "A coefficient matrix A and a right-hand side b, ready for elimination, "
        "factorization and projection.",
    )

    def __init__(self, win):
        super().__init__(win)
        card = Card("Linear system", "Double-click a cell to edit. The last column is the right-hand side b.")
        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 12)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 12)
        reset = QPushButton("Reset to sample system")
        reset.setCursor(Qt.PointingHandCursor)
        reset.clicked.connect(self.reset)
        controls.addWidget(QLabel("Equations"))
        controls.addWidget(self.rows_spin)
        controls.addWidget(QLabel("Unknowns"))
        controls.addWidget(self.cols_spin)
        controls.addStretch()
        controls.addWidget(reset)
        card.lay.addLayout(controls)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        card.add_left(self.table)
        self.info = QLabel()
        self.info.setObjectName("note")
        card.lay.addWidget(self.info)
        self.body.insertWidget(0, card)

        self.rows_spin.valueChanged.connect(self.resize_table)
        self.cols_spin.valueChanged.connect(self.resize_table)
        self.reset()

    def _texts(self):
        t = self.table
        return [[(t.item(i, j).text() if t.item(i, j) else "0") for j in range(t.columnCount())]
                for i in range(t.rowCount())]

    def _fill(self, texts, n, m):
        t = self.table
        t.setRowCount(n)
        t.setColumnCount(m + 1)
        t.setHorizontalHeaderLabels([f"x{j + 1}" for j in range(m)] + ["b"])
        t.setVerticalHeaderLabels([f"eq {i + 1}" for i in range(n)])
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(10)
        for i in range(n):
            for j in range(m + 1):
                old = texts[i][j] if i < len(texts) and j < len(texts[i]) else "0"
                item = QTableWidgetItem(old)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(mono)
                t.setItem(i, j, item)
        fit_table(t, max_h=380, max_w=900)
        if n == m:
            kind = "square system"
        elif n > m:
            kind = "overdetermined: more equations than unknowns"
        else:
            kind = "underdetermined: fewer equations than unknowns"
        self.info.setText(f"{n} equations \u00d7 {m} unknowns ({kind})")

    def resize_table(self):
        self._fill(self._texts(), self.rows_spin.value(), self.cols_spin.value())

    def reset(self):
        n, m = SAMPLE_A.shape
        for spin, v in ((self.rows_spin, n), (self.cols_spin, m)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        texts = [[fmt(v).rstrip("0").rstrip(".") for v in list(SAMPLE_A[i]) + [SAMPLE_B[i]]]
                 for i in range(n)]
        self._fill(texts, n, m)

    def get_system(self):
        n, m = self.table.rowCount(), self.table.columnCount() - 1
        M = np.zeros((n, m + 1))
        for i, row in enumerate(self._texts()):
            for j, text in enumerate(row):
                try:
                    M[i, j] = float(text.strip())
                except ValueError:
                    name = "b" if j == m else f"x{j + 1}"
                    raise ValueError(f"Equation {i + 1}, column {name} is not a number: '{text}'.")
        return M[:, :m], M[:, m]


# ---- Page 2: direct methods -------------------------------------------

class DirectPage(Page):
    title = "Direct methods"
    subtitle = "Solve Ax = b three ways and check each against NumPy."
    run_label = "Run direct methods"
    empty_text = "Press Run direct methods to solve the system."
    viva = (
        "Gaussian elimination, RREF and LU decomposition.",
        "Reduce A to a simpler form so the unknowns can be read off, and confirm "
        "that independent methods agree.",
        "The same solution vector x from every method, matching NumPy to rounding error.",
    )

    def compute(self):
        A, b = self.win.system()
        n, m = A.shape
        if n != m:
            raise ValueError("Direct methods need a square system. Set equations equal to "
                             "unknowns on the Data page.")
        names = [f"x{i + 1}" for i in range(m)]
        sols = {}

        def gauss(c):
            x = np.asarray(gaussian_elimination(A, b), float).ravel()
            require_finite(x, "Gaussian elimination")
            sols["Gaussian elimination"] = x
            c.add_matrix("Solution x", x, rows=names, cols=["x"])

        def rr(c):
            x = np.asarray(rref_solve(A, b), float).ravel()
            require_finite(x, "RREF")
            sols["RREF"] = x
            c.add_matrix("RREF of [A | b]", rref(np.column_stack((A, b))), cols=names + ["b"])
            c.add_matrix("Solution x", x, rows=names, cols=["x"])

        def lu(c):
            x, L, U = lu_solve(A, b)
            x = np.asarray(x, float).ravel()
            require_finite(x, "LU decomposition")
            sols["LU decomposition"] = x
            c.add_matrix("L (lower triangular)", L)
            c.add_matrix("U (upper triangular)", U)
            c.add_matrix("Solution x", x, rows=names, cols=["x"])
            c.add_status("L @ U reproduces A", bool(np.allclose(np.asarray(L) @ np.asarray(U), A)))

        def ref(c):
            x = np.linalg.solve(A, b)
            sols["NumPy reference"] = x
            c.add_matrix("NumPy solution", x, rows=names, cols=["x"])

        def compare(c):
            if "NumPy reference" not in sols:
                raise RuntimeError("the NumPy reference is missing, so there is nothing to compare against.")
            ref_x = sols["NumPy reference"]
            rows, worst = [], 0.0
            for name, x in sols.items():
                err = float(np.linalg.norm(x - ref_x))
                worst = max(worst, err)
                rows.append([name, vec_str(x), f"{err:.2e}"])
            c.add_table(["Method", "Solution x", "Error vs NumPy"], rows)
            c.add_status("All methods agree with NumPy" if worst < 1e-8
                         else "Some methods differ from NumPy", worst < 1e-8)

        self.section("Gaussian elimination", "Forward elimination, then back-substitution.", gauss)
        self.section("Reduced row echelon form", "Gauss-Jordan elimination on the augmented matrix.", rr)
        self.section("LU decomposition", "Factor A into lower and upper triangular matrices.", lu)
        self.section("NumPy reference", "Library solution used as the ground truth.", ref)
        self.section("Comparison", "Distance of each solution from the reference.", compare)


# ---- Page 3: structure -----------------------------------------------------

class StructurePage(Page):
    title = "Matrix structure"
    subtitle = "Rank, independent columns, orthogonal bases and QR factorization."
    run_label = "Analyze structure"
    empty_text = "Press Analyze structure to inspect the matrix."
    viva = (
        "Rank and nullity, linear independence, basis selection, Gram-Schmidt and QR.",
        "Find out how much of A is redundant, then rebuild its column space "
        "from perpendicular unit vectors.",
        "A basis of independent columns, an orthonormal matrix Q and a verified A = QR.",
    )

    def compute(self):
        A, b = self.win.system()
        state = {}

        def diag(c):
            d = diagnose(A, b)
            c.add_table(["Property", "Value"], [
                ["Matrix size", f"{d['rows']} \u00d7 {d['columns']}"],
                ["Rank of A", fmt(d["rank"])],
                ["Nullity of A", fmt(d["nullity"])],
                ["Rank of [A | b]", fmt(d["rank_augmented"])],
                ["Consistency", str(d["consistency"])],
                ["System shape", str(d["system_type"])],
                ["Solution type", str(d["solution_type"])],
            ])

        def indep(c):
            basis, selected, redundant = get_basis(A)
            state["basis"] = np.asarray(basis, float)
            sel = ", ".join(str(i + 1) for i in selected) or "none"
            red = ", ".join(str(i + 1) for i in redundant) or "none"
            c.add_text(f"Independent columns: {sel}")
            c.add_text(f"Redundant columns: {red}")
            c.add_matrix("Basis from the independent columns", basis,
                         cols=[f"a{i + 1}" for i in selected])

        def gs(c):
            if "basis" not in state:
                raise RuntimeError("the independence step above has to succeed first.")
            orth = np.asarray(gram_schmidt(state["basis"]), float)
            ortho_n = np.asarray(normalize(orth), float)
            c.add_matrix("Orthogonal basis", orth)
            c.add_matrix("Orthonormal basis", ortho_n)
            k = ortho_n.shape[1]
            c.add_status("Columns are orthonormal (Q\u1d40Q = I)",
                         bool(np.allclose(ortho_n.T @ ortho_n, np.eye(k))))

        def qr(c):
            Q, R = qr_factorization(A)
            Q, R = np.asarray(Q, float), np.asarray(R, float)
            c.add_matrix("Q", Q)
            c.add_matrix("R", R)
            c.add_matrix("Q @ R", Q @ R)
            c.add_status("A = QR", truthy(verify_qr(A, Q, R)))
            c.add_status("Q\u1d40Q = I", truthy(verify_orthonormal(Q)))

        def qrs(c):
            x = np.asarray(qr_solve(A, b)[0], float).ravel()
            c.add_matrix("Solution x", x, rows=[f"x{i + 1}" for i in range(len(x))], cols=["x"])

        self.section("Diagnostics", "Size, rank, nullity and consistency of the system.", diag)
        self.section("Linear independence", "Keep only the columns that add a new direction.", indep)
        self.section("Gram-Schmidt orthogonalization", "Turn the basis into perpendicular, then unit-length, vectors.", gs)
        self.section("QR factorization", "A written as an orthonormal Q times an upper-triangular R.", qr)
        self.section("Solution from QR", "Solve Rx = Q\u1d40b by back-substitution.", qrs)


# ---- Page 4: least squares -------------------------------------------------

class LeastSquaresPage(Page):
    title = "Least squares"
    subtitle = "Fit a line to noisy measurements by projecting b onto the column space of A."
    run_label = "Fit line"
    empty_text = "Press Fit line to run the regression."
    viva = (
        "Least squares solution as an orthogonal projection of b onto Col(A).",
        "The data has more equations than unknowns, so Ax = b has no exact solution. "
        "Projection gives the closest one.",
        "A slope and intercept with the smallest possible residual, "
        "found two ways (normal equations and QR) that agree.",
    )

    def __init__(self, win):
        self.csv = None
        self._plot = None
        self.canvas = None
        super().__init__(win)

        box = QFrame()
        box.setObjectName("card")
        row = QHBoxLayout(box)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(8)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(5, 500)
        self.n_spin.setValue(20)
        self.slope = QDoubleSpinBox()
        self.slope.setRange(-100, 100)
        self.slope.setSingleStep(0.1)
        self.slope.setValue(2.5)
        self.icpt = QDoubleSpinBox()
        self.icpt.setRange(-100, 100)
        self.icpt.setSingleStep(0.1)
        self.icpt.setValue(5.0)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0, 20)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.seed.setValue(42)
        for label, w in (("Points", self.n_spin), ("True slope", self.slope),
                         ("Intercept", self.icpt), ("Noise", self.sigma), ("Seed", self.seed)):
            row.addWidget(QLabel(label))
            row.addWidget(w)
        row.addStretch()
        self.src = QLabel("Source: synthetic data")
        self.src.setObjectName("note")
        load = QPushButton("Load CSV")
        load.setCursor(Qt.PointingHandCursor)
        load.clicked.connect(self.load_csv)
        synth = QPushButton("Use synthetic")
        synth.setCursor(Qt.PointingHandCursor)
        synth.clicked.connect(self.use_synthetic)
        row.addWidget(self.src)
        row.addWidget(load)
        row.addWidget(synth)
        self.extra.addWidget(box)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open data", "", "CSV files (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            try:
                d = np.loadtxt(path, delimiter=",", ndmin=2)
            except ValueError:
                d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            if d.shape[1] < 2 or d.shape[0] < 3:
                raise ValueError("The file needs at least 3 rows and 2 columns (x, y).")
        except Exception as exc:       # noqa: BLE001
            QMessageBox.warning(self, "Cannot read file", str(exc))
            return
        self.csv = (d[:, 0], d[:, 1])
        self.src.setText(f"Source: {os.path.basename(path)} ({d.shape[0]} points)")

    def use_synthetic(self):
        self.csv = None
        self.src.setText("Source: synthetic data")

    def data(self):
        if self.csv is not None:
            return self.csv
        np.random.seed(self.seed.value())
        x = np.linspace(1, 10, self.n_spin.value())
        noise = np.random.normal(0, self.sigma.value(), size=x.shape)
        return x, self.slope.value() * x + self.icpt.value() + noise

    def compute(self):
        x, y = self.data()
        A = np.column_stack((x, np.ones_like(x)))
        res = {}

        def design(c):
            c.add_matrix("A (first 5 rows)", A[:5], cols=["x", "1"])
            c.add_matrix("b (first 5 values)", y[:5], cols=["y"])

        def normal(c):
            x_hat, b_hat, resid, rn = solve_least_squares(A, y)
            res["fit"] = (np.asarray(b_hat, float), np.asarray(resid, float))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - float(np.sum(np.square(resid))) / ss_tot if ss_tot else float("nan")
            c.add_table(["Quantity", "Value"], [
                ["Slope", f"{x_hat[0]:.6f}"],
                ["Intercept", f"{x_hat[1]:.6f}"],
                ["Residual norm", f"{rn:.6f}"],
                ["R\u00b2", f"{r2:.6f}"],
            ])
            c.add_matrix("Projected vector b\u0302 (first 5 values)", np.asarray(b_hat)[:5], cols=["b\u0302"])
            res["x_hat"] = np.asarray(x_hat, float)

        def qr_ls(c):
            Q, R = qr_factorization(A)
            x_qr, _, _, rn = solve_least_squares_qr(Q, R, y)
            c.add_table(["Quantity", "Value"], [
                ["Slope", f"{x_qr[0]:.6f}"],
                ["Intercept", f"{x_qr[1]:.6f}"],
                ["Residual norm", f"{rn:.6f}"],
            ])
            if "x_hat" in res:
                c.add_status("Normal equations and QR give the same answer",
                             bool(np.allclose(res["x_hat"], x_qr)))

        def plot(c):
            if "fit" not in res:
                raise RuntimeError("the normal-equations step above has to succeed first.")
            self._plot = (x, y, *res["fit"])
            self.canvas = FigureCanvasQTAgg(Figure(figsize=(10, 4)))
            self.canvas.setMinimumHeight(340)
            c.lay.addWidget(self.canvas)
            self.draw_plot()

        self.section("Design matrix", "Each row is [x, 1], so Ax = b models y = mx + c.", design)
        self.section("Normal equations", "Solve A\u1d40A x = A\u1d40b for the best-fit line.", normal)
        self.section("QR least squares", "The same fit through the orthonormal factor Q.", qr_ls)
        self.section("Fit and residuals", "Left: data and fitted line. Right: error at each point.", plot)

    def draw_plot(self):
        if not self._plot or self.canvas is None:
            return
        t = THEMES[self.win.theme]
        x, y, b_hat, resid = self._plot
        fig = self.canvas.figure
        fig.clear()
        fig.set_facecolor(t["surface"])
        ax1, ax2 = fig.subplots(1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor(t["surface"])
            ax.tick_params(colors=t["muted"])
            for s in ax.spines.values():
                s.set_color(t["border"])
            ax.grid(True, linestyle="--", alpha=0.35, color=t["muted"])
        order = np.argsort(x)
        ax1.scatter(x, y, color=t["pt"], edgecolor=t["surface"], zorder=3, label="Data (b)")
        ax1.plot(x[order], b_hat[order], color=t["line"], linewidth=2, label="Fitted line (b\u0302)")
        ax1.set_title("Projection onto Col(A)", color=t["text"])
        ax1.set_xlabel("x", color=t["text"])
        ax1.set_ylabel("y", color=t["text"])
        ax1.legend(frameon=False, labelcolor=t["text"])
        marker, stems, base = ax2.stem(x, resid)
        marker.set_color(t["line"])
        for artist in (stems if isinstance(stems, (list, tuple)) else [stems]):
            artist.set_color(t["line"])
        base.set_color(t["muted"])
        ax2.set_title(r"Residuals ($e = b - \hat{b}$)", color=t["text"])
        ax2.set_xlabel("x", color=t["text"])
        ax2.set_ylabel("error", color=t["text"])
        fig.tight_layout()
        self.canvas.draw_idle()


# =======================================================================
# MAIN WINDOW
# =======================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theme = "light"
        self.setWindowTitle("Linear Algebra Mini-Project - PES University")
        self.resize(1320, 840)
        self.setMinimumSize(1040, 700)

        root = QWidget()
        root.setObjectName("root")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(240)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 24, 0, 18)
        sl.setSpacing(2)
        brand = QLabel("Linear Algebra")
        brand.setObjectName("brand")
        brand_sub = QLabel("Mini-project, PES University")
        brand_sub.setObjectName("brandSub")
        for w in (brand, brand_sub):
            w.setContentsMargins(20, 0, 20, 0)
            sl.addWidget(w)
        sl.addSpacing(22)

        self.stack = QStackedWidget()
        self.data_page = DataPage(self)
        self.direct_page = DirectPage(self)
        self.structure_page = StructurePage(self)
        self.ls_page = LeastSquaresPage(self)
        pages = [self.data_page, self.direct_page, self.structure_page, self.ls_page]

        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, page in enumerate(pages):
            btn = QPushButton(page.title)
            btn.setProperty("nav", True)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=i: self.stack.setCurrentIndex(k))
            group.addButton(btn)
            sl.addWidget(btn)
            self.stack.addWidget(page)
            if i == 0:
                btn.setChecked(True)
        self.nav_group = group

        sl.addStretch()
        actions = QVBoxLayout()
        actions.setContentsMargins(16, 0, 16, 0)
        actions.setSpacing(8)
        run_all = QPushButton("Run all stages")
        run_all.setObjectName("primary")
        run_all.setCursor(Qt.PointingHandCursor)
        run_all.clicked.connect(self.run_all)
        export = QPushButton("Export report")
        export.setProperty("side", True)
        export.setCursor(Qt.PointingHandCursor)
        export.clicked.connect(self.export_report)
        self.theme_btn = QPushButton("Switch to dark theme")
        self.theme_btn.setProperty("side", True)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.clicked.connect(self.toggle_theme)
        for w in (run_all, export, self.theme_btn):
            actions.addWidget(w)
        sl.addLayout(actions)
        sl.addSpacing(10)
        foot = QLabel("Built on NumPy and your own\nlinear algebra modules")
        foot.setObjectName("footer")
        foot.setContentsMargins(20, 0, 20, 0)
        sl.addWidget(foot)

        layout.addWidget(side)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready. Edit the system on the Data page, then run a stage.")
        self.apply_theme()

    # -- shared state --------------------------------------------------
    def system(self):
        return self.data_page.get_system()

    # -- actions -------------------------------------------------------
    def run_all(self):
        try:
            self.system()
        except ValueError as exc:
            QMessageBox.warning(self, "Check your input", str(exc))
            return
        problems = []
        for page in (self.direct_page, self.structure_page, self.ls_page):
            err = page.run(interactive=False)
            if err:
                problems.append(f"{page.title}: {err}")
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
            self.nav_group.buttons()[1].setChecked(True)
        if problems:
            QMessageBox.warning(self, "Some stages could not run", "\n\n".join(problems))
        else:
            self.statusBar().showMessage("All stages finished.")

    def export_report(self):
        sections = [(p.title, p.report_text()) for p in
                    (self.direct_page, self.structure_page, self.ls_page) if p.cards]
        if not sections:
            QMessageBox.information(self, "Nothing to export", "Run at least one stage first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export report", "linear_algebra_report.txt",
                                              "Text files (*.txt)")
        if not path:
            return
        lines = ["LINEAR ALGEBRA MINI-PROJECT - PES UNIVERSITY",
                 f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]
        try:
            A, b = self.system()
            lines += ["INPUT SYSTEM", "A =", np.array2string(A, precision=4),
                      "b =", np.array2string(b, precision=4), ""]
        except ValueError:
            pass
        for title, text in sections:
            lines += ["=" * 60, title.upper(), "=" * 60, text]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError as exc:
            QMessageBox.warning(self, "Cannot save", str(exc))
            return
        self.statusBar().showMessage(f"Report saved to {path}")

    def apply_theme(self):
        QApplication.instance().setStyleSheet(stylesheet(THEMES[self.theme]))
        self.theme_btn.setText("Switch to light theme" if self.theme == "dark" else "Switch to dark theme")
        self.ls_page.draw_plot()

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
