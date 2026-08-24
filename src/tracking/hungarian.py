"""
hungarian.py — Optimal Detection-to-Track Assignment
=====================================================
Every tracking frame we must answer: "which of my existing tracks should
consume which of the new detections?"  This is the classic *assignment
problem*, solved optimally by the Hungarian algorithm (a.k.a. Kuhn-Munkres).

Given a cost matrix C where C[i, j] = cost of assigning row i to column j,
find the one-to-one assignment minimizing total cost.

For SORT:
    cost[i, j] = 1 - IoU(track_i, detection_j)
so minimizing cost == maximizing overlap.

How the algorithm works (conceptual, see Wikipedia walkthrough):
  1. Subtract each row's minimum from that row   -> every row has a 0
  2. Subtract each column's minimum              -> every column has a 0
  3. A set of zeros with no two sharing a row/column is a candidate optimal
     assignment (because potentials keep total cost provably minimal).
  4. If fewer than n zeros can be selected, adjust the "potentials"
     (u, v below) to create new zeros and repeat.

Our implementation uses the O(n^2 * m) shortest-augmenting-path variant
with dual potentials u/v — same result, much faster in practice than the
line-covering textbook formulation:

  - For each new row i, we grow a alternating tree until it finds an
    unassigned column, updating potentials along the way so reduced costs
    stay non-negative (this is what guarantees optimality).
  - When the tree reaches a free column, we flip the matched/unmatched
    edges along the path (an "augmenting step"), increasing the matching
    by exactly one.

Supports rectangular matrices: if rows > cols we solve the transposed
problem and swap the output back.
"""

import numpy as np

try:  # optional cross-check only — never required at runtime
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover
    _scipy_lsa = None


# ─── From-scratch implementation ───────────────────────────────────
def hungarian_algorithm(cost_matrix: np.ndarray):
    """
    Solve the rectangular linear assignment problem from scratch.

    Args:
        cost_matrix (ndarray): [N, M] cost matrix. NaN is not allowed;
            use a large finite value to forbid an assignment pair.

    Returns:
        (row_ind, col_ind):
            Arrays of length min(N, M). row_ind[k] is assigned col_ind[k].
            The sum cost_matrix[row_ind, col_ind].sum() is minimal.

    Reference:
        http://e-maxx.ru/algo/assignment_hungarian (1-indexed pseudocode)
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost_matrix must be 2-D, got shape {cost.shape}")
    if np.isnan(cost).any():
        raise ValueError("cost_matrix contains NaN; use a large finite value "
                         "to forbid pairs instead")

    # Work on the orientation where rows <= cols
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape  # guarantee: n <= m

    if n == 0 or m == 0:
        return (
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
        )

    INF = 1e18  # finite sentinel (avoids inf-inf = nan in potential updates)

    # Dual potentials: reduced cost of (row i, col j) is c[i,j] - u[i] - v[j]
    # Invariant: all reduced costs >= 0 throughout the algorithm.
    u = np.zeros(n + 1)          # indexed by rows      (1-based)
    v = np.zeros(m + 1)          # indexed by columns   (1-based)

    # p[j] = row currently assigned to column j (1-based; 0 = free column)
    p = np.zeros(m + 1, dtype=int)
    # way[j] = previous column on the alternating path that reached j
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):          # insert row i into the matching
        p[0] = i                       # start search from virtual column 0
        j0 = 0
        minv = np.full(m + 1, INF)     # min reduced cost seen per column
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1

            # Relax all edges from row i0 to unused columns
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Advance every potential so that column j1 becomes a new zero
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:             # reached a FREE column -> augment
                break

        # Walk back along `way` pointers, flipping assignments
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Read off the final assignment: row p[j] -> column j
    rows = [p[j] - 1 for j in range(1, m + 1) if p[j] != 0]
    cols = [j - 1 for j in range(1, m + 1) if p[j] != 0]
    row_ind = np.asarray(rows, dtype=int)
    col_ind = np.asarray(cols, dtype=int)

    if transposed:  # we solved C^T: swap roles back
        row_ind, col_ind = col_ind, row_ind

    return row_ind, col_ind


def assign(cost_matrix: np.ndarray,
           valid_mask: np.ndarray = None,
           invalid_cost: float = 1e6):
    """
    Convenience wrapper used by trackers: solve assignment while forbidding
    certain (track, detection) pairs.

    Strategy: replace forbidden entries with a huge-but-finite cost, run the
    Hungarian algorithm, then drop any returned pair whose original entry
    was forbidden. Because `invalid_cost` exceeds any realistic total of
    valid costs, the solver prefers leaving those rows/columns unassigned
    over pairing them illegally.

    Args:
        cost_matrix (ndarray): [N, M] raw costs
        valid_mask  (ndarray): [N, M] bool matrix, True = allowed pair.
                               None = everything allowed.
        invalid_cost (float): sentinel cost substituted for forbidden pairs

    Returns:
        (matches, unmatched_rows, unmatched_cols)
            matches         : list of (i, j) tuples, all originally valid
            unmatched_rows  : list of row indices with no partner
            unmatched_cols  : list of col indices with no partner
    """
    cost = np.asarray(cost_matrix, dtype=np.float64).copy()

    if valid_mask is None:
        valid_mask = np.ones(cost.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != cost.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} != cost shape {cost.shape}"
            )

    # Sentinel substitution (finite! NaN/inf would break the solver)
    cost[~valid_mask] = invalid_cost

    row_ind, col_ind = hungarian_algorithm(cost)

    matches, used_rows, used_cols = [], set(), set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if valid_mask[r, c]:
            matches.append((r, c))
            used_rows.add(r)
            used_cols.add(c)

    unmatched_rows = [i for i in range(cost.shape[0]) if i not in used_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in used_cols]
    return matches, unmatched_rows, unmatched_cols


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("Hungarian algorithm sanity tests")

    # Test 1: known toy example
    # Optimal: row0->col1 (cost 1), row1->col0 (cost 2); total = 3
    cost = np.array([[4., 1., 3.],
                     [2., 0., 5.]])
    rows, cols = hungarian_algorithm(cost)
    total = cost[rows, cols].sum()
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)], \
        f"toy example wrong: {list(zip(rows.tolist(), cols.tolist()))}"
    print(f"  Test 1 (toy example): total cost = {total} (expect 3.0)")
    assert total == 3.0

    # Test 2: brute-force comparison on random matrices (all sizes)
    import itertools

    def brute_force_min_cost(c):
        n, m = c.shape
        k = min(n, m)
        best = float("inf")
        for row_perm in itertools.permutations(range(max(n, m)), k):
            if n <= m:
                cand = sum(c[r, col] for r, col in enumerate(row_perm))
            else:
                cand = sum(c[row, col] for col, row in enumerate(row_perm))
            best = min(best, cand)
        return best

    for trial in range(50):
        n = int(rng.integers(1, 6))
        m = int(rng.integers(1, 6))
        c = rng.random((n, m)) * 10
        rows, cols = hungarian_algorithm(c)
        got = c[rows, cols].sum()
        want = brute_force_min_cost(c)
        assert abs(got - want) < 1e-9, \
            f"trial {trial}: got {got}, brute force {want} ({n}x{m})"
    print("  Test 2 (vs brute force, 50 random matrices): OK")

    # Test 3: cross-check against scipy when available
    if _scipy_lsa is not None:
        for trial in range(20):
            n = int(rng.integers(1, 40))
            m = int(rng.integers(1, 40))
            c = rng.random((n, m)) * 100
            rows, cols = hungarian_algorithm(c)
            sr, sc = _scipy_lsa(c)
            assert abs(c[rows, cols].sum() - c[sr, sc].sum()) < 1e-6, \
                f"scipy mismatch on {n}x{m}"
        print("  Test 3 (matches scipy on random matrices): OK")
    else:
        print("  Test 3 skipped (scipy not installed)")

    # Test 4: gating via assign() — forbid bad pairs
    cost = np.array([[0.9, 0.1],
                     [0.2, 0.8]])
    mask = np.array([[True, True],
                     [False, True]])  # row1 may NOT take col0
    matches, unr, unc = assign(cost, valid_mask=mask)
    assert (0, 1) in matches and (1, 1) not in matches, \
        f"forbidden pair was assigned: {matches}"
    assert 1 in unc, "row 1 should end unmatched"
    print(f"  Test 4 (gated assignment): matches={matches}, "
          f"unmatched rows={unr}")

    print("hungarian.py sanity checks passed!")
