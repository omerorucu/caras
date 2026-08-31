# -*- coding: utf-8 -*-
"""
CARAS core :: disagreement components
=====================================

Decomposition of total disagreement between a map and a reference into
interpretable components, as the recommended replacement for Cohen's kappa.

References
----------
Pontius, R.G. Jr. & Millones, M. (2011). Death to Kappa: birth of quantity
    disagreement and allocation disagreement for accuracy assessment.
    *International Journal of Remote Sensing*, 32(15), 4407-4429.
    https://doi.org/10.1080/01431161.2011.552923
Pontius, R.G. Jr. & Santacruz, A. (2014). Quantity, exchange and shift
    components of difference in a square contingency table.
    *International Journal of Remote Sensing*, 35(21), 7543-7554.
    https://doi.org/10.1080/2150704X.2014.969814

Definitions
-----------
Let ``p`` be the estimated area-proportion matrix with rows = map classes and
columns = reference classes, ``sum(p) == 1``.  Write ``p_g.`` for the row sum
and ``p_.g`` for the column sum of category ``g``.

Total disagreement
    ``D = 1 - OA = sum_{g != h} p_gh``

Quantity disagreement
    the part of ``D`` caused by a difference in the *amount* of each
    category::

        q_g = |p_g. - p_.g|          Q = (1/2) sum_g q_g

Allocation disagreement
    the remaining part, caused by a difference in the spatial *placement*
    of categories that are otherwise present in matching amounts::

        a_g = 2 * min(p_g. - p_gg, p_.g - p_gg)      A = (1/2) sum_g a_g

    and, identically, ``A = D - Q``.

Exchange and shift (Pontius & Santacruz 2014) split ``A`` further:

    Exchange
        pairwise swaps - category g is mapped where h belongs and h is mapped
        where g belongs::

            E = sum_{g < h} 2 * min(p_gh, p_hg)

    Shift
        the residual, non-reciprocal misallocation::

            S = A - E

All four components are expressed on the same scale as ``1 - OA`` so they add
up exactly: ``Q + E + S = D``.
"""

from __future__ import division

import numpy as np

__all__ = ["Disagreement", "disagreement"]


class Disagreement(object):
    """Container for the disagreement decomposition."""

    def __init__(self, categories, labels, total, quantity, allocation,
                 exchange, shift, per_category):
        self.categories = list(categories)
        self.labels = list(labels)
        self.total = float(total)
        self.quantity = float(quantity)
        self.allocation = float(allocation)
        self.exchange = float(exchange)
        self.shift = float(shift)
        self.per_category = per_category      # list of dicts

    @property
    def overall_agreement(self):
        return 1.0 - self.total

    def to_dict(self):
        return {
            "total_disagreement": self.total,
            "overall_agreement": self.overall_agreement,
            "quantity_disagreement": self.quantity,
            "allocation_disagreement": self.allocation,
            "exchange_component": self.exchange,
            "shift_component": self.shift,
            "per_category": self.per_category,
            "identity_check": {
                "quantity_plus_allocation": self.quantity + self.allocation,
                "exchange_plus_shift": self.exchange + self.shift,
                "closes": bool(
                    abs(self.quantity + self.allocation - self.total) < 1e-9
                    and abs(self.exchange + self.shift - self.allocation) < 1e-9),
            },
            "reference": (
                "Pontius & Millones (2011) IJRS 32:4407-4429; "
                "Pontius & Santacruz (2014) IJRS 35:7543-7554"),
        }


def disagreement(proportions, categories=None, labels=None):
    """Compute the quantity / allocation (and exchange / shift) components.

    ``proportions`` must be a square matrix of estimated *area* proportions
    with rows = map classes and columns = reference classes.  Under a
    stratified design this is the weighted matrix produced by
    :func:`caras.core.estimators.stratified_estimate`, never the raw counts -
    computing disagreement on raw stratified counts would inherit exactly the
    bias the weighting removes.
    """
    p = np.asarray(proportions, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("proportion matrix must be square")
    total_mass = p.sum()
    if total_mass <= 0:
        raise ValueError("proportion matrix is empty")
    p = p / total_mass

    k = p.shape[0]
    if categories is None:
        categories = list(range(1, k + 1))
    if labels is None:
        labels = ["Class %s" % c for c in categories]

    diag = np.diag(p)
    row = p.sum(axis=1)      # p_g.  (map)
    col = p.sum(axis=0)      # p_.g  (reference)

    oa = float(diag.sum())
    D = 1.0 - oa

    q_g = np.abs(row - col)
    Q = 0.5 * float(q_g.sum())

    a_g = 2.0 * np.minimum(row - diag, col - diag)
    A_from_sum = 0.5 * float(a_g.sum())
    A = D - Q                                  # authoritative (exact identity)

    # exchange
    e_g = np.zeros(k)
    E = 0.0
    for g in range(k):
        acc = 0.0
        for h in range(k):
            if h == g:
                continue
            acc += 2.0 * min(p[g, h], p[h, g])
        e_g[g] = acc
    E = 0.5 * float(e_g.sum())
    E = min(E, A)                              # guard against fp drift
    S = A - E

    per_category = []
    for g in range(k):
        per_category.append({
            "category": categories[g],
            "label": labels[g],
            "map_proportion": float(row[g]),
            "reference_proportion": float(col[g]),
            "quantity": float(q_g[g]),
            "allocation": float(max(a_g[g], 0.0)),
            "exchange": float(e_g[g]),
            "shift": float(max(a_g[g] - e_g[g], 0.0)),
        })

    obj = Disagreement(categories, labels, D, Q, A, E, S, per_category)
    obj.allocation_from_category_sum = A_from_sum
    return obj
