"""Single-period immediate-delivery newsvendor baseline."""
import math


def choose_order(probabilities, on_hand, max_order, purchase, lost, holding):
    p = list(map(float, probabilities))
    if len(p) != 4 or any(v < 0 for v in p) or not math.isclose(sum(p), 1.0, abs_tol=1e-9):
        raise ValueError("expected normalized count probabilities for 0..3")
    if any(v < 0 for v in (on_hand, max_order, purchase, lost, holding)):
        raise ValueError("nonnegative inventory and costs required")
    choices = []
    for q in range(max_order + 1):
        available = on_hand + q
        cost = purchase * q + sum(p[d] * (lost * max(0, d - available)
                    + holding * max(0, available - d)) for d in range(4))
        choices.append((cost, q))
    cost, q = min(choices)
    return {"order": q, "expected_cost": cost, "available": on_hand + q}
