"""Independent, fully observed, truncated-count baseline."""
from itertools import product
import math
import numpy as np
from scipy.optimize import minimize


def probabilities(eta):
    logw = np.arange(4, dtype=float) * float(eta) - np.array([math.lgamma(d + 1) for d in range(4)])
    w = np.exp(logw - logw.max())
    return w / w.sum()


def objective(theta, counts, x, ridge=0.0):
    """NLL and analytic gradient for independent observed counts."""
    theta = np.asarray(theta, dtype=float)
    counts, x = np.asarray(counts), np.asarray(x, dtype=float)
    if theta.shape != (2,) or counts.shape != x.shape or len(counts) == 0:
        raise ValueError("expected b,beta and aligned nonempty observations")
    if np.any(counts < 0) or np.any(counts > 3) or np.any(counts != np.floor(counts)) or ridge < 0:
        raise ValueError("counts must be in 0..3; ridge must be nonnegative")
    value = ridge * float(theta @ theta) / 2
    gradient = ridge * theta.copy()
    for d, z in zip(counts.astype(int), x):
        p = probabilities(theta[0] + theta[1] * z)
        value -= math.log(float(p[d]))
        gradient += (float(np.arange(4) @ p) - d) * np.array([1.0, z])
    return float(value), gradient


def fit(counts, x, initial=(0.0, 0.0), ridge=0.0):
    """Return the actual best feasible visited fit and optimizer stop."""
    best = [None, math.inf]
    def evaluate(theta):
        value, gradient = objective(theta, counts, x, ridge)
        if value < best[1]:
            best[:] = [np.asarray(theta).copy(), value]
        return value, gradient
    result = minimize(evaluate, np.asarray(initial, dtype=float), jac=True,
                      bounds=[(-2.0, 2.0)] * 2, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    return {"theta": best[0].tolist(), "objective": best[1],
            "success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit)}


def forecast(theta, future_x):
    """Complete joint table under this baseline's independent model."""
    b, beta = map(float, theta)
    rows = [probabilities(b + beta * float(z)) for z in future_x]
    return [{"path": list(path), "probability": math.prod(float(rows[t][d]) for t, d in enumerate(path))}
            for path in product(range(4), repeat=len(rows))]
