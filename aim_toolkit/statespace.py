"""
State-space estimation of Dynamic Nelson-Siegel: one-step Kalman MLE
and a shadow-rate extension via the unscented Kalman filter.

KalmanDNS  -- the textbook upgrade from Diebold-Li two-step to one-step:

    state:        f_t = c + A f_{t-1} + eta_t,   eta ~ N(0, diag(q^2))
    observation:  y_t = Lambda(lam) f_t + eps_t, eps ~ N(0, sig^2 I)

  All parameters (lam, c, A, q, sig) are estimated JOINTLY by maximizing
  the prediction-error-decomposition likelihood from the Kalman filter,
  initialized at the two-step estimates (which are consistent, so the
  optimizer starts close). Filtered + RTS-smoothed factors come out.

ShadowRateDNS -- Black (1995) at the yield level: observed yields are
  the shadow curve floored at a lower bound,

    y_t = max(Lambda f_t, lb) + eps_t.

  The max makes the observation nonlinear, so an UNSCENTED Kalman filter
  propagates sigma points through it. NOTE the honest caveat: proper
  shadow-rate models (Krippner, Wu-Xia) floor the SHORT RATE under Q and
  derive the whole curve; flooring yields directly is a pragmatic
  approximation that recovers a shadow-stance factor, not an
  arbitrage-free shadow term structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .yield_curve import DNSModel, ns_loadings


# ---------------------------------------------------------------- helpers
def _pack(lam, c, A, log_q, log_sig):
    return np.concatenate([[np.log(lam)], c, A.ravel(), log_q, [log_sig]])


def _unpack(theta, k=3):
    lam = float(np.exp(theta[0]))
    c = theta[1:1 + k]
    A = theta[1 + k:1 + k + k * k].reshape(k, k)
    log_q = theta[1 + k + k * k:1 + 2 * k + k * k]
    log_sig = float(theta[-1])
    return lam, c, A, np.exp(2 * log_q), np.exp(2 * log_sig)


def _init_state(c, A, Q):
    """Unconditional mean/covariance of the stationary VAR(1) state."""
    k = len(c)
    try:
        f0 = np.linalg.solve(np.eye(k) - A, c)
        vecP = np.linalg.solve(np.eye(k * k) - np.kron(A, A), Q.ravel())
        P0 = vecP.reshape(k, k)
        if not np.all(np.isfinite(P0)) or np.any(np.linalg.eigvalsh(
                (P0 + P0.T) / 2) < 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        f0, P0 = c.copy(), np.eye(k) * 10.0
    return f0, (P0 + P0.T) / 2


# ---------------------------------------------------------------- Kalman
class KalmanDNS:
    """One-step Kalman-filter MLE of the DNS state-space model."""

    def __init__(self, maxiter: int = 200):
        self.maxiter = maxiter

    # ---- likelihood
    @staticmethod
    def _filter(Y, mats, lam, c, A, q_diag, sig2, store=False):
        T, N = Y.shape
        k = len(c)
        Lam = ns_loadings(mats, lam)
        Q = np.diag(q_diag)
        R = sig2 * np.eye(N)
        f, P = _init_state(c, A, Q)
        ll = 0.0
        if store:
            F_filt = np.zeros((T, k)); P_filt = np.zeros((T, k, k))
            F_pred = np.zeros((T, k)); P_pred = np.zeros((T, k, k))
        for t in range(T):
            fp = c + A @ f
            Pp = A @ P @ A.T + Q
            v = Y[t] - Lam @ fp
            S = Lam @ Pp @ Lam.T + R
            try:
                Sinv_v = np.linalg.solve(S, v)
                _, logdet = np.linalg.slogdet(S)
            except np.linalg.LinAlgError:
                return -np.inf, None
            ll += -0.5 * (N * np.log(2 * np.pi) + logdet + v @ Sinv_v)
            K = Pp @ Lam.T @ np.linalg.solve(S, np.eye(N))
            f = fp + K @ v
            P = (np.eye(k) - K @ Lam) @ Pp
            P = (P + P.T) / 2
            if store:
                F_pred[t], P_pred[t] = fp, Pp
                F_filt[t], P_filt[t] = f, P
        if store:
            return ll, (F_filt, P_filt, F_pred, P_pred, Lam)
        return ll, None

    def fit(self, yields: pd.DataFrame) -> "KalmanDNS":
        Y = yields.to_numpy(dtype=float)
        mats = yields.columns.to_numpy(dtype=float)
        # two-step initialization (consistent -> optimizer starts close)
        two = DNSModel().fit(yields)
        q0 = np.sqrt(np.clip(np.diag(two.var.Sigma), 1e-6, None))
        resid = two.reconstruct(two.factors).to_numpy() - Y
        sig0 = max(resid.std(), 1e-4)
        x0 = _pack(two.lam, two.var.c, two.var.A, np.log(q0), np.log(sig0))
        self.loglik_init = self._filter(Y, mats, two.lam, two.var.c,
                                        two.var.A, q0 ** 2, sig0 ** 2)[0]

        def nll(theta):
            lam, c, A, qd, s2 = _unpack(theta)
            if not (0.05 < lam < 5.0) or np.max(np.abs(
                    np.linalg.eigvals(A))) > 0.9995:
                return 1e10
            ll, _ = self._filter(Y, mats, lam, c, A, qd, s2)
            return 1e10 if not np.isfinite(ll) else -ll

        res = minimize(nll, x0, method="L-BFGS-B",
                       options={"maxiter": self.maxiter, "maxfun": 20000})
        self.theta = res.x
        self.lam, c, A, qd, s2 = _unpack(res.x)
        self.c, self.A, self.q_diag, self.sig2 = c, A, qd, s2
        self.loglik = -res.fun
        ll, packs = self._filter(Y, mats, self.lam, c, A, qd, s2, store=True)
        F_filt, P_filt, F_pred, P_pred, Lam = packs
        # RTS smoother
        T, k = F_filt.shape
        F_sm = F_filt.copy()
        P_sm = P_filt.copy()
        for t in range(T - 2, -1, -1):
            G = P_filt[t] @ A.T @ np.linalg.solve(P_pred[t + 1], np.eye(k))
            F_sm[t] = F_filt[t] + G @ (F_sm[t + 1] - F_pred[t + 1])
            P_sm[t] = P_filt[t] + G @ (P_sm[t + 1] - P_pred[t + 1]) @ G.T
        self.factors = pd.DataFrame(F_sm, index=yields.index,
                                    columns=["level", "slope", "curvature"])
        fitted = self.factors.to_numpy() @ Lam.T
        self.fitted = pd.DataFrame(fitted, index=yields.index,
                                   columns=yields.columns)
        self.rmse_bp = float(np.sqrt(np.mean((fitted - Y) ** 2)) * 100)
        self.maturities = mats
        self.converged = bool(res.success)
        return self

    def forecast_curve(self, h: int) -> pd.DataFrame:
        f = self.factors.iloc[-1].to_numpy().copy()
        out = []
        for _ in range(h):
            f = self.c + self.A @ f
            out.append(f.copy())
        Lam = ns_loadings(self.maturities, self.lam)
        return pd.DataFrame(np.array(out) @ Lam.T,
                            columns=self.maturities,
                            index=pd.RangeIndex(1, h + 1, name="h"))


# ------------------------------------------------------------ shadow rate
class ShadowRateDNS:
    """Shadow-rate DNS via the unscented Kalman filter.

    Observation: y_t = max(Lambda f_t, lb) + eps. The shadow curve
    Lambda f_t can go below lb (e.g. 0); the filter infers how far below
    from the shape of the *uncensored* long end -- the classic
    'how negative is policy really' question at the ZLB.

    Transition parameters are taken from a linear KalmanDNS pre-fit
    (or two-step DNS); the UKF then re-extracts the states under the
    censored observation. Yield-level censoring is a documented
    simplification of Krippner/Wu-Xia (see module docstring).
    """

    def __init__(self, lb: float = 0.0, kappa: float = 0.0):
        self.lb, self.kappa = lb, kappa

    def fit(self, yields: pd.DataFrame,
            params: KalmanDNS | None = None) -> "ShadowRateDNS":
        Y = yields.to_numpy(dtype=float)
        mats = yields.columns.to_numpy(dtype=float)
        T, N = Y.shape
        if params is None:
            two = DNSModel().fit(yields)
            lam, c, A = two.lam, two.var.c, two.var.A
            qd = np.clip(np.diag(two.var.Sigma), 1e-6, None)
            resid = two.reconstruct(two.factors).to_numpy() - Y
            s2 = max(resid.std(), 1e-4) ** 2
        else:
            lam, c, A = params.lam, params.c, params.A
            qd, s2 = params.q_diag, params.sig2
        k = 3
        Lam = ns_loadings(mats, lam)
        Q, R = np.diag(qd), s2 * np.eye(N)
        # UKF weights (Julier), 2k+1 sigma points
        n_sig = 2 * k + 1
        w = np.full(n_sig, 1 / (2 * (k + self.kappa)))
        w[0] = self.kappa / (k + self.kappa) if self.kappa else 0.0
        if self.kappa == 0.0:                      # avoid zero-weight issue
            w = np.full(n_sig, 1 / n_sig)
        f, P = _init_state(c, A, Q)
        F_sh = np.zeros((T, k))
        for t in range(T):
            fp = c + A @ f
            Pp = A @ P @ A.T + Q
            # sigma points around the prediction
            sqrtP = np.linalg.cholesky((k + max(self.kappa, 1e-9))
                                       * ((Pp + Pp.T) / 2)
                                       + 1e-10 * np.eye(k))
            pts = np.vstack([fp, fp + sqrtP.T, fp - sqrtP.T])   # (2k+1, k)
            Ypts = np.maximum(pts @ Lam.T, self.lb)             # h(f)
            ybar = w @ Ypts
            dY = Ypts - ybar
            dX = pts - w @ pts
            S = dY.T @ (w[:, None] * dY) + R
            Cxy = dX.T @ (w[:, None] * dY)
            K = Cxy @ np.linalg.solve(S, np.eye(N))
            f = fp + K @ (Y[t] - ybar)
            P = Pp - K @ S @ K.T
            P = (P + P.T) / 2
            F_sh[t] = f
        self.lam, self.maturities = lam, mats
        self.factors = pd.DataFrame(
            F_sh, index=yields.index,
            columns=["level", "slope", "curvature"])
        shadow = self.factors.to_numpy() @ Lam.T
        self.shadow_curve = pd.DataFrame(shadow, index=yields.index,
                                         columns=yields.columns)
        self.fitted = self.shadow_curve.clip(lower=self.lb)
        self.rmse_bp = float(np.sqrt(np.mean(
            (self.fitted.to_numpy() - Y) ** 2)) * 100)
        return self

    def shadow_short_rate(self, tenor: float = 0.25) -> pd.Series:
        """Shadow yield at a short tenor -- can be NEGATIVE below the
        bound; the policy-stance measure the observed rate can't show."""
        Lam = ns_loadings(np.array([tenor]), self.lam)
        s = self.factors.to_numpy() @ Lam.T
        return pd.Series(s[:, 0], index=self.factors.index,
                         name=f"shadow_{tenor}y")
