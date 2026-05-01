
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import yfinance as yf
import time
from scipy.stats import t as student_t, jarque_bera, skew, kurtosis
from scipy.optimize import minimize
from scipy.special import gammaln
from datetime import datetime, date, timedelta

app = FastAPI(title="V-Lab Volatility API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CACHE = {}
CACHE_TTL = 3600


def download_data(ticker: str, start: str = "2021-01-01"):
    """Télécharge les données et retourne dates, prix, rendements log (%)."""
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"Aucune donnee pour {ticker}")
    
    # ── Extraire la colonne Close de manière robuste ──────────────────
    if isinstance(df.columns, pd.MultiIndex) if hasattr(df.columns, "levels") else False:
        close = df["Close"][ticker]
    else:
        close = df["Close"]
    
    # Aplatir si nécessaire
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    close = close.dropna()
    close = close.astype(float)
    
    n = len(close)
    if n < 50:
        raise ValueError(f"Pas assez de donnees ({n} observations)")
    
    dates  = [str(d.date()) for d in close.index]
    prices = [round(float(v), 2) for v in close.values]
    
    # Rendements log en %
    vals = close.values.astype(float)
    rets = np.log(vals[1:] / vals[:-1]) * 100.0
    rets = np.array([round(float(r), 6) for r in rets])
    
    return dates, prices, rets


def garch11_student_loglik(params, eps):
    """Log-vraisemblance négative GARCH(1,1) Student."""
    omega, alpha, beta, nu = (float(p) for p in params)
    
    # Contraintes
    if omega <= 0 or alpha <= 0 or beta <= 0:
        return 1e12
    if alpha + beta >= 0.9999:
        return 1e12
    if nu <= 2.01:
        return 1e12
    
    # Constante log-vraisemblance Student
    log_c = (gammaln((nu + 1) / 2)
             - gammaln(nu / 2)
             - 0.5 * np.log(np.pi * (nu - 2)))
    
    # Variance inconditionnelle comme point de départ
    sig2 = omega / (1.0 - alpha - beta)
    ll = 0.0
    
    for e in eps:
        sig2 = omega + alpha * e * e + beta * sig2
        if sig2 <= 0:
            return 1e12
        ll += log_c - 0.5 * np.log(sig2) - ((nu + 1) / 2) * np.log(
            1.0 + e * e / (sig2 * (nu - 2))
        )
    return -ll


def fit_garch11_student(returns: np.ndarray) -> dict:
    """Estime GARCH(1,1) Student par MLE avec plusieurs points de départ."""
    mu  = float(returns.mean())
    eps = returns - mu
    
    best_ll  = np.inf
    best_res = None
    
    # Plusieurs points de départ pour éviter les minima locaux
    starts = [
        [0.10, 0.05, 0.93, 5.0],
        [0.05, 0.08, 0.90, 6.0],
        [0.15, 0.10, 0.85, 4.0],
        [0.08, 0.06, 0.92, 7.0],
        [0.20, 0.12, 0.82, 5.5],
    ]
    
    bounds = [
        (1e-6, 5.0),    # omega
        (1e-6, 0.30),   # alpha
        (0.50, 0.9998), # beta  — force la persistance réaliste
        (2.10, 25.0),   # nu
    ]
    
    for x0 in starts:
        try:
            res = minimize(
                garch11_student_loglik,
                x0,
                args=(eps,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.success and res.fun < best_ll:
                best_ll  = res.fun
                best_res = res
        except Exception:
            continue
    
    if best_res is None:
        raise RuntimeError("Echec de l optimisation GARCH")
    
    omega, alpha, beta, nu = (float(x) for x in best_res.x)
    loglik = float(-best_res.fun)
    n, k   = len(eps), 5  # omega, alpha, beta, nu, mu
    
    return {
        "omega":    round(omega, 6),
        "alpha":    round(alpha, 6),
        "beta":     round(beta,  6),
        "nu":       round(nu,    4),
        "mu":       round(mu,    6),
        "loglik":   round(loglik, 3),
        "aic":      round(2 * k - 2 * loglik, 3),
        "bic":      round(k * float(np.log(n)) - 2 * loglik, 3),
        "converged": bool(best_res.success),
    }


def compute_conditional_series(returns: np.ndarray, params: dict):
    """Calcule la série des volatilités conditionnelles et des VaR."""
    omega = params["omega"]
    alpha = params["alpha"]
    beta  = params["beta"]
    nu    = params["nu"]
    mu    = params["mu"]
    
    q01 = float(student_t.ppf(0.01, df=nu))
    
    sig2 = omega / (1.0 - alpha - beta)
    eps  = returns - mu
    
    vols, vars_ = [], []
    for e in eps:
        sig2 = omega + alpha * float(e) ** 2 + beta * sig2
        sig2 = max(sig2, 1e-10)
        s    = float(np.sqrt(sig2))
        VaR  = float(-(mu + s * q01))
        vols.append(round(s, 6))
        vars_.append(round(VaR, 6))
    
    return vols, vars_, float(q01)


def volatility_term_structure(sig0: float, params: dict, horizons: list) -> list:
    """Term structure de la volatilité (retour vers la moyenne)."""
    omega = params["omega"]
    alpha = params["alpha"]
    beta  = params["beta"]
    sig_inf = float(np.sqrt(omega / (1.0 - alpha - beta)))
    result  = []
    for h in horizons:
        if h == 0:
            result.append(round(sig0, 6))
        else:
            v = float(np.sqrt(
                sig_inf ** 2 + (alpha + beta) ** h * (sig0 ** 2 - sig_inf ** 2)
            ))
            result.append(round(v, 6))
    return result


def news_impact_curve(params: dict, sig0: float, n_points: int = 200) -> dict:
    """News Impact Curve : σ_{t+1} en fonction de ε_t."""
    omega = params["omega"]
    alpha = params["alpha"]
    beta  = params["beta"]
    sig2_0 = sig0 ** 2
    
    xs = np.linspace(-10.0, 10.0, n_points)
    ys = [round(float(np.sqrt(omega + alpha * float(r) ** 2 + beta * sig2_0)), 6)
          for r in xs]
    return {
        "returns": [round(float(x), 3) for x in xs],
        "vols":    ys,
    }


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "V-Lab Volatility API — OK", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/volatility/{ticker}")
def get_volatility(
    ticker: str,
    start:  str  = Query("2021-01-01", description="Date de debut YYYY-MM-DD"),
    refit:  bool = Query(False,        description="Forcer la ré-estimation"),
):
    cache_key = f"{ticker}|{start}"
    now       = time.time()
    
    # Retourner le cache si valide
    if not refit and cache_key in CACHE:
        cached_at, data = CACHE[cache_key]
        if now - cached_at < CACHE_TTL:
            return JSONResponse(content=data)
    
    try:
        # 1. Données
        dates, prices, returns = download_data(ticker, start)
        n = len(returns)
        
        # 2. Estimation GARCH
        params = fit_garch11_student(returns)
        
        # 3. Séries conditionnelles
        vols, vars_, q01 = compute_conditional_series(returns, params)
        
        sig_today = float(vols[-1])
        var_today = float(vars_[-1])
        per       = round(float(params["alpha"] + params["beta"]), 4)
        sig_inf   = round(float(np.sqrt(params["omega"] / (1.0 - per))), 4)
        halflife  = round(float(np.log(0.5) / np.log(per)), 1) if per < 1 else 999
        
        # 4. Term structure
        horizons = [0, 1, 5, 10, 22, 66, 132, 252]
        labels   = ["Auj.", "1j", "1s", "10j", "1m", "3m", "6m", "1a"]
        forecast = volatility_term_structure(sig_today, params, horizons)
        
        # 5. NIC
        nic = news_impact_curve(params, sig_today)
        
        # 6. Stats descriptives
        jb_stat, jb_p = jarque_bera(returns)
        
        # 7. VaR par capital
        var_by_capital = {
            str(c): round(float(c) * abs(var_today) / 100.0, 2)
            for c in [10_000, 50_000, 100_000, 500_000, 1_000_000]
        }
        
        # 8. Séries allégées (tout garder sur 2 ans, 1/3 avant)
        cutoff = (date.today() - timedelta(days=730)).isoformat()
        idx2y  = next(
            (i for i, d in enumerate(dates[1:]) if d >= cutoff), 0
        )
        keep = sorted(
            set(list(range(0, max(idx2y, 1), 3)) + list(range(idx2y, n)))
        )
        
        def thin(lst, offset=0):
            return [float(lst[i + offset])
                    for i in keep if i + offset < len(lst)]
        
        result = {
            "ticker":      ticker,
            "last_date":   dates[-1],
            "last_price":  float(prices[-1]),
            "n_obs":       int(n + 1),
            "start_date":  dates[0],
            
            # Paramètres
            "params":      params,
            "q01":         round(float(q01), 4),
            "sig_inf":     sig_inf,
            "halflife":    halflife,
            "persistence": per,
            
            # Résultats clés
            "sig_today":   round(sig_today, 4),
            "var_today":   round(var_today, 4),
            "ann_vol":     round(float(returns.std() * np.sqrt(252)), 2),
            "var_by_capital": var_by_capital,
            
            # Term structure
            "forecast": {
                "horizons": horizons,
                "labels":   labels,
                "vols":     forecast,
            },
            
            # NIC
            "nic": nic,
            
            # Séries temporelles
            "series": {
                "dates":   [dates[i + 1] for i in keep if i + 1 < len(dates)],
                "prices":  thin(prices, 1),
                "returns": [float(returns[i]) for i in keep if i < n],
                "vols":    thin(vols),
                "vars":    thin(vars_),
            },
            
            # Stats descriptives
            "stats": {
                "mean":     round(float(returns.mean()), 6),
                "std":      round(float(returns.std()), 6),
                "skewness": round(float(skew(returns)), 4),
                "kurtosis": round(float(kurtosis(returns)), 4),
                "jb_stat":  round(float(jb_stat), 1),
                "jb_p":     round(float(jb_p), 6),
                "min":      round(float(returns.min()), 4),
                "max":      round(float(returns.max()), 4),
            },
            
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        CACHE[cache_key] = (now, result)
        return JSONResponse(content=result)
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
