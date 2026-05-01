
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import yfinance as yf
import time, random
from scipy.stats import t as student_t, jarque_bera, skew, kurtosis
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln
from datetime import datetime, date, timedelta

app = FastAPI(title="V-Lab Volatility API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

CACHE = {}
CACHE_TTL = 14400


def download_data(ticker, start="2021-01-01", retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep((2**attempt) + random.uniform(1, 3))
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True, timeout=30)
            if df is None or df.empty:
                raise ValueError(f"No data for {ticker}")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze().dropna().astype(float)
            if len(close) < 100:
                raise ValueError(f"Not enough data: {len(close)}")
            dates  = [str(d.date()) for d in close.index]
            prices = [round(float(v), 2) for v in close.values]
            vals   = close.values.astype(float)
            rets   = np.log(vals[1:] / vals[:-1]) * 100.0
            return dates, prices, rets.astype(float)
        except Exception as e:
            last_err = e
            if "Rate" in str(e) or "429" in str(e) or "Too Many" in str(e):
                continue
            raise
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def garch_filter(omega, alpha, beta, eps):
    """Filtre GARCH(1,1) — retourne la série des variances conditionnelles."""
    n    = len(eps)
    sig2 = np.empty(n)
    # Initialisation par la variance inconditionnelle
    sig2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t-1]**2 + beta * sig2[t-1]
        if sig2[t] <= 0:
            sig2[t] = 1e-8
    return sig2


def loglik_normal(omega, alpha, beta, eps):
    """Log-vraisemblance GARCH(1,1) Normale."""
    sig2 = garch_filter(omega, alpha, beta, eps)
    ll   = -0.5 * np.sum(np.log(2*np.pi * sig2) + eps**2 / sig2)
    return ll


def loglik_student(omega, alpha, beta, nu, eps):
    """Log-vraisemblance GARCH(1,1) Student-t."""
    sig2  = garch_filter(omega, alpha, beta, eps)
    log_c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*(nu-2))
    ll    = np.sum(log_c - 0.5*np.log(sig2)
                   - ((nu+1)/2) * np.log(1.0 + eps**2 / (sig2*(nu-2))))
    return ll


def neg_ll_student(params, eps):
    omega, alpha, beta, nu = params
    # Contraintes intégrées
    if omega <= 0 or alpha <= 0 or beta <= 0: return 1e12
    if alpha + beta >= 1.0: return 1e12
    if nu <= 2.0: return 1e12
    return -loglik_student(omega, alpha, beta, nu, eps)


def fit_garch(returns):
    """Estimation GARCH(1,1) Student par MLE en 2 étapes."""
    mu  = float(returns.mean())
    eps = (returns - mu).astype(float)
    var = float(eps.var())
    
    # ── Étape 1 : estimation gaussienne pour initialiser ──────────────
    def neg_ll_norm(p):
        omega, alpha, beta = p
        if omega<=0 or alpha<=0 or beta<=0 or alpha+beta>=1: return 1e12
        return -loglik_normal(omega, alpha, beta, eps)
    
    best_norm = None
    for x0 in [[var*0.05, 0.08, 0.90], [var*0.03, 0.05, 0.93],
                [var*0.10, 0.10, 0.87], [var*0.02, 0.04, 0.95]]:
        try:
            r = minimize(neg_ll_norm, x0, method="L-BFGS-B",
                         bounds=[(1e-8, var), (1e-6, 0.3), (0.5, 0.999)],
                         options={"maxiter":2000,"ftol":1e-12})
            if best_norm is None or r.fun < best_norm.fun:
                best_norm = r
        except: pass
    
    if best_norm is None:
        w0, a0, b0 = var*0.05, 0.08, 0.90
    else:
        w0, a0, b0 = best_norm.x
    
    # ── Étape 2 : estimation Student à partir des valeurs gaussiennes ──
    best_ll, best_res = np.inf, None
    
    # Points de départ basés sur l'étape 1
    starts = [
        [w0,       a0,       b0,       5.0],
        [w0,       a0,       b0,       6.0],
        [w0,       a0,       b0,       8.0],
        [w0*0.8,   a0*0.9,   b0*1.01,  5.0],
        [w0*1.2,   a0*1.1,   b0*0.99,  5.0],
        [w0*0.5,   a0*0.8,   b0*1.02,  4.0],
    ]
    
    # Bornes réalistes — nu libre jusqu'à 30
    bounds = [
        (1e-8,  var * 0.5),    # omega
        (1e-6,  0.20),         # alpha — max 20%
        (0.70,  0.9995),       # beta  — min 70% pour persistance réaliste
        (2.05,  30.0),         # nu
    ]
    
    for x0 in starts:
        # Clipper x0 dans les bornes
        x0_clipped = [np.clip(x0[i], bounds[i][0]+1e-9, bounds[i][1]-1e-9)
                      for i in range(4)]
        try:
            res = minimize(
                neg_ll_student, x0_clipped, args=(eps,),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter":5000, "ftol":1e-13, "gtol":1e-10, "eps":1e-8},
            )
            if res.fun < best_ll:
                best_ll  = res.fun
                best_res = res
        except: pass
    
    # Dernier recours : differential evolution (global optimizer)
    if best_res is None or best_res.fun > 1e10:
        try:
            de = differential_evolution(
                neg_ll_student, bounds=bounds, args=(eps,),
                seed=42, maxiter=300, tol=1e-8, workers=1,
            )
            if de.fun < best_ll:
                best_ll  = de.fun
                best_res = de
        except: pass
    
    if best_res is None:
        raise RuntimeError("GARCH optimization failed completely")
    
    omega, alpha, beta, nu = (float(x) for x in best_res.x)
    loglik = float(-best_res.fun)
    n, k   = len(eps), 5
    
    return {
        "omega":     round(omega,  6),
        "alpha":     round(alpha,  6),
        "beta":      round(beta,   6),
        "nu":        round(nu,     4),
        "mu":        round(mu,     6),
        "loglik":    round(loglik, 3),
        "aic":       round(2*k - 2*loglik, 3),
        "bic":       round(k*float(np.log(n)) - 2*loglik, 3),
        "converged": bool(best_res.success if hasattr(best_res,"success") else True),
    }


def compute_series(returns, params):
    w, a, b = params["omega"], params["alpha"], params["beta"]
    nu, mu  = params["nu"], params["mu"]
    q01     = float(student_t.ppf(0.01, df=nu))
    eps     = (returns - mu).astype(float)
    sig2    = garch_filter(w, a, b, eps)
    vols    = [round(float(np.sqrt(s)), 6) for s in sig2]
    vars_   = [round(float(-(mu + np.sqrt(s)*q01)), 6) for s in sig2]
    return vols, vars_, float(q01)


def term_structure(s0, params, horizons):
    w, a, b = params["omega"], params["alpha"], params["beta"]
    si      = float(np.sqrt(w / (1-a-b)))
    return [round(float(s0) if h==0
                  else float(np.sqrt(si**2 + (a+b)**h * (s0**2-si**2))), 6)
            for h in horizons]


def nic(params, s0):
    w, a, b = params["omega"], params["alpha"], params["beta"]
    xs = np.linspace(-10, 10, 200)
    return {
        "returns": [round(float(x),3) for x in xs],
        "vols":    [round(float(np.sqrt(w + a*float(r)**2 + b*s0**2)), 6) for r in xs],
    }


@app.get("/")
def root(): return {"message": "V-Lab API OK"}

@app.get("/api/health")
def health(): return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/volatility/{ticker}")
def get_vol(ticker:str, start:str=Query("2021-01-01"), refit:bool=Query(False)):
    key = f"{ticker}|{start}"
    now = time.time()
    if not refit and key in CACHE:
        t0, data = CACHE[key]
        if now - t0 < CACHE_TTL:
            return JSONResponse(content=data)
    try:
        dates, prices, R = download_data(ticker, start)
        n = len(R)
        params = fit_garch(R)
        vols, vars_, q01 = compute_series(R, params)
        s0  = float(vols[-1])
        v0  = float(vars_[-1])
        per = round(float(params["alpha"]+params["beta"]), 4)
        si  = round(float(np.sqrt(params["omega"]/(1-per))), 4)
        hl  = round(float(np.log(0.5)/np.log(per)), 1) if 0 < per < 1 else 999
        hz  = [0,1,5,10,22,66,132,252]
        lb  = ["Auj.","1j","1s","10j","1m","3m","6m","1a"]
        jbs, jbp = jarque_bera(R)
        vc  = {str(c): round(float(c)*abs(v0)/100,2)
               for c in [10000,50000,100000,500000,1000000]}
        cutoff = (date.today()-timedelta(days=730)).isoformat()
        idx2y  = next((i for i,d in enumerate(dates[1:]) if d>=cutoff), 0)
        keep   = sorted(set(list(range(0,max(idx2y,1),3))+list(range(idx2y,n))))
        def th(lst, off=0): return [float(lst[i+off]) for i in keep if i+off<len(lst)]
        result = {
            "ticker":      ticker,
            "last_date":   dates[-1],
            "last_price":  float(prices[-1]),
            "n_obs":       int(n+1),
            "start_date":  dates[0],
            "params":      params,
            "q01":         round(q01, 4),
            "sig_inf":     si,
            "halflife":    hl,
            "persistence": per,
            "sig_today":   round(s0, 4),
            "var_today":   round(v0, 4),
            "ann_vol":     round(float(R.std()*np.sqrt(252)), 2),
            "var_by_capital": vc,
            "forecast":    {"horizons":hz,"labels":lb,"vols":term_structure(s0,params,hz)},
            "nic":         nic(params, s0),
            "series": {
                "dates":   [dates[i+1] for i in keep if i+1<len(dates)],
                "prices":  th(prices, 1),
                "returns": [float(R[i]) for i in keep if i<n],
                "vols":    th(vols),
                "vars":    th(vars_),
            },
            "stats": {
                "mean":     round(float(R.mean()),6),
                "std":      round(float(R.std()),6),
                "skewness": round(float(skew(R)),4),
                "kurtosis": round(float(kurtosis(R)),4),
                "jb_stat":  round(float(jbs),1),
                "jb_p":     round(float(jbp),6),
                "min":      round(float(R.min()),4),
                "max":      round(float(R.max()),4),
            },
            "updated_at": datetime.utcnow().isoformat()+"Z",
        }
        CACHE[key] = (now, result)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
